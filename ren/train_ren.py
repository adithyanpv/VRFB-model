"""
REN Training Script — VRFB SOC Estimator  (v3)
===============================================

IMPROVEMENTS OVER v2
---------------------

1. Stateful episode training  (most important)
   Problem: v2 always starts every chunk with z=0. At inference z carries
   forward continuously for 18,000 steps. This train/inference mismatch
   means the model never learns to use a warm hidden state during training,
   wasting the REN's memory capacity.

   Fix: EpisodeStatefulLoader processes episodes as ordered sequences.
   z is carried forward between consecutive chunks of the same episode
   using z.detach() (truncated BPTT boundary). Episodes are shuffled each
   epoch so ordering within the episode is preserved but episode order
   is not. Each epoch sees all data.

2. Warmup-step masking
   The first WARMUP_STEPS steps of each chunk (while z recovers from the
   chunk's starting state) do not contribute to the loss. This prevents
   systematically-high early errors from drowning out the signal from
   steps where z has meaningful context.

3. Linear LR warmup
   Avoids large early gradients destabilising the spectral-norm projection
   on A_free. LR ramps linearly over LR_WARMUP_EPOCHS, then cosine decays.

4. Boundary-weighted loss
   Errors near soc_min and soc_max are safety-critical (BMS protection
   triggers). An exponential boundary weight upweights those samples in
   the MSE component without affecting the MAE term.

5. Diagnostic logging
   Tracks z0 norm and contraction rate across epochs.

FEATURE COLS (9):
  voltage, current, temperature_stack, temperature_tank,
  flow_rate, soc_cc, I_limit, transport_ratio, soc_imbalance

OUTPUT FILES:
  ren/scaler.pkl          — StandardScaler fitted on training features only
  ren/ren_soc_best.pth    — best model (lowest val loss)
  ren/ren_soc_last.pth    — final epoch model
  ren/training_log.csv    — epoch metrics
  ren/training_curves.png — loss + RMSE + LR curves
"""

import os
import time
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from ren.ren_model import REN

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
HIDDEN_DIM      = 128
ALPHA           = 0.5      # sigma_max(A_bar) < 0.5  (was 0.95 → too tight)
DROPOUT         = 0.1
N_POWER_ITERS   = 10

# Features — must match dataset_gen.py v3 exactly
# Strictly physically measurable signals — no derived values, no CC output.
# This makes the comparison with Coulomb Counting honest:
# CC uses only current (integral of I*dt).
# REN uses voltage + current + temperature + flow — strictly more information.
FEATURE_COLS = [
    "voltage",            # stack terminal voltage [V]   — Nernst equation encodes SOC
    "current",            # stack current [A]            — same signal CC integrates
    "temperature_stack",  # stack thermocouple [K]       — affects OCV via RT/nF
    "temperature_tank",   # tank thermocouple [K]        — thermal lag dynamics
    "flow_rate",          # electrolyte flow [m3/s]      — mass transport conditions
]
TARGET_COL = "SOC_true"
INPUT_DIM  = len(FEATURE_COLS)   # 5

# Training
SEQ_LEN          = 256    # steps per BPTT chunk
BATCH_SIZE       = 32     # episodes processed in parallel per gradient step
                           # (lower than v2 because each episode yields many chunks)
EPOCHS           = 80
LR               = 3e-4
LR_WARMUP_EPOCHS = 5      # linear LR ramp-up before cosine decay
WEIGHT_DECAY     = 1e-5
PATIENCE         = 15     # increased from 12 (stateful training converges slower)
GRAD_CLIP        = 1.0
WARMUP_STEPS     = 32     # steps per chunk excluded from loss (z settling)
MSE_WEIGHT       = 0.7
MAE_WEIGHT       = 0.3
BOUNDARY_SCALE   = 3.0    # extra MSE weight multiplier near soc_min/soc_max
SOC_MIN          = 0.05   # BMS protection limits (must match config.py)
SOC_MAX          = 0.95

# Paths
TRAIN_CSV = "datasets/vrfb_train.csv"
TEST_CSV  = "datasets/vrfb_test.csv"
SAVE_DIR  = "ren"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATA LOADING — stateful episode format
# =============================================================================

def load_episodes(
    df:          pd.DataFrame,
    scaler:      StandardScaler,
    fit_scaler:  bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns a list of (X_ep, y_ep) tensors, one per episode, spanning
    the full episode length.  Chunking is done during the training loop
    so that z can be carried forward between chunks.

    X_ep : (episode_len, input_dim)  float32
    y_ep : (episode_len, 1)          float32
    """
    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_all = df[TARGET_COL].values.astype(np.float32)

    if fit_scaler:
        scaler.fit(X_all)
    X_scaled = scaler.transform(X_all).astype(np.float32)

    episodes = []
    for _, group in df.groupby("episode_id", sort=True):
        idx  = group.index
        X_ep = torch.tensor(X_scaled[idx], dtype=torch.float32)
        y_ep = torch.tensor(y_all[idx],    dtype=torch.float32).unsqueeze(-1)
        episodes.append((X_ep, y_ep))

    return episodes


# =============================================================================
# LOSS
# =============================================================================

def composite_loss(
    pred:   torch.Tensor,   # (batch, seq, 1)
    target: torch.Tensor,   # (batch, seq, 1)
) -> torch.Tensor:
    """
    Boundary-weighted MSE + MAE.

    The MSE component upweights samples where the true SOC is close to
    soc_min or soc_max.  These are the safety-critical regions where BMS
    protection triggers; an error of 0.01 near soc_min can cause an
    unwanted shutdown or over-discharge.

    Weight function: 1 + BOUNDARY_SCALE * (exp(-50*(y-soc_min)) +
                                            exp(-50*(soc_max-y)))
    This is ~1.0 in the mid-SOC range and rises to ~(1+BOUNDARY_SCALE)
    within ~0.1 SOC units of either limit.
    """
    # Boundary weight (no gradient through target)
    with torch.no_grad():
        w = 1.0 + BOUNDARY_SCALE * (
            torch.exp(-50.0 * (target - SOC_MIN).clamp(min=0.0)) +
            torch.exp(-50.0 * (SOC_MAX - target).clamp(min=0.0))
        )

    sq_err      = (pred - target) ** 2
    weighted_mse = (w * sq_err).mean()
    mae          = F.l1_loss(pred, target)

    return MSE_WEIGHT * weighted_mse + MAE_WEIGHT * mae


# =============================================================================
# TRAINING — stateful episode loop
# =============================================================================

def run_stateful_epoch(
    model:        REN,
    episodes:     list,
    optimiser:    torch.optim.Optimizer | None,
    warmup_steps: int,
) -> tuple[float, float, float, float]:
    """
    Stateful episode training epoch.

    For each episode:
      - z starts from model.z0 (learned initial state)
      - Chunks are processed in order; z carries forward with .detach()
        between chunks (truncated BPTT — gradient does not flow across
        chunk boundaries, preventing vanishing gradients over 18,000 steps)
      - Loss is computed only on steps [warmup_steps:] of each chunk

    Episodes are grouped into mini-batches of BATCH_SIZE for parallel
    processing.  Episode order within a batch is consistent chunk-by-chunk
    so z aligns correctly between chunks.  Episode groups are shuffled
    each epoch.
    """
    is_train = optimiser is not None
    model.train() if is_train else model.eval()

    # Shuffle episode order each epoch
    perm = torch.randperm(len(episodes)).tolist()

    total_loss  = 0.0
    sq_errors   = []
    abs_errors  = []
    n_batches   = 0

    ctx = torch.enable_grad if is_train else torch.no_grad

    for batch_start in range(0, len(perm), BATCH_SIZE):
        batch_idxs = perm[batch_start : batch_start + BATCH_SIZE]
        batch_eps  = [episodes[i] for i in batch_idxs]
        B          = len(batch_eps)

        # Minimum episode length in this batch (safe chunking)
        min_len  = min(ep[0].shape[0] for ep in batch_eps)
        n_chunks = min_len // SEQ_LEN
        if n_chunks == 0:
            continue

        # Initialise z from learned z0
        z = model.z0.expand(B, -1).contiguous().to(DEVICE)
        if is_train:
            z = z.detach()

        batch_loss_sum = 0.0

        for chunk_idx in range(n_chunks):
            s = chunk_idx * SEQ_LEN
            e = s + SEQ_LEN

            X_chunk = torch.stack(
                [ep[0][s:e] for ep in batch_eps]
            ).to(DEVICE)   # (B, SEQ_LEN, input_dim)

            y_chunk = torch.stack(
                [ep[1][s:e] for ep in batch_eps]
            ).to(DEVICE)   # (B, SEQ_LEN, 1)

            with ctx():
                y_pred, z_next = model(X_chunk, z=z)

                # Warmup masking: skip first warmup_steps from loss
                if warmup_steps > 0 and SEQ_LEN > warmup_steps:
                    y_pred_l  = y_pred[:, warmup_steps:, :]
                    y_chunk_l = y_chunk[:, warmup_steps:, :]
                else:
                    y_pred_l  = y_pred
                    y_chunk_l = y_chunk

                loss = composite_loss(y_pred_l, y_chunk_l)
                batch_loss_sum += loss.item()

                err = (y_pred_l - y_chunk_l).detach().cpu().numpy().ravel()
                sq_errors.append(err ** 2)
                abs_errors.append(np.abs(err))

                if is_train:
                    optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimiser.step()

            # Carry z forward — detach to truncate BPTT at chunk boundary
            z = z_next.detach()

        total_loss += batch_loss_sum / max(n_chunks, 1)
        n_batches  += 1

    sq_all  = np.concatenate(sq_errors)
    abs_all = np.concatenate(abs_errors)

    return (
        total_loss / max(n_batches, 1),
        math.sqrt(sq_all.mean()),
        abs_all.mean(),
        abs_all.max(),
    )


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr(epoch: int, optimiser: torch.optim.Optimizer) -> float:
    return optimiser.param_groups[0]["lr"]


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for LR_WARMUP_EPOCHS epochs, then cosine annealing
    to LR * 0.01.  Avoids destabilising the spectral-norm projection
    on A_free with large early gradients.
    """

    def __init__(self, optimiser, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr_ratio  = min_lr_ratio
        super().__init__(optimiser)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  REN Training  —  VRFB SOC Estimator  (v3, stateful)")
    print("=" * 65)
    print(f"  Device         : {DEVICE}")
    print(f"  Hidden dim     : {HIDDEN_DIM}   Alpha : {ALPHA}   Dropout : {DROPOUT}")
    print(f"  Power iters    : {N_POWER_ITERS}  (spectral norm)")
    print(f"  Seq len        : {SEQ_LEN}     Batch : {BATCH_SIZE}   Epochs : {EPOCHS}")
    print(f"  LR             : {LR}   Warmup : {LR_WARMUP_EPOCHS} epochs   Patience : {PATIENCE}")
    print(f"  Loss           : {MSE_WEIGHT}xMSE(boundary-weighted) + {MAE_WEIGHT}xMAE")
    print(f"  Warmup steps   : {WARMUP_STEPS}  (masked from loss per chunk)")
    print(f"  Training mode  : STATEFUL (z carried between chunks per episode)")
    print(f"  Features ({INPUT_DIM})   : {FEATURE_COLS}")
    print("-" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    missing = [c for c in FEATURE_COLS + [TARGET_COL, "episode_id"]
               if c not in df_train.columns]
    if missing:
        raise ValueError(
            f"Missing columns in training CSV: {missing}\n"
            f"Available: {list(df_train.columns)}\n"
            f"Did you run dataset_gen.py v3?"
        )

    print(f"  Train : {len(df_train):>9,} rows  "
          f"({df_train['episode_id'].nunique()} episodes)")
    print(f"  Test  : {len(df_test):>9,} rows  "
          f"({df_test['episode_id'].nunique()} episodes)")

    # ── Scaler (fit on train features only) ──────────────────────────────
    scaler       = StandardScaler()
    train_eps    = load_episodes(df_train, scaler, fit_scaler=True)
    test_eps     = load_episodes(df_test,  scaler, fit_scaler=False)

    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Scaler saved -> {scaler_path}")
    print(f"  Train episodes : {len(train_eps)}   "
          f"chunks/ep : ~{train_eps[0][0].shape[0] // SEQ_LEN}")
    print(f"  Test  episodes : {len(test_eps)}")

    # ── CC baseline on test set ───────────────────────────────────────────
    cc_errors = np.abs(df_test[TARGET_COL].values - df_test["soc_cc"].values)
    cc_rmse   = math.sqrt(np.mean(cc_errors ** 2))
    cc_mae    = cc_errors.mean()
    cc_max    = cc_errors.max()

    # ── Model ─────────────────────────────────────────────────────────────
    model = REN(
        input_dim     = INPUT_DIM,
        hidden_dim    = HIDDEN_DIM,
        output_dim    = 1,
        alpha         = ALPHA,
        dropout       = DROPOUT,
        n_power_iters = N_POWER_ITERS,
    ).to(DEVICE)

    n_params = model.count_parameters()
    init_cr  = model.contraction_rate()
    print(f"\n  Parameters       : {n_params:,}")
    print(f"  Contraction rate : {init_cr:.4f}  (enforced < {1-ALPHA:.2f})")
    print(f"  z0 norm (init)   : {model.z0_norm():.4f}")

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = WarmupCosineScheduler(
        optimiser,
        warmup_epochs = LR_WARMUP_EPOCHS,
        total_epochs  = EPOCHS,
        min_lr_ratio  = 0.01,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n  {'Ep':>4}  {'Trn Loss':>10}  {'Val Loss':>10}  "
          f"{'Val RMSE':>10}  {'Val MAE':>9}  {'MaxErr':>8}  "
          f"{'CR':>7}  {'z0‖':>6}  {'LR':>9}")
    print("  " + "-" * 88)

    best_val_loss  = math.inf
    patience_count = 0
    log            = []
    t_losses, v_losses, v_rmses, lrs_list = [], [], [], []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()

        trn_loss, trn_rmse, trn_mae, _ = run_stateful_epoch(
            model, train_eps, optimiser, WARMUP_STEPS
        )
        val_loss, val_rmse, val_mae, val_max = run_stateful_epoch(
            model, test_eps, None, WARMUP_STEPS
        )
        scheduler.step()

        cr         = model.contraction_rate()
        z0_n       = model.z0_norm()
        current_lr = get_lr(epoch, optimiser)
        ep_secs    = time.time() - t_ep

        t_losses.append(trn_loss)
        v_losses.append(val_loss)
        v_rmses.append(val_rmse)
        lrs_list.append(current_lr)

        log.append(dict(
            epoch=epoch, trn_loss=trn_loss, val_loss=val_loss,
            val_rmse=val_rmse, val_mae=val_mae, val_max=val_max,
            contraction_rate=cr, z0_norm=z0_n, lr=current_lr
        ))

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "ren_soc_best.pth"))
            marker = " ◀"
        else:
            patience_count += 1

        print(f"  {epoch:>4d}  {trn_loss:>10.6f}  {val_loss:>10.6f}  "
              f"{val_rmse:>10.5f}  {val_mae:>9.5f}  {val_max:>8.5f}  "
              f"{cr:>7.4f}  {z0_n:>6.3f}  {current_lr:>9.2e}"
              f"  [{ep_secs:.0f}s]{marker}")

        if patience_count >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    # ── Save last model + log ─────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ren_soc_last.pth"))
    pd.DataFrame(log).to_csv(
        os.path.join(SAVE_DIR, "training_log.csv"), index=False
    )

    # ── Final metrics using best model ────────────────────────────────────
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "ren_soc_best.pth"),
                   map_location=DEVICE)
    )
    _, ren_rmse, ren_mae, ren_max = run_stateful_epoch(
        model, test_eps, None, WARMUP_STEPS
    )

    print(f"\n{'=' * 65}")
    print(f"  FINAL RESULTS — best model vs Coulomb Counter baseline")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<16} {'REN':>12}  {'CC':>12}  {'Improvement':>12}")
    print(f"  {'-' * 56}")
    print(f"  {'RMSE':<16} {ren_rmse:>12.5f}  {cc_rmse:>12.5f}  "
          f"{(1 - ren_rmse / cc_rmse) * 100:>10.1f}%")
    print(f"  {'MAE':<16} {ren_mae:>12.5f}  {cc_mae:>12.5f}  "
          f"{(1 - ren_mae / cc_mae) * 100:>10.1f}%")
    print(f"  {'Max Error':<16} {ren_max:>12.5f}  {cc_max:>12.5f}  "
          f"{(1 - ren_max / cc_max) * 100:>10.1f}%")
    print(f"\n  Total training time : {(time.time() - t0) / 60:.1f} min")
    print(f"  Best val loss       : {best_val_loss:.6f}")
    print(f"  Final contraction   : {model.contraction_rate():.4f}  "
          f"(target < {1-ALPHA:.2f})")
    print(f"  Final z0 norm       : {model.z0_norm():.4f}")
    print(f"{'=' * 65}")

    # ── Training curves ───────────────────────────────────────────────────
    epochs_x = list(range(1, len(t_losses) + 1))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(epochs_x, t_losses, label="Train", color="steelblue", lw=1.5)
    axes[0].plot(epochs_x, v_losses, label="Val",   color="tomato",    lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs_x, v_rmses, color="purple", lw=2, label="REN Val RMSE")
    axes[1].axhline(cc_rmse, color="orange", ls="--", lw=1.5,
                    label=f"CC RMSE = {cc_rmse:.4f}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE")
    axes[1].set_title("Val RMSE vs CC Baseline")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    axes[2].plot(epochs_x, lrs_list, color="green", lw=1.5)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title(f"LR Schedule (warmup {LR_WARMUP_EPOCHS} ep)")
    axes[2].grid(True, alpha=0.4)

    cr_vals = [row["contraction_rate"] for row in log]
    z0_vals = [row["z0_norm"]          for row in log]
    ax4a = axes[3]
    ax4b = ax4a.twinx()
    ax4a.plot(epochs_x, cr_vals, color="coral",  lw=1.5, label="σ_max(A_bar)")
    ax4a.axhline(1.0 - ALPHA, color="coral", ls="--", lw=1,
                 label=f"limit {1-ALPHA:.2f}")
    ax4b.plot(epochs_x, z0_vals, color="teal",   lw=1.5, ls=":", label="‖z0‖")
    ax4a.set_xlabel("Epoch")
    ax4a.set_ylabel("Contraction rate", color="coral")
    ax4b.set_ylabel("‖z0‖", color="teal")
    ax4a.set_title("Contractivity & z0 evolution")
    ax4a.grid(True, alpha=0.4)
    lines1, labels1 = ax4a.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4a.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.suptitle("REN Training v3 — VRFB SOC Estimator  (stateful BPTT)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved -> {fig_path}")
    print(f"  Saved -> {SAVE_DIR}/ren_soc_best.pth")
    print(f"  Saved -> {SAVE_DIR}/ren_soc_last.pth")
    print(f"  Saved -> {SAVE_DIR}/training_log.csv")


if __name__ == "__main__":
    main()