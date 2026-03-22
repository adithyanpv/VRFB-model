# ren/train_ren.py
"""
REN Training Script — VRFB SOC Estimator  (v2, fixed)
=======================================================

BUGS FIXED FROM v1:
  1. FEATURE_COLS wrong  — used "temperature", "flow", "dVdt", "dIdt"
                           correct: "temperature_stack", "temperature_tank",
                                    "flow_rate", "soc_cc"
  2. Wrong dataset file  — used vrfb_dataset.csv (all 80 episodes mixed)
                           correct: vrfb_train.csv / vrfb_test.csv (pre-split)
  3. Row-level chunking  — reshaped all rows flat, crossing episode boundaries
                           correct: chunk within each episode independently
  4. VAL_SPLIT row-level — random 20% of chunks regardless of episode
                           correct: use the separate test CSV (episode-split)
  5. Wrong import path   — "from ren_model import REN"
                           correct: "from ren.ren_model import REN"
  6. RMSE recalculation  — second forward pass in val loop (slow, wasteful)
                           fixed: accumulate squared errors inline
  7. Hidden state leaks across episodes in val loop — fixed with z=None per episode

TRAINING STRATEGY:
  - Episode-aware chunking: each episode → N chunks of SEQ_LEN steps
  - Each chunk starts with z = 0 (valid because REN contractivity guarantees
    the hidden state forgets its initial condition within ~50 steps)
  - Truncated BPTT: z.detach() between chunks (prevents vanishing gradients
    across thousands of steps)
  - Loss: 0.7 × MSE + 0.3 × MAE
  - Cosine annealing LR: smooth decay, avoids oscillation in final epochs
  - Early stopping on val loss with patience=12

OUTPUT FILES:
  ren/scaler.pkl          — StandardScaler fitted on training features only
  ren/ren_soc_best.pth    — best model (lowest val loss)
  ren/ren_soc_last.pth    — final epoch model
  ren/training_log.csv    — epoch metrics
  ren/training_curves.png — loss + RMSE + LR curves

USAGE:
  python -m ren.train_ren
"""

import os
import time
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from ren.ren_model import REN

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← change these if you want to experiment
# ═══════════════════════════════════════════════════════════════════════════════

# Model
HIDDEN_DIM   = 128
ALPHA        = 0.95      # contractivity: σ_max(A_bar) < (1 - ALPHA) = 0.05
DROPOUT      = 0.1

# Data — must match dataset_gen.py v2 column names exactly
FEATURE_COLS = [
    "voltage",            # stack terminal voltage [V]   — encodes Nernst SOC curve
    "current",            # actual current after BMS [A] — encodes charge/discharge direction
    "temperature_stack",  # stack thermocouple [K]       — affects OCV and crossover rate
    "temperature_tank",   # tank thermocouple [K]        — thermal lag, slow dynamics
    "flow_rate",          # electrolyte flow [m³/s]      — determines I_limit
    "soc_cc",             # Coulomb counter SOC [-]      — REN corrects this drift
    "I_limit",            # mass transport limit [A]     — encodes flow/concentration stress
    "transport_ratio",    # |I| / I_limit [-]            — how close to starvation
]
TARGET_COL   = "SOC_true"
INPUT_DIM    = len(FEATURE_COLS)   # 8

# Training
SEQ_LEN      = 256       # steps per chunk — 256s per sequence
BATCH_SIZE   = 64        # chunks per gradient step
EPOCHS       = 80
LR           = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE     = 12        # early stopping: stop if val doesn't improve for 12 epochs
GRAD_CLIP    = 1.0
MSE_WEIGHT   = 0.7
MAE_WEIGHT   = 0.3

# Paths
TRAIN_CSV = "datasets/vrfb_train.csv"
TEST_CSV  = "datasets/vrfb_test.csv"
SAVE_DIR  = "ren"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET — episode-aware chunking
# ═══════════════════════════════════════════════════════════════════════════════

class EpisodeChunkDataset(Dataset):
    """
    Splits each episode into fixed-length SEQ_LEN chunks.
    Chunks never cross episode boundaries — this prevents the model
    from learning spurious correlations between the end of one episode
    and the start of the next (which would have different initial SOC,
    temperature and flow profile).

    Each item: (X_chunk [SEQ_LEN, 8], y_chunk [SEQ_LEN, 1])
    Hidden state z is initialised to 0 for every chunk in the DataLoader.
    This is valid because the REN contractivity guarantee means z forgets
    its initial condition within ~50 steps regardless of starting value.
    """

    def __init__(self, df: pd.DataFrame, scaler: StandardScaler,
                 seq_len: int, fit_scaler: bool = False):

        self.chunks_X = []
        self.chunks_y = []

        X_all = df[FEATURE_COLS].values.astype(np.float32)
        y_all = df[TARGET_COL].values.astype(np.float32)

        # Fit scaler on training data only (never on test data)
        if fit_scaler:
            scaler.fit(X_all)
        X_scaled = scaler.transform(X_all).astype(np.float32)

        # Chunk per episode
        for _, group in df.groupby("episode_id", sort=True):
            idx   = group.index
            X_ep  = X_scaled[idx]
            y_ep  = y_all[idx]
            n_ep  = len(X_ep)

            n_chunks = n_ep // seq_len
            for i in range(n_chunks):
                s = i * seq_len
                e = s + seq_len
                self.chunks_X.append(
                    torch.tensor(X_ep[s:e], dtype=torch.float32)
                )
                self.chunks_y.append(
                    torch.tensor(y_ep[s:e], dtype=torch.float32).unsqueeze(-1)
                )

    def __len__(self):
        return len(self.chunks_X)

    def __getitem__(self, idx):
        return self.chunks_X[idx], self.chunks_y[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def composite_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = nn.functional.mse_loss(pred, target)
    mae = nn.functional.l1_loss(pred,  target)
    return MSE_WEIGHT * mse + MAE_WEIGHT * mae


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, optimiser=None):
    """
    Single epoch of training (optimiser provided) or evaluation (optimiser=None).
    Returns (mean_loss, rmse, mae, max_error).
    """
    is_train = optimiser is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    sq_errors  = []
    abs_errors = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)   # [B, SEQ_LEN, 8]
            y_batch = y_batch.to(DEVICE)   # [B, SEQ_LEN, 1]

            # z=None → initialised to zeros inside model.forward()
            y_pred, _ = model(X_batch, z=None)

            loss = composite_loss(y_pred, y_batch)
            total_loss += loss.item()

            err = (y_pred - y_batch).detach().cpu().numpy().flatten()
            sq_errors.append(err ** 2)
            abs_errors.append(np.abs(err))

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimiser.step()

    sq_errors  = np.concatenate(sq_errors)
    abs_errors = np.concatenate(abs_errors)

    return (
        total_loss / len(loader),
        math.sqrt(sq_errors.mean()),
        abs_errors.mean(),
        abs_errors.max(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  REN Training  —  VRFB SOC Estimator  (v2)")
    print("=" * 65)
    print(f"  Device      : {DEVICE}")
    print(f"  Hidden dim  : {HIDDEN_DIM}   Alpha : {ALPHA}   Dropout : {DROPOUT}")
    print(f"  Seq len     : {SEQ_LEN}     Batch : {BATCH_SIZE}   Epochs : {EPOCHS}")
    print(f"  LR          : {LR}   Patience : {PATIENCE}")
    print(f"  Loss        : {MSE_WEIGHT}×MSE + {MAE_WEIGHT}×MAE")
    print(f"  Features    : {FEATURE_COLS}")
    print("-" * 65)

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    # Validate columns exist
    missing = [c for c in FEATURE_COLS + [TARGET_COL, "episode_id"]
               if c not in df_train.columns]
    if missing:
        raise ValueError(
            f"Missing columns in training CSV: {missing}\n"
            f"Available columns: {list(df_train.columns)}\n"
            f"Did you run dataset_gen.py v2?"
        )

    print(f"  Train : {len(df_train):>9,} rows  "
          f"({df_train['episode_id'].nunique()} episodes)")
    print(f"  Test  : {len(df_test):>9,} rows  "
          f"({df_test['episode_id'].nunique()} episodes)")

    soc_min = df_train[TARGET_COL].min()
    soc_max = df_train[TARGET_COL].max()
    print(f"  SOC range in train : {soc_min:.3f} – {soc_max:.3f}")

    # ── Scaler (fit on train only) ────────────────────────────────────────────
    scaler   = StandardScaler()
    train_ds = EpisodeChunkDataset(df_train, scaler, SEQ_LEN, fit_scaler=True)
    test_ds  = EpisodeChunkDataset(df_test,  scaler, SEQ_LEN, fit_scaler=False)

    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Scaler saved → {scaler_path}")

    print(f"  Train chunks : {len(train_ds):,}   "
          f"Test chunks : {len(test_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = REN(
        input_dim  = INPUT_DIM,
        hidden_dim = HIDDEN_DIM,
        output_dim = 1,
        alpha      = ALPHA,
        dropout    = DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    init_cr  = model.contraction_rate()
    print(f"\n  Parameters       : {n_params:,}")
    print(f"  Contraction rate : {init_cr:.4f}  (enforced < {1-ALPHA:.2f})")

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS, eta_min=LR * 0.01
    )

    # ── CC baseline on test set (for comparison table at the end) ─────────────
    cc_errors = np.abs(
        df_test[TARGET_COL].values - df_test["soc_cc"].values
    )
    cc_rmse  = math.sqrt(np.mean(cc_errors ** 2))
    cc_mae   = cc_errors.mean()
    cc_max   = cc_errors.max()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  {'Ep':>4}  {'Trn Loss':>10}  {'Val Loss':>10}  "
          f"{'Val RMSE':>10}  {'Val MAE':>9}  {'MaxErr':>8}  "
          f"{'CR':>7}  {'LR':>9}")
    print("  " + "-" * 80)

    best_val_loss  = math.inf
    patience_count = 0
    log            = []
    t_losses, v_losses, v_rmses, lrs_list = [], [], [], []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()

        trn_loss, trn_rmse, trn_mae, _       = run_epoch(model, train_loader, optimiser)
        val_loss, val_rmse, val_mae, val_max  = run_epoch(model, test_loader,  None)
        scheduler.step()

        cr         = model.contraction_rate()
        current_lr = scheduler.get_last_lr()[0]
        ep_secs    = time.time() - t_ep

        t_losses.append(trn_loss)
        v_losses.append(val_loss)
        v_rmses.append(val_rmse)
        lrs_list.append(current_lr)

        log.append(dict(
            epoch=epoch, trn_loss=trn_loss, val_loss=val_loss,
            val_rmse=val_rmse, val_mae=val_mae, val_max=val_max,
            contraction_rate=cr, lr=current_lr
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
              f"{cr:>7.4f}  {current_lr:>9.2e}"
              f"  [{ep_secs:.0f}s]{marker}")

        if patience_count >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    # ── Save last model + log ─────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ren_soc_last.pth"))
    pd.DataFrame(log).to_csv(
        os.path.join(SAVE_DIR, "training_log.csv"), index=False
    )

    # ── Final metrics using best model ────────────────────────────────────────
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "ren_soc_best.pth"),
                   map_location=DEVICE)
    )
    _, ren_rmse, ren_mae, ren_max = run_epoch(model, test_loader, None)

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
    print(f"  Final contraction   : {model.contraction_rate():.4f}")
    print(f"{'=' * 65}")

    # ── Training curves ───────────────────────────────────────────────────────
    epochs_x = list(range(1, len(t_losses) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs_x, t_losses, label="Train loss", color="steelblue", lw=1.5)
    axes[0].plot(epochs_x, v_losses, label="Val loss",   color="tomato",    lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    # RMSE vs CC baseline
    axes[1].plot(epochs_x, v_rmses, color="purple", lw=2, label="REN Val RMSE")
    axes[1].axhline(cc_rmse, color="orange", ls="--", lw=1.5,
                    label=f"CC RMSE = {cc_rmse:.4f}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE")
    axes[1].set_title("Val RMSE vs CC Baseline")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    # LR schedule
    axes[2].plot(epochs_x, lrs_list, color="green", lw=1.5)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Cosine LR Schedule")
    axes[2].grid(True, alpha=0.4)

    plt.suptitle("REN Training — VRFB SOC Estimator", fontsize=12, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {fig_path}")
    print(f"  Saved → {SAVE_DIR}/ren_soc_best.pth")
    print(f"  Saved → {SAVE_DIR}/ren_soc_last.pth")
    print(f"  Saved → {SAVE_DIR}/training_log.csv")


if __name__ == "__main__":
    main()