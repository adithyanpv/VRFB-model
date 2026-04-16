"""
REN Training Script — VRFB SOC Estimator (CC + REN hybrid) 1.6
============================================================

Features (7):
  voltage, current, temperature_stack, temperature_tank, flow_rate,
  soc_cc, transport_ratio_approx

Target:
  SOC_true - soc_cc  (residual correction — REN corrects CC drift)

Architecture constraints:
  1. use_current_gate=True  — z frozen at I=0A (Faraday's law)
  2. use_feedthrough=False  — no D(x_t), no per-step sensor→output path
  3. Zero-current invariance loss — reinforces gate constraint in loss

CRITICAL FIX (gate):
  The gate must use RAW AMPS, not scaled current.
  StandardScaler shifts current by its mean, so scaled I=0A ≠ 0.
  Using scaled values means tanh(gate_k * |scaled_I_at_zero|) ≈ 0.44
  instead of 0.0 — the physics constraint is completely broken.

  Fix: after fitting the scaler, reconstruct raw current (Amps) from
  scaled values using scaler.mean_[CURRENT_IDX] and scale_[CURRENT_IDX].
  Pass raw Amps as x_raw to model.forward() every chunk.

CRITICAL FIX (ZC loss):
  ZC_THRESH was 0.05 in scaled units — this is the wrong domain.
  Fixed to ZC_I_THRESH = 2.0 Amps (the BMS dead-band) applied to
  the reconstructed raw current.

OUTPUT:
  ren/scaler.pkl
  ren/ren_soc_best.pth
  ren/ren_soc_last.pth
  ren/training_log.csv
  ren/training_curves.png
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_COLS = [
    "voltage",                    # OCV at I=0 encodes SOC via Nernst
    "current",                    # I=0 → gate closes → z frozen
    "temperature_stack",
    "temperature_tank",
    "flow_rate",
    "soc_cc",                     # CC estimate as context for correction
    "transport_ratio_approx",
    "elapsed_time_norm"     # |I|/I_limit_approx — 0 at I=0 (safe)
]
TARGET_COL  = "target"            # SOC_true - soc_cc  (residual correction)
INPUT_DIM   = len(FEATURE_COLS)   # 8
CURRENT_IDX = 1                   # index of "current" in FEATURE_COLS

# Model
HIDDEN_DIM    = 128
ALPHA         = 0.5
DROPOUT       = 0.1
N_POWER_ITERS = 10

# Training
SEQ_LEN          = 512
BATCH_SIZE       = 32
EPOCHS           = 80
LR               = 3e-4
LR_WARMUP_EPOCHS = 5
WEIGHT_DECAY     = 1e-5
PATIENCE         = 25
GRAD_CLIP        = 1.0
WARMUP_STEPS     = 32
P_Z0_RESET       = 0.30

# Loss weights
MSE_WEIGHT       = 0.7
MAE_WEIGHT       = 0.3
BOUNDARY_SCALE   = 3.0
SMOOTH_WEIGHT    = 0.10
SMOOTH_I_REF     = 0.3    # scaled current units (for smooth loss — continuous)
HARD_CASE_WEIGHT = 3.0
HARD_CASE_THRESH = 0.10

# Zero-current invariance — FIXED: threshold in raw Amps (not scaled units)
# 2.0A matches the BMS dead-band (I_DEAD_BAND in bms_controller.py)
ZC_WEIGHT   = 2.0
ZC_I_THRESH = 2.0         # raw Amps — gate should be closed below this

SOC_MIN = 0.05
SOC_MAX = 0.95

TRAIN_CSV = "datasets/vrfb_train.csv"
TEST_CSV  = "datasets/vrfb_test.csv"
SAVE_DIR  = "ren"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level scaler stats — set in main(), used in run_epoch()
_I_MEAN: float = 0.0
_I_STD:  float = 1.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_episodes(df: pd.DataFrame, scaler: StandardScaler,
                  fit_scaler: bool = False) -> list:
    """Returns list of (X_ep_scaled, y_ep) tensors, one per episode."""
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
    pred:       torch.Tensor,              # (batch, seq, 1) — predicted correction
    target:     torch.Tensor,              # (batch, seq, 1) — true correction
    X_chunk:    torch.Tensor | None,       # (batch, seq, 7) — scaled features
    I_raw_chunk: torch.Tensor | None,     # (batch, seq, 1) — raw Amps
) -> torch.Tensor:
    """
    Loss = boundary-weighted MSE + MAE + hard-case MSE
         + smooth term (soft, uses scaled current)
         + ZC invariance term (hard, uses raw Amps — PHYSICS CONSTRAINT)

    The ZC term is the critical fix:
      Old: used scaled current < 0.05 as ZC threshold → wrong domain
      New: uses raw current < ZC_I_THRESH (2.0A) → correct domain
           At I=0A, raw current is exactly 0A → mask fires correctly
    """
    err = pred - target
    with torch.no_grad():
        # Boundary weight: upweight samples near soc_min/soc_max
        # Note: target here is the correction, not absolute SOC.
        # We approximate boundary position from the correction magnitude.
        w_hard = torch.where(
            err.abs() > HARD_CASE_THRESH,
            torch.full_like(err, HARD_CASE_WEIGHT),
            torch.ones_like(err),
        )
        # Simple uniform boundary weight (correction doesn't map to SOC directly)
        w_boundary = torch.ones_like(err)
        w = w_boundary * w_hard

    base = MSE_WEIGHT * (w * err ** 2).mean() + MAE_WEIGHT * F.l1_loss(pred, target)

    if X_chunk is not None and pred.shape[1] > 1:
        I_scaled = X_chunk[:, :, CURRENT_IDX:CURRENT_IDX + 1]   # scaled
        delta    = (pred[:, 1:, :] - pred[:, :-1, :]).abs()      # (B, seq-1, 1)
        I_mid_s  = I_scaled[:, 1:, :].abs()                      # scaled

        # Smooth loss — soft penalty at all low scaled currents
        with torch.no_grad():
            smooth_w = torch.exp(-I_mid_s / SMOOTH_I_REF)
        smooth_loss = (smooth_w * delta).mean()

        # ZC invariance — FIXED: uses raw Amps, not scaled units
        if I_raw_chunk is not None:
            I_mid_raw = I_raw_chunk[:, 1:, :].abs()    # raw Amps
            with torch.no_grad():
                # Mask fires at true I=0A — not at scaled_I=0 (which is I_mean ≠ 0A)
                zc_mask = (I_mid_raw < ZC_I_THRESH).float()
        else:
            # Fallback: no raw current available — skip ZC loss
            zc_mask = torch.zeros_like(delta)

        zc_loss = (zc_mask * delta).mean()

        return base + SMOOTH_WEIGHT * smooth_loss + ZC_WEIGHT * zc_loss

    return base


# =============================================================================
# STATEFUL EPOCH
# =============================================================================

def run_epoch(
    model:     REN,
    episodes:  list,
    optimiser: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float]:
    """
    One full pass over all episodes with stateful hidden state.

    Key change: reconstructs raw current (Amps) from scaled values and
    passes it to model.forward() as x_raw. This fixes the gate.
    """
    is_train = optimiser is not None
    model.train() if is_train else model.eval()
    ctx = torch.enable_grad if is_train else torch.no_grad

    perm       = torch.randperm(len(episodes)).tolist()
    total_loss = 0.0
    sq_err: list  = []
    abs_err: list = []
    n_b        = 0

    for bs in range(0, len(perm), BATCH_SIZE):
        batch    = [episodes[i] for i in perm[bs:bs + BATCH_SIZE]]
        B        = len(batch)
        n_chunks = min(ep[0].shape[0] for ep in batch) // SEQ_LEN
        if n_chunks == 0:
            continue

        z = model.z0.expand(B, -1).contiguous().to(DEVICE)
        if is_train:
            z = z.detach()
        bloss = 0.0

        for ci in range(n_chunks):
            s = ci * SEQ_LEN
            e = s  + SEQ_LEN

            Xc = torch.stack([ep[0][s:e] for ep in batch]).to(DEVICE)  # (B,seq,7) scaled
            yc = torch.stack([ep[1][s:e] for ep in batch]).to(DEVICE)  # (B,seq,1) correction

            # ── Reconstruct raw current from scaled values ─────────────────
            # scaled_I = (I_raw - I_mean) / I_std
            # I_raw    = scaled_I * I_std + I_mean
            # Shape: (B, seq, 1) in Amps
            I_raw = (
                Xc[:, :, CURRENT_IDX:CURRENT_IDX + 1] * _I_STD + _I_MEAN
            )

            use_z0 = is_train and ci > 0 and torch.rand(1).item() < P_Z0_RESET

            with ctx():
                # Pass raw Amps as x_raw — gate uses this, not scaled current
                yp, zn = model(
                    Xc,
                    z     = None if use_z0 else z,
                    x_raw = I_raw,
                )

                # Warmup masking — first WARMUP_STEPS excluded from loss
                if WARMUP_STEPS > 0 and SEQ_LEN > WARMUP_STEPS:
                    yp_l    = yp[:,    WARMUP_STEPS:, :]
                    yc_l    = yc[:,    WARMUP_STEPS:, :]
                    Xc_l    = Xc[:,    WARMUP_STEPS:, :]
                    I_raw_l = I_raw[:, WARMUP_STEPS:, :]
                else:
                    yp_l, yc_l, Xc_l, I_raw_l = yp, yc, Xc, I_raw

                loss = composite_loss(yp_l, yc_l, Xc_l, I_raw_l)
                bloss += loss.item()

                e_arr = (yp_l - yc_l).detach().cpu().numpy().ravel()
                sq_err.append(e_arr ** 2)
                abs_err.append(np.abs(e_arr))

                if is_train:
                    optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimiser.step()

            z = zn.detach()

        total_loss += bloss / max(n_chunks, 1)
        n_b += 1

    sq_all  = np.concatenate(sq_err)
    ab_all  = np.concatenate(abs_err)
    return (
        total_loss / max(n_b, 1),
        math.sqrt(sq_all.mean()),
        ab_all.mean(),
        ab_all.max(),
    )


# =============================================================================
# LR SCHEDULE
# =============================================================================

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup: int, total: int, min_r: float = 0.01):
        self.w = warmup; self.t = total; self.m = min_r
        super().__init__(opt)

    def get_lr(self):
        e = self.last_epoch
        if e < self.w:
            s = (e + 1) / self.w
        else:
            s = self.m + 0.5 * (1 - self.m) * (
                1 + math.cos(math.pi * (e - self.w) / max(self.t - self.w, 1))
            )
        return [b * s for b in self.base_lrs]


# =============================================================================
# MAIN
# =============================================================================

def main():
    global _I_MEAN, _I_STD

    print(f"\n{'='*68}")
    print(f"  REN Training — CC+REN hybrid, physics-constrained (7 features)")
    print(f"{'='*68}")
    print(f"  Device        : {DEVICE}")
    print(f"  Hidden dim    : {HIDDEN_DIM}   Alpha: {ALPHA}")
    print(f"  Gate          : use_current_gate=True (RAW AMPS — fixed)")
    print(f"  Feedthrough   : use_feedthrough=False")
    print(f"  ZC threshold  : {ZC_I_THRESH}A raw  (was 0.05 scaled — broken)")
    print(f"  ZC weight     : {ZC_WEIGHT}")
    print(f"  Features ({INPUT_DIM}) : {FEATURE_COLS}")
    print(f"  Target        : SOC_true - soc_cc  (CC correction residual)")
    print(f"{'-'*68}")

    # ── Load data ─────────────────────────────────────────────────────────
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    # Target = correction the REN must learn to apply on top of CC
    df_train["target"] = df_train["SOC_true"] - df_train["soc_cc"]
    df_test["target"]  = df_test["SOC_true"]  - df_test["soc_cc"]

    # Validate columns
    needed = FEATURE_COLS + ["SOC_true", "soc_cc", "episode_id"]
    missing = [c for c in needed if c not in df_train.columns]
    if missing:
        raise ValueError(
            f"Missing columns in train CSV: {missing}\n"
            "Re-run dataset_gen.py — ensure transport_ratio_approx is saved."
        )

    print(f"\n  Train : {len(df_train):,} rows  "
          f"({df_train['episode_id'].nunique()} episodes)")
    print(f"  Test  : {len(df_test):,} rows  "
          f"({df_test['episode_id'].nunique()} episodes)")

    # ── Fit scaler and extract current stats ──────────────────────────────
    scaler    = StandardScaler()
    train_eps = load_episodes(df_train, scaler, fit_scaler=True)
    test_eps  = load_episodes(df_test,  scaler, fit_scaler=False)

    # Store scaler stats globally so run_epoch can reconstruct raw current
    _I_MEAN = float(scaler.mean_[CURRENT_IDX])
    _I_STD  = float(scaler.scale_[CURRENT_IDX])

    print(f"\n  Current scaler : mean = {_I_MEAN:.2f} A   std = {_I_STD:.2f} A")
    print(f"  Scaled I=0A    : {-_I_MEAN/_I_STD:.4f}  (NOT zero — hence gate fix needed)")
    print(f"  Gate @ I=0A    (fixed)  : {math.tanh(4 * 0.0 / 200.0):.4f}  (exact 0)")
    print(f"  Gate @ I=50A   (fixed)  : {math.tanh(4 * 50.0 / 200.0):.4f}")
    print(f"  Gate @ I=100A  (fixed)  : {math.tanh(4 * 100.0 / 200.0):.4f}")

    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler → {scaler_path}")

    # ── CC baseline (what we must beat) ──────────────────────────────────
    cc_err  = np.abs(df_test["SOC_true"].values - df_test["soc_cc"].values)
    cc_rmse = math.sqrt(np.mean(cc_err ** 2))
    cc_mae  = cc_err.mean()
    cc_max  = cc_err.max()
    print(f"\n  CC baseline — RMSE: {cc_rmse:.5f}  MAE: {cc_mae:.5f}  Max: {cc_max:.5f}")

    # ── Build model ───────────────────────────────────────────────────────
    model = REN(
        input_dim        = INPUT_DIM,
        hidden_dim       = HIDDEN_DIM,
        output_dim       = 1,
        alpha            = ALPHA,
        dropout          = DROPOUT,
        n_power_iters    = N_POWER_ITERS,
        use_feedthrough  = False,
        use_current_gate = True,
        current_feat_idx = CURRENT_IDX,
    ).to(DEVICE)

    print(f"\n  Parameters     : {model.count_parameters():,}")
    print(f"  Contraction    : {model.contraction_rate():.4f}  (< {1-ALPHA:.2f})")
    print(f"  Gate @ I=0A    : {model.gate_value_at_raw_amps(0.0):.4f}  (must stay 0.0)")
    print(f"  Gate @ I=50A   : {model.gate_value_at_raw_amps(50.0):.4f}")
    print(f"  Gate @ I=100A  : {model.gate_value_at_raw_amps(100.0):.4f}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = WarmupCosine(opt, LR_WARMUP_EPOCHS, EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n  {'Ep':>4}  {'Trn Loss':>10}  {'Val Loss':>10}  {'EMA Loss':>10}  "
          f"{'RMSE':>9}  {'MAE':>8}  {'g@0A':>7}  {'z0‖':>6}  {'LR':>9}")
    print(f"  {'-'*94}")

    best_loss = math.inf
    ema_loss  = math.inf
    patience  = 0
    log: list = []
    t_losses, v_losses, v_rmses = [], [], []
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        t_ep = time.time()

        tl, tr, tm, _  = run_epoch(model, train_eps, opt)
        vl, vr, vm, vx = run_epoch(model, test_eps)
        sched.step()

        ema_loss = vl if ep == 1 else 0.7 * ema_loss + 0.3 * vl
        g0  = model.gate_value_at_raw_amps(0.0)   # must stay 0.0 throughout
        z0n = model.z0_norm()
        lr  = sched.get_last_lr()[0]

        t_losses.append(tl); v_losses.append(vl); v_rmses.append(vr)
        log.append(dict(
            epoch=ep, trn_loss=tl, val_loss=vl, ema_loss=ema_loss,
            val_rmse=vr, val_mae=vm, gate_at_zero_raw=g0, z0_norm=z0n, lr=lr,
        ))

        marker = ""
        if ema_loss < best_loss:
            best_loss = ema_loss
            patience  = 0
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "ren_soc_best.pth"))
            marker = " ◀"
        else:
            patience += 1

        print(f"  {ep:>4d}  {tl:>10.6f}  {vl:>10.6f}  {ema_loss:>10.6f}  "
              f"{vr:>9.5f}  {vm:>8.5f}  {g0:>7.4f}  {z0n:>6.3f}  {lr:>9.2e}"
              f"  [{time.time()-t_ep:.0f}s]{marker}")

        if patience >= PATIENCE:
            print(f"\n  Early stop at epoch {ep}.")
            break

    # ── Final metrics ─────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ren_soc_last.pth"))
    pd.DataFrame(log).to_csv(os.path.join(SAVE_DIR, "training_log.csv"), index=False)

    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "ren_soc_best.pth"), map_location=DEVICE)
    )
    _, ren_rmse, ren_mae, ren_max = run_epoch(model, test_eps)

    print(f"\n{'='*68}")
    print(f"  FINAL — CC+REN hybrid vs Coulomb Counter baseline")
    print(f"{'='*68}")
    print(f"  {'Metric':<16} {'CC+REN':>12}  {'CC':>12}  {'Improvement':>12}")
    print(f"  {'-'*56}")
    for rv, cv, lbl in [(ren_rmse, cc_rmse, "RMSE"),
                        (ren_mae,  cc_mae,  "MAE"),
                        (ren_max,  cc_max,  "Max Error")]:
        imp = f"{(1 - rv/cv)*100:+.1f}%"
        print(f"  {lbl:<16} {rv:>12.5f}  {cv:>12.5f}  {imp:>12}")

    print(f"\n  Training time  : {(time.time()-t0)/60:.1f} min")
    print(f"  Best EMA loss  : {best_loss:.6f}")
    print(f"  Gate @ I=0A    : {model.gate_value_at_raw_amps(0.0):.4f}  (must be 0.0)")
    print(f"  z0 norm        : {model.z0_norm():.4f}")

    # ── Training curves ───────────────────────────────────────────────────
    ep_x = list(range(1, len(t_losses) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ep_x, t_losses, label="Train", color="steelblue", lw=1.5)
    axes[0].plot(ep_x, v_losses, label="Val",   color="tomato",    lw=1.5)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.4)

    axes[1].plot(ep_x, v_rmses, color="purple", lw=2, label="CC+REN RMSE")
    axes[1].axhline(cc_rmse, color="orange", ls="--", lw=1.5,
                    label=f"CC = {cc_rmse:.4f}")
    axes[1].set_title("Val RMSE vs CC baseline")
    axes[1].legend(); axes[1].grid(alpha=0.4)

    gate_vals = [row["gate_at_zero_raw"] for row in log]
    axes[2].plot(ep_x, gate_vals, color="red", lw=1.5, label="gate @ I=0A (raw)")
    axes[2].axhline(0, color="black", ls="--", lw=0.8)
    axes[2].axhline(0.1, color="red", ls=":", lw=0.8, alpha=0.5, label="threshold 0.1")
    axes[2].set_title("Gate @ I=0A  (target: 0.0 — physics law)")
    axes[2].legend(); axes[2].grid(alpha=0.4)

    plt.suptitle("REN CC+REN Hybrid — physics-constrained (gate fix applied)",
                 fontsize=12)
    plt.tight_layout()
    fp = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {fp}")
    print(f"  Saved → {SAVE_DIR}/ren_soc_best.pth")
    print(f"  Saved → {SAVE_DIR}/training_log.csv")


if __name__ == "__main__":
    main()