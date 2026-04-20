"""
REN Training Script — VRFB SOC Estimator  v2.0  (Pure Observer)
================================================================

Pure Observer Architecture
---------------------------
  INPUT_DIM = 6:  voltage, current, T_stack, T_tank, flow_rate, soc_cc
  TARGET    = SOC_true - soc_cc   (raw CC drift, no pre-correction)

  All derived/integrated features removed:
    - bias_est         (caused runaway feedback at inference)
    - cumulative_ah_norm (OOD after 10h; explodes at 60h)
    - transport_ratio_approx (redundant with current+flow)
    - soc_cc_corrected  (data leakage — needed ground truth at inference)

  The REN hidden state z is the ONLY long-horizon integrator.
  It must learn to accumulate drift evidence from the Nernst voltage
  signal across many timesteps to produce a bounded correction.

Architecture constraints
-------------------------
  1. use_current_gate=True  — z frozen at I=0A (Faraday's law)
  2. use_feedthrough=False  — no D(x_t) direct path
  3. ZC invariance loss     — reinforces gate constraint in loss space
  4. Drift regularisation   — forces long-horizon mean correction to track
                              mean target over a 1-hour rolling window
  5. Bias penalty           — forces episode-level mean correction = mean target
  6. z0 regularisation      — suppresses init spike at step 0

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

# ── 6 strictly observable features — must match dataset_gen.py exactly ───────
FEATURE_COLS = [
    "v_ocv_approx",      # SOLE voltage signal — Nernst OCV without Ohmic contamination
                         # v_ocv = V_terminal - I*R_nom; smooth at current reversals
                         # 'voltage' removed: V_terminal = v_ocv + I*R is linearly
                         # dependent given 'current' → collinearity causes overfitting
    "current",           # I=0 → gate closes → z frozen (Faraday); independent signal
    "temperature_stack",
    "temperature_tank",
    "flow_rate",
    "soc_cc",          # raw drifting CC — REN learns to correct this
]
TARGET_COL  = "target"           # SOC_true - soc_cc  (raw CC drift)
INPUT_DIM   = len(FEATURE_COLS)  # 6
CURRENT_IDX = 1                  # index of "current" in FEATURE_COLS

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
HARD_CASE_WEIGHT = 3.0
HARD_CASE_THRESH = 0.10

# Zero-current invariance (raw Amps — matches BMS dead-band)
ZC_WEIGHT   = 2.0
ZC_I_THRESH = 2.0

SMOOTH_WEIGHT = 0.15       # reduced from 0.30 — less smoothing allows step tracking
SMOOTH_I_REF  = 0.3
DRIFT_WEIGHT  = 0.30       # was 1.0 — 1.0 was overpowering MSE, making model output
                           # the average drift instead of tracking it step-by-step.
                           # Correlation dropped to -0.57 because the model satisfied
                           # the 1-hour mean constraint by outputting a flat constant.
                           # 0.30 keeps the long-horizon signal without killing precision.
DRIFT_WINDOW  = 3600
BIAS_PENALTY_WEIGHT = 0.50 # was 1.0 — same issue as DRIFT_WEIGHT
Z0_REG_WEIGHT = 3.0        # increased from 2.0 — harder constraint on init spike.
                           # z0_norm=1.08 caused visible 0.07 correction spike at t=0.
                           # 3.0 forces z0 output near-zero regardless of z0_norm.
TRANSITION_WEIGHT = 0.50   # reduced from 1.0

TRAIN_CSV = "datasets/vrfb_train.csv"
TEST_CSV  = "datasets/vrfb_test.csv"
SAVE_DIR  = "ren"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set after scaler is fit in main() — used in run_epoch() for gate
_I_MEAN: float = 0.0
_I_STD:  float = 1.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_episodes(df: pd.DataFrame, scaler: StandardScaler,
                  fit_scaler: bool = False) -> list:
    """Returns list of (X_ep_scaled, y_ep) tensors per episode."""
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
    pred:        torch.Tensor,          # (B, seq, 1) predicted correction
    target:      torch.Tensor,          # (B, seq, 1) true correction
    X_chunk:     torch.Tensor | None,   # (B, seq, 6) scaled features
    I_raw_chunk: torch.Tensor | None,   # (B, seq, 1) raw Amps
) -> torch.Tensor:
    err = pred - target
    with torch.no_grad():
        w = torch.where(
            err.abs() > HARD_CASE_THRESH,
            torch.full_like(err, HARD_CASE_WEIGHT),
            torch.ones_like(err),
        )

    base = MSE_WEIGHT * (w * err**2).mean() + MAE_WEIGHT * F.l1_loss(pred, target)

    if X_chunk is None or pred.shape[1] <= 1:
        return base

    I_scaled = X_chunk[:, :, CURRENT_IDX:CURRENT_IDX+1]
    delta    = (pred[:, 1:, :] - pred[:, :-1, :]).abs()
    I_mid_s  = I_scaled[:, 1:, :].abs()

    # Smooth loss — penalise large corrections when current is low
    # Smooth loss — penalise large corrections when current is low
    with torch.no_grad():
        smooth_w = torch.exp(-I_mid_s / SMOOTH_I_REF)
    smooth_loss = (smooth_w * delta).mean()

    # Transition-aware smooth loss (shark-fin fix)
    # Penalises large corrections at current reversal events.
    # At charge↔discharge transition: |dI/dt| is large (up to 300A in one step).
    # The model should NOT produce a spike correction here — it should recognise
    # the voltage jump as Ohmic (2*I*R), not Nernst (SOC change).
    # Using I_raw_chunk for physical amplitude detection.
    if I_raw_chunk is not None:
        # |dI| between consecutive steps in raw Amps
        dI_raw     = (I_raw_chunk[:, 1:, :] - I_raw_chunk[:, :-1, :]).abs()
        # Normalise: at 150A→-150A reversal dI=300A → normalised=1.5 → clamp to 1
        dI_norm    = torch.clamp(dI_raw / 200.0, 0.0, 1.0)
        with torch.no_grad():
            transition_w = dI_norm   # 0 during steady operation, 1 at full reversal
        transition_loss = (transition_w * delta).mean()
    else:
        transition_loss = torch.tensor(0.0)

    # ZC invariance — gate constraint enforced in loss space
    if I_raw_chunk is not None:
        I_mid_raw = I_raw_chunk[:, 1:, :].abs()
        with torch.no_grad():
            zc_mask = (I_mid_raw < ZC_I_THRESH).float()
        zc_loss = (zc_mask * delta).mean()
    else:
        zc_loss = torch.tensor(0.0)

    # Drift regularisation — rolling 1-hour window consistency
    if pred.shape[1] >= DRIFT_WINDOW:
        hw          = DRIFT_WINDOW // 2
        pred_roll   = pred[:, :pred.shape[1]-hw, :].unfold(1, hw, 1).mean(dim=-1)
        target_roll = target[:, :target.shape[1]-hw, :].unfold(1, hw, 1).mean(dim=-1)
        drift_loss  = ((pred_roll - target_roll)**2).mean()
    else:
        drift_loss = ((pred.mean(dim=1) - target.mean(dim=1))**2).mean()

    # Bias penalty — episode-level mean correction = mean target
    bias_penalty = ((pred.mean(dim=1) - target.mean(dim=1))**2).mean()

    return (base
            + SMOOTH_WEIGHT       * smooth_loss
            + ZC_WEIGHT           * zc_loss
            + DRIFT_WEIGHT        * drift_loss
            + BIAS_PENALTY_WEIGHT * bias_penalty
            + TRANSITION_WEIGHT   * transition_loss)


# =============================================================================
# STATEFUL EPOCH
# =============================================================================

def run_epoch(
    model:     REN,
    episodes:  list,
    optimiser: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float]:
    is_train = optimiser is not None
    model.train() if is_train else model.eval()
    ctx = torch.enable_grad if is_train else torch.no_grad

    perm       = torch.randperm(len(episodes)).tolist()
    total_loss = 0.0
    sq_err:  list = []
    abs_err: list = []
    n_b        = 0

    for bs in range(0, len(perm), BATCH_SIZE):
        batch    = [episodes[i] for i in perm[bs:bs+BATCH_SIZE]]
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

            Xc = torch.stack([ep[0][s:e] for ep in batch]).to(DEVICE)  # (B, seq, 6)
            yc = torch.stack([ep[1][s:e] for ep in batch]).to(DEVICE)  # (B, seq, 1)

            # Reconstruct raw current from scaler stats for gate
            I_raw = Xc[:, :, CURRENT_IDX:CURRENT_IDX+1] * _I_STD + _I_MEAN

            use_z0 = is_train and ci > 0 and torch.rand(1).item() < P_Z0_RESET

            with ctx():
                yp, zn = model(Xc, z=None if use_z0 else z, x_raw=I_raw)

                # Warmup masking
                if WARMUP_STEPS > 0 and SEQ_LEN > WARMUP_STEPS:
                    yp_l    = yp[:,    WARMUP_STEPS:, :]
                    yc_l    = yc[:,    WARMUP_STEPS:, :]
                    Xc_l    = Xc[:,    WARMUP_STEPS:, :]
                    I_raw_l = I_raw[:, WARMUP_STEPS:, :]
                else:
                    yp_l, yc_l, Xc_l, I_raw_l = yp, yc, Xc, I_raw

                loss = composite_loss(yp_l, yc_l, Xc_l, I_raw_l)

                # z0 regularisation at first chunk — suppresses init spike
                if ci == 0 and is_train:
                    y_step0 = yp[:, 0, :]
                    loss    = loss + Z0_REG_WEIGHT * (y_step0**2).mean()

                bloss += loss.item()

                e_arr = (yp_l - yc_l).detach().cpu().numpy().ravel()
                sq_err.append(e_arr**2)
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
            s = self.m + 0.5*(1-self.m)*(1+math.cos(math.pi*(e-self.w)/max(self.t-self.w,1)))
        return [b * s for b in self.base_lrs]


# =============================================================================
# MAIN
# =============================================================================

def main():
    global _I_MEAN, _I_STD

    print(f"\n{'='*68}")
    print(f"  REN Training  v2.0  (Pure Observer Architecture)")
    print(f"{'='*68}")
    print(f"  Device         : {DEVICE}")
    print(f"  Features ({INPUT_DIM})   : {FEATURE_COLS}")
    print(f"  Target         : SOC_true - soc_cc  (raw CC drift)")
    print(f"  Gate           : use_current_gate=True (raw Amps)")
    print(f"  Feedthrough    : False")
    print(f"  Drift window   : {DRIFT_WINDOW}s  (DRIFT_WEIGHT={DRIFT_WEIGHT})")
    print(f"  Bias penalty   : {BIAS_PENALTY_WEIGHT}")
    print(f"  z0 reg         : {Z0_REG_WEIGHT}")
    print(f"  PI observer    : REMOVED — z is the sole integrator")
    print(f"{'-'*68}")

    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    # Validate columns
    needed = FEATURE_COLS + [TARGET_COL, "episode_id"]
    for col in needed:
        if col not in df_train.columns:
            raise ValueError(
                f"Missing column '{col}' in training CSV.\n"
                f"Re-run dataset_gen.py (v5 Pure Observer)."
            )

    print(f"\n  Train : {len(df_train):,} rows  "
          f"({df_train['episode_id'].nunique()} episodes)")
    print(f"  Test  : {len(df_test):,} rows  "
          f"({df_test['episode_id'].nunique()} episodes)")

    # Fit scaler and extract current stats for gate reconstruction
    scaler    = StandardScaler()
    train_eps = load_episodes(df_train, scaler, fit_scaler=True)
    test_eps  = load_episodes(df_test,  scaler, fit_scaler=False)

    _I_MEAN = float(scaler.mean_[CURRENT_IDX])
    _I_STD  = float(scaler.scale_[CURRENT_IDX])

    print(f"\n  Current scaler : mean={_I_MEAN:.2f}A  std={_I_STD:.2f}A")
    print(f"  Scaled I=0A    : {-_I_MEAN/_I_STD:.4f}  (gate uses raw Amps — correct)")

    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved   : {scaler_path}")

    # CC baseline
    cc_err  = np.abs(df_test["SOC_true"].values - df_test["soc_cc"].values)
    cc_rmse = math.sqrt(np.mean(cc_err**2))
    cc_mae  = cc_err.mean()
    print(f"\n  CC baseline  RMSE={cc_rmse:.5f}  MAE={cc_mae:.5f}")

    # Build model
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
    print(f"  Gate @ I=0A    : {model.gate_value_at_raw_amps(0.0):.6f}  (must be 0.0)")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = WarmupCosine(opt, LR_WARMUP_EPOCHS, EPOCHS)

    print(f"\n  {'Ep':>4}  {'TrnLoss':>9}  {'ValLoss':>9}  {'EMA':>9}  "
          f"{'RMSE':>8}  {'MAE':>7}  {'g@0A':>7}  {'z0n':>5}  {'LR':>9}")
    print(f"  {'-'*92}")

    best_loss = math.inf
    ema_loss  = math.inf
    patience  = 0
    log: list = []
    t_losses, v_losses, v_rmses = [], [], []
    t0 = time.time()

    for ep in range(1, EPOCHS+1):
        t_ep = time.time()

        tl, tr, tm, _  = run_epoch(model, train_eps, opt)
        vl, vr, vm, vx = run_epoch(model, test_eps)
        sched.step()

        ema_loss = vl if ep == 1 else 0.7*ema_loss + 0.3*vl
        g0  = model.gate_value_at_raw_amps(0.0)
        z0n = model.z0_norm()
        lr  = sched.get_last_lr()[0]

        t_losses.append(tl); v_losses.append(vl); v_rmses.append(vr)
        log.append(dict(epoch=ep, trn_loss=tl, val_loss=vl, ema_loss=ema_loss,
                        val_rmse=vr, val_mae=vm, gate_at_zero=g0, z0_norm=z0n, lr=lr))

        marker = ""
        if ema_loss < best_loss:
            best_loss = ema_loss
            patience  = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ren_soc_best.pth"))
            marker = " <"
        else:
            patience += 1

        print(f"  {ep:>4d}  {tl:>9.6f}  {vl:>9.6f}  {ema_loss:>9.6f}  "
              f"{vr:>8.5f}  {vm:>7.5f}  {g0:>7.4f}  {z0n:>5.3f}  {lr:>9.2e}"
              f"  [{time.time()-t_ep:.0f}s]{marker}")

        if patience >= PATIENCE:
            print(f"\n  Early stop at epoch {ep}.")
            break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ren_soc_last.pth"))
    pd.DataFrame(log).to_csv(os.path.join(SAVE_DIR, "training_log.csv"), index=False)

    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "ren_soc_best.pth"), map_location=DEVICE)
    )
    _, ren_rmse, ren_mae, ren_max = run_epoch(model, test_eps)

    print(f"\n{'='*68}")
    print(f"  FINAL RESULTS  (CC baseline vs CC+REN)")
    print(f"{'='*68}")
    print(f"  {'Metric':<16} {'CC+REN':>12}  {'CC':>12}  {'Improvement':>12}")
    print(f"  {'-'*56}")
    for rv, cv, lbl in [(ren_rmse,cc_rmse,"RMSE"),(ren_mae,cc_mae,"MAE")]:
        print(f"  {lbl:<16} {rv:>12.5f}  {cv:>12.5f}  {(1-rv/cv)*100:>+11.1f}%")

    print(f"\n  Training time  : {(time.time()-t0)/60:.1f} min")
    print(f"  Gate @ I=0A    : {model.gate_value_at_raw_amps(0.0):.6f}  (must be 0.0)")
    print(f"  z0 norm        : {model.z0_norm():.4f}")

    # Training curves
    ep_x = list(range(1, len(t_losses)+1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep_x, t_losses, label="Train", color="steelblue", lw=1.5)
    axes[0].plot(ep_x, v_losses, label="Val",   color="tomato",    lw=1.5)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.4)
    axes[1].plot(ep_x, v_rmses, color="purple", lw=2, label="CC+REN RMSE")
    axes[1].axhline(cc_rmse, color="orange", ls="--", lw=1.5,
                    label=f"CC baseline = {cc_rmse:.4f}")
    axes[1].set_title("Val RMSE vs CC baseline"); axes[1].legend(); axes[1].grid(alpha=0.4)
    gate_vals = [r["gate_at_zero"] for r in log]
    axes[2].plot(ep_x, gate_vals, color="red", lw=1.5, label="gate @ I=0A (raw)")
    axes[2].axhline(0, color="black", ls="--", lw=0.8)
    axes[2].set_title("Gate @ I=0A  (target: 0.0)"); axes[2].legend(); axes[2].grid(alpha=0.4)
    plt.suptitle("REN Pure Observer — 6-feature model", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {SAVE_DIR}/ren_soc_best.pth")
    print(f"  Saved: {SAVE_DIR}/training_curves.png")


if __name__ == "__main__":
    main()