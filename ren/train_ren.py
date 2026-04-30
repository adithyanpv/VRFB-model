"""
REN Training Script — VRFB SOC Estimator  v4.0  (Direct SOC — Structurally Corrected)
=======================================================================================

What changed from v3.0 and why
--------------------------------
  TARGET: SOC_true (unchanged from v3.0 — but now the architecture matches it)

  FEATURE SET — 6 inputs:
    v_ocv_approx, current, temperature_stack, temperature_tank, flow_rate,
    soc_init_decay   ← replaces soc_cc

  WHY soc_cc was removed:
    With 3A sensor bias, soc_cc ≈ SOC_true − tiny_drift.  In one epoch the
    model learned output ≈ f(soc_cc), giving Corr(CC_err, correction)=+0.517
    (target −1.0).  The shortcut required no temporal integration at all —
    the hidden state z was unused.  0/5 cycles improved in cycle test.

  WHY soc_init_decay replaces it:
    soc_init_decay[t] = SOC_true[t=0] × exp(−t / 1800s)
    Carries the same cold-start information as soc_cc at t=0, but decays to
    near-zero by t=7200s.  The shortcut is unavailable after 30 minutes —
    the model must decode SOC from v_ocv_approx (Nernst) for the remaining
    9.5 hours of each 10-hour episode.
    At INFERENCE: computed from Nernst inversion of v_ocv_approx[0].

  LOSS CHANGES:
    REMOVED — ZC invariance (ZC_WEIGHT was 2.0):
      At I=0A, OCV encodes SOC via Nernst with no Ohmic drop — best signal.
      Penalising ΔSOC at rest blocked this update entirely.  Root cause of
      flat dashboard output during BMS standby.

    REMOVED — Smooth loss (SMOOTH_WEIGHT was 0.20):
      SMOOTH_I_REF=0.3 (scaled current) gave weight≈0.89 at I=0A, same
      incorrect behaviour as ZC loss.  Suppressed rest-period OCV updates.

    ADDED — Directional loss (DIRECTIONAL_WEIGHT=1.0):
      Window-level (128-step) monotonicity constraint.
      Discharge (mean_I > 5A): penalises positive ΔSOC windows.
      Charge    (mean_I < −5A): penalises negative ΔSOC windows.
      Window delta ≈ O(1e-3) — competes with base MSE O(1e-2).
      Per-step delta O(1e-5) was too small to produce useful gradients.

    ADDED — Slope loss chunk-level (SLOPE_WEIGHT=1.5, was per-step):
      Expected ΔSOC over full SEQ_LEN chunk = −mean(I)×T/(Q_NOMINAL×3600).
      At 150A for 992 steps: ΔSOC≈−0.0193, magnitude O(1e-2) — meaningful.
      Disabled for rest chunks (|mean_I|<5A) to avoid noise-dominated loss.
      Previous per-step version O(1e-5) was negligible.

    RETAINED — drift, bias_penalty, transition, z0_reg (all valid for SOC).

  ARCHITECTURE NOTE — apply gate floor in ren_model.py:
    In _step(), replace:
      gate = torch.tanh(self.gate_k.abs() * I_abs)
    With:
      GATE_FLOOR = 0.15
      gate = GATE_FLOOR + (1-GATE_FLOOR) * torch.tanh(self.gate_k.abs() * I_abs)
    Without this, gate=0 at I=0 freezes z, blocking OCV reads during standby.

OUTPUT:
  ren/scaler.pkl          ← new 6-feature scaler, incompatible with v3.0
  ren/ren_soc_best.pth    ← delete old weights before deploying
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
    "v_ocv_approx",      # Nernst OCV: V_terminal − I×R_nom (no Ohmic contamination)
    "current",           # raw Amps — gate uses this directly for I=0 detection
    "temperature_stack",
    "temperature_tank",
    "flow_rate",
    "soc_init_decay",    # SOC_true[t=0] × exp(−t / INIT_DECAY_STEPS)
                         # Provides a per-episode initial SOC prior that fades over
                         # 30 min.  Eliminates sigmoid-0.5 cold-start trap without
                         # z0_proj architecture changes.
                         # At t=0: full initial SOC.  At t=7200s: ≈0.02×initial_soc.
                         # INFERENCE: compute from Nernst inversion of v_ocv_approx[0]
                         # — see server.py / cycle_test.py for implementation.
                         #
                         # WHY NOT soc_cc:
                         # With 3A bias, soc_cc ≈ SOC_true − tiny_drift.  The model
                         # learned output ≈ f(soc_cc) in one epoch (shortcut), producing
                         # Corr(CC_err, correction) = +0.517 instead of ≈−1.0.
                         # soc_init_decay carries the same cold-start information but
                         # decays to zero — the shortcut is unavailable after 30 min.
]
TARGET_COL  = "SOC_true"        # direct SOC ∈ [0.05, 0.95]
INPUT_DIM   = len(FEATURE_COLS) # 6
CURRENT_IDX = 1                 # index of "current" in FEATURE_COLS

# ── Initial SOC decay constant ────────────────────────────────────────────────
INIT_DECAY_STEPS = 1800.0   # seconds  (30-min time constant)
# t=0:    decay = initial_soc            (full prior)
# t=1800: decay = 0.368 × initial_soc   (model half-way self-reliant)
# t=7200: decay = 0.018 × initial_soc   (model fully self-reliant)

# ── Physics constant for slope loss ──────────────────────────────────────────
Q_NOMINAL = 2144.0   # Ah  n×F×C_total×V_tank/3600  (from config.py)
# ΔSOC_physics per step = −I / (Q_NOMINAL × 3600)

# Model
HIDDEN_DIM    = 128
ALPHA         = 0.5
DROPOUT       = 0.1
N_POWER_ITERS = 10

# Training
SEQ_LEN          = 1024
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

# Zero-current invariance — DISABLED for direct SOC estimation.
# At I=0A (standby/rest), the open-circuit voltage encodes SOC via Nernst
# with no Ohmic drop — it is the BEST available signal.  Penalising ΔSOC
# at rest (as this loss did) prevented the model from using that signal.
# Root cause of flat dashboard output during BMS standby periods.
ZC_WEIGHT   = 0.0    # ← was 2.0  DISABLED
ZC_I_THRESH = 2.0    # kept for reference; weight is zero

# Smooth loss — DISABLED, replaced by directional_loss below.
# Previous formulation used SMOOTH_I_REF=0.3 (scaled units) which gave
# exp(−0.036/0.3) ≈ 0.89 weight at I=0A — nearly fully penalising OCV
# updates during rest.  Same incorrect behaviour as ZC loss.
SMOOTH_WEIGHT = 0.0  # ← was 0.20  DISABLED
SMOOTH_I_REF  = 0.3  # kept for reference; weight is zero

# Directional loss — NEW.
# Window-level physical monotonicity: SOC must decrease during discharge
# (I > DIRECTIONAL_I_THRESH) and increase during charge (I < −threshold).
# Uses 128-step windows so delta ≈ O(1e-3) — large enough to compete with
# base MSE ≈ O(1e-2).  Per-step delta O(1e-5) was too small to matter.
DIRECTIONAL_WEIGHT   = 1.0
DIRECTIONAL_I_THRESH = 5.0    # raw Amps — filters sensor noise from direction test
DIRECTIONAL_WINDOW   = 128    # steps per window (matches DRIFT_WINDOW)

DRIFT_WEIGHT        = 0.40
DRIFT_WINDOW        = 128
BIAS_PENALTY_WEIGHT = 0.80
Z0_REG_WEIGHT       = 5.0
TRANSITION_WEIGHT   = 2.0
PEARSON_WEIGHT      = 0.0

# Slope loss — ENABLED with chunk-level Faraday physics.
# At 150A for SEQ_LEN=1024 steps: ΔSOC ≈ −150×1024/(2144×3600) ≈ −0.0199.
# Magnitude O(1e-2) → meaningful gradient.  Disabled for rest chunks
# (|mean_I| < 5A) where noise dominates the expected ΔSOC signal.
# Previous version used per-step slope (O(1e-5)) — too small, wrong physics.
SLOPE_WEIGHT = 1.5   # ← was 0.0 or 1.0 (per-step, incorrect)  NOW CHUNK-LEVEL

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
    """
    Build (X_scaled, y) tensors per episode.

    Feature pipeline:
      1. Read FEATURE_COLS_CSV = [v_ocv_approx, current, T_stack, T_tank, flow_rate]
         directly from the dataframe (5 columns).
      2. Compute soc_init_decay per episode:
           soc_init_decay[t] = SOC_true[t=0] × exp(−t / INIT_DECAY_STEPS)
         This is the 6th feature.  At INFERENCE, replace SOC_true[t=0] with
         the Nernst-inverted OCV at step 0 — see server.py for code.
      3. Concatenate → X_full (N, 6).
      4. Fit StandardScaler on full concatenated training set (if fit_scaler).
      5. Scale and return tensors.

    soc_cc is NOT read as a feature.  It IS still in the CSV and is used
    only for CC baseline computation in main() — not as model input.
    """
    FEATURE_COLS_CSV = [
        "v_ocv_approx", "current", "temperature_stack",
        "temperature_tank", "flow_rate",
    ]

    ep_X_raw = []
    ep_y_raw = []

    for ep_id, group in df.groupby("episode_id", sort=True):
        T       = len(group)
        X_csv   = group[FEATURE_COLS_CSV].values.astype(np.float32)   # (T, 5)

        # Decaying initial SOC prior — computed from ground truth during training.
        # At inference: use Nernst inversion of v_ocv_approx[0] instead.
        initial_soc = float(group["SOC_true"].iloc[0])
        steps       = np.arange(T, dtype=np.float32)
        soc_decay   = (initial_soc * np.exp(-steps / INIT_DECAY_STEPS)
                       ).astype(np.float32)                            # (T,)

        X_full = np.concatenate([X_csv, soc_decay[:, np.newaxis]], axis=1)  # (T, 6)
        y_ep   = group["SOC_true"].values.astype(np.float32)                # (T,)

        ep_X_raw.append(X_full)
        ep_y_raw.append(y_ep)

    # Fit scaler on concatenated full training set
    X_all = np.concatenate(ep_X_raw, axis=0)
    if fit_scaler:
        scaler.fit(X_all)

    # Scale and wrap as tensors
    episodes = []
    for X_ep, y_ep in zip(ep_X_raw, ep_y_raw):
        X_scaled = scaler.transform(X_ep).astype(np.float32)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_ep,     dtype=torch.float32).unsqueeze(-1)
        episodes.append((X_t, y_t))

    return episodes


# =============================================================================
# LOSS
# =============================================================================

def directional_loss(
    pred:  torch.Tensor,   # (B, seq, 1) predicted SOC
    I_raw: torch.Tensor,   # (B, seq, 1) raw Amps
) -> torch.Tensor:
    """
    Window-level physical monotonicity constraint.

    SOC must decrease during discharge (I > DIRECTIONAL_I_THRESH) and
    increase during charge (I < -DIRECTIONAL_I_THRESH).

    Uses non-overlapping DIRECTIONAL_WINDOW-step windows so window delta
    is O(1e-3) — large enough to compete with base MSE O(1e-2).
    Per-step delta O(1e-5) at 150A was too small to produce useful gradients.
    """
    hw    = DIRECTIONAL_WINDOW
    B, T, _ = pred.shape
    n_win = T // hw
    if n_win == 0:
        return torch.tensor(0.0, device=pred.device)

    pred_w = pred[:, :n_win * hw, :].reshape(B, n_win, hw, 1)
    I_w    = I_raw[:, :n_win * hw, :].reshape(B, n_win, hw, 1)

    delta_win  = pred_w[:, :, -1, :] - pred_w[:, :, 0, :]   # (B, n_win, 1)
    I_mean_win = I_w.mean(dim=2)                              # (B, n_win, 1)

    # Discharge: I > thresh -> DELTA must be <= 0 -> penalise positive delta
    dis_viol = torch.relu(delta_win)  * (I_mean_win >  DIRECTIONAL_I_THRESH).float()
    # Charge:    I < -thresh -> DELTA must be >= 0 -> penalise negative delta
    chg_viol = torch.relu(-delta_win) * (I_mean_win < -DIRECTIONAL_I_THRESH).float()

    return (dis_viol + chg_viol).mean()


def composite_loss(
    pred:        torch.Tensor,          # (B, seq, 1) predicted SOC  in (0, 1)
    target:      torch.Tensor,          # (B, seq, 1) SOC_true        in [0.05, 0.95]
    X_chunk:     torch.Tensor | None,   # (B, seq, 6) scaled features
    I_raw_chunk: torch.Tensor | None,   # (B, seq, 1) raw Amps
) -> torch.Tensor:
    """
    Multi-term loss for direct SOC estimation.

    ACTIVE:
      base         weighted MSE + MAE with hard-case emphasis
      directional  window monotonicity (discharge down, charge up)
      slope        chunk-level Faraday physics: DELTA_SOC = -mean(I)*T/(Q*3600)
      drift        rolling-128-step mean pred ~ mean target
      bias_penalty episode-level mean pred ~ episode mean SOC
      transition   spike suppression at current reversals

    DISABLED (weight=0):
      smooth_loss  was penalising OCV updates at rest -- wrong direction
      zc_loss      was freezing predictions at I=0 -- blocked Nernst signal
    """
    # Base: weighted MSE + MAE
    err = pred - target
    with torch.no_grad():
        w = torch.where(
            err.abs() > HARD_CASE_THRESH,
            torch.full_like(err, HARD_CASE_WEIGHT),
            torch.ones_like(err),
        )
    base = MSE_WEIGHT * (w * err ** 2).mean() + MAE_WEIGHT * F.l1_loss(pred, target)

    if X_chunk is None or pred.shape[1] <= 1:
        return base

    delta = (pred[:, 1:, :] - pred[:, :-1, :]).abs()   # |step DELTA_SOC|

    # Directional loss (window-level monotonicity)
    if I_raw_chunk is not None:
        dir_loss = directional_loss(pred, I_raw_chunk)
    else:
        dir_loss = torch.tensor(0.0, device=pred.device)

    # Transition loss: spike suppression at current reversal
    if I_raw_chunk is not None:
        dI_raw        = (I_raw_chunk[:, 1:, :] - I_raw_chunk[:, :-1, :]).abs()
        dI_norm       = torch.clamp(dI_raw / 200.0, 0.0, 1.0)
        with torch.no_grad():
            transition_w = dI_norm
        transition_loss = (transition_w * delta).mean()
    else:
        transition_loss = torch.tensor(0.0, device=pred.device)

    # DISABLED: smooth loss was penalising OCV updates at I=0 (weight~0.89 at rest)
    smooth_loss = torch.tensor(0.0, device=pred.device)   # SMOOTH_WEIGHT = 0.0

    # DISABLED: ZC invariance blocked Nernst updates during standby
    zc_loss = torch.tensor(0.0, device=pred.device)        # ZC_WEIGHT = 0.0

    # Drift loss: rolling-window mean consistency
    hw          = DRIFT_WINDOW
    pred_roll   = pred.unfold(1, hw, 1).mean(dim=-1)
    target_roll = target.unfold(1, hw, 1).mean(dim=-1)
    drift_loss  = ((pred_roll - target_roll) ** 2).mean()

    # Bias penalty: episode-level mean SOC alignment
    bias_penalty = ((pred.mean(dim=1) - target.mean(dim=1)) ** 2).mean()

    # Slope loss: CHUNK-LEVEL Faraday physics (not per-step -- too small)
    # DELTA_SOC over full chunk = -mean(I) * n_steps / (Q_NOMINAL * 3600)
    # At 150A for 992 steps: ~-0.0193. Magnitude O(1e-2) = meaningful gradient.
    # Disabled for rest chunks (|mean_I| < 5A) to avoid noise-dominated loss.
    if I_raw_chunk is not None:
        I_chunk_mean   = I_raw_chunk.mean(dim=1)                             # (B, 1)
        active_mask    = (I_chunk_mean.abs() > 5.0).float()                  # (B, 1)
        expected_delta = -I_chunk_mean * pred.shape[1] / (Q_NOMINAL * 3600.0)
        actual_delta   = pred[:, -1, :] - pred[:, 0, :]                     # (B, 1)
        slope_loss     = (active_mask * (actual_delta - expected_delta) ** 2).mean()
    else:
        slope_loss = torch.tensor(0.0, device=pred.device)

    return (
        base
        + DIRECTIONAL_WEIGHT  * dir_loss
        + SLOPE_WEIGHT        * slope_loss
        + DRIFT_WEIGHT        * drift_loss
        + BIAS_PENALTY_WEIGHT * bias_penalty
        + TRANSITION_WEIGHT   * transition_loss
        + SMOOTH_WEIGHT       * smooth_loss  # = 0.0  disabled
        + ZC_WEIGHT           * zc_loss      # = 0.0  disabled
    )

def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation loss over the sequence dimension.

    L = mean over batch of (1 - r(pred_b, target_b))^2
    where r is the Pearson correlation coefficient.

    A flat constant correction → r=0 → loss=1.0 (maximum penalty).
    Perfect dynamic tracking  → r=1 → loss=0.0.

    This is the critical loss that breaks the "lazy constant offset"
    local minimum: the model cannot satisfy this loss by outputting
    mean(target) — it must actually track the dynamic drift signal.

    Shape: pred and target are (batch, seq, 1).
    Computed per-batch-element over the seq dimension then averaged.
    """
    # Flatten seq and feature dims: (batch, seq)
    p = pred.squeeze(-1)    # (B, seq)
    t = target.squeeze(-1)  # (B, seq)

    p_mean = p.mean(dim=1, keepdim=True)
    t_mean = t.mean(dim=1, keepdim=True)
    p_c    = p - p_mean
    t_c    = t - t_mean

    cov    = (p_c * t_c).sum(dim=1)                          # (B,)
    std_p  = p_c.pow(2).sum(dim=1).sqrt().clamp(min=1e-8)    # (B,)
    std_t  = t_c.pow(2).sum(dim=1).sqrt().clamp(min=1e-8)    # (B,)

    r = cov / (std_p * std_t)                                # (B,) in [-1, 1]
    # For direct SOC: r→1 means predicted trajectory tracks true SOC shape.
    # Penalty = (1-r)^2 is zero only when r=1 (perfect tracking).
    return ((1.0 - r) ** 2).mean()


# =============================================================================
# STATEFUL EPOCH
# =============================================================================

def run_epoch(
    model:     REN,
    episodes:  list,
    optimiser: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float]:
    """
    Stateful training/evaluation epoch with truncated BPTT.

    Hidden state z is carried across chunks within an episode so the model
    builds long-horizon state without backpropagating through the full 36000
    steps. P_Z0_RESET randomly resets z between chunks during training,
    forcing recovery from mid-episode cold starts (inference robustness).

    Returns: (mean_loss, RMSE, MAE, max_abs_error)
    """
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
                    z0_spike  = ((yp[:, 0, :] - yc[:, 0, :]) ** 2).mean()
                    z0_warmup = ((yp[:, :300, :] - yc[:, :300, :]) ** 2).mean()
                    
                    loss = (loss
                            + Z0_REG_WEIGHT       * z0_spike
                            + Z0_REG_WEIGHT * 0.8 * z0_warmup)
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

    print(f"\n{'='*72}")
    print(f"  REN Training  v4.0  (Direct SOC — Structurally Corrected)")
    print(f"{'='*72}")
    print(f"  Device           : {DEVICE}")
    print(f"  Features ({INPUT_DIM})     : {FEATURE_COLS}")
    print(f"  soc_cc           : REMOVED  (shortcut path, was Corr=+0.517)")
    print(f"  soc_init_decay   : ADDED    (initial SOC prior, decay={INIT_DECAY_STEPS:.0f}s)")
    print(f"  Target           : SOC_true  (direct absolute SOC)")
    print(f"  Gate             : use_current_gate=True (raw Amps)")
    print(f"  ZC loss          : DISABLED  (was blocking OCV at I=0)")
    print(f"  Smooth loss      : DISABLED  (was penalising rest-period updates)")
    print(f"  Directional loss : ENABLED  w={DIRECTIONAL_WEIGHT}  win={DIRECTIONAL_WINDOW}s  I_thresh={DIRECTIONAL_I_THRESH}A")
    print(f"  Slope loss       : ENABLED  w={SLOPE_WEIGHT}  chunk-level Faraday Q={Q_NOMINAL}Ah")
    print(f"  Drift window     : {DRIFT_WINDOW}s  (DRIFT_WEIGHT={DRIFT_WEIGHT})")
    print(f"  Bias penalty     : {BIAS_PENALTY_WEIGHT}")
    print(f"  z0 reg           : {Z0_REG_WEIGHT}")
    print(f"  Transition       : {TRANSITION_WEIGHT}")
    print(f"{'─'*72}")

    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    # soc_init_decay is computed dynamically from SOC_true[t=0] — not in CSV.
    # soc_cc IS still in CSV and is used only for CC baseline metric below.
    needed = ["v_ocv_approx", "current", "temperature_stack", "temperature_tank",
              "flow_rate", "SOC_true", "soc_cc", "episode_id"]
    for col in needed:
        if col not in df_train.columns:
            raise ValueError(
                f"Missing column '{col}' in training CSV.\n"
                f"Re-run dataset_gen.py.  soc_init_decay is computed "
                f"dynamically — it does not need to be in the CSV."
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

    print(f"\n  Current scaler   : mean={_I_MEAN:.2f}A  std={_I_STD:.2f}A")
    print(f"  Scaled I=0A      : {-_I_MEAN/_I_STD:.4f}  (gate uses raw Amps — correct)")
    print(f"  soc_init_decay   : mean={scaler.mean_[5]:.4f}  std={scaler.scale_[5]:.4f}  (idx 5)")
    print(f"  *** New scaler — incompatible with v3.0 weights.  Delete old .pth ***")

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
    print(f"  Gate @ I=0A    : {model.gate_value_at_raw_amps(0.0):.6f}  "
          f"(0.0 without floor; 0.15 after ren_model.py gate fix)")

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

    print(f"\n{'='*72}")
    print(f"  FINAL RESULTS  (CC baseline vs REN Direct SOC)")
    print(f"{'='*72}")
    print(f"  {'Metric':<16} {'REN Direct':>12}  {'CC':>12}  {'Improvement':>12}")
    print(f"  {'-'*56}")
    for rv, cv, lbl in [(ren_rmse,cc_rmse,"RMSE"),(ren_mae,cc_mae,"MAE")]:
        print(f"  {lbl:<16} {rv:>12.5f}  {cv:>12.5f}  {(1-rv/cv)*100:>+11.1f}%")

    print(f"\n  Training time  : {(time.time()-t0)/60:.1f} min")
    print(f"  Gate @ I=0A    : {model.gate_value_at_raw_amps(0.0):.6f}  "
          f"(apply ren_model.py gate floor fix before deployment)")
    print(f"  z0 norm        : {model.z0_norm():.4f}")

    # Training curves
    ep_x = list(range(1, len(t_losses)+1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep_x, t_losses, label="Train", color="steelblue", lw=1.5)
    axes[0].plot(ep_x, v_losses, label="Val",   color="tomato",    lw=1.5)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.4)
    axes[1].plot(ep_x, v_rmses, color="purple", lw=2, label="REN Direct RMSE")
    axes[1].axhline(cc_rmse, color="orange", ls="--", lw=1.5,
                    label=f"CC baseline = {cc_rmse:.4f}")
    axes[1].set_title("Val RMSE vs CC baseline"); axes[1].legend(); axes[1].grid(alpha=0.4)
    gate_vals = [r["gate_at_zero"] for r in log]
    axes[2].plot(ep_x, gate_vals, color="red", lw=1.5, label="gate @ I=0A (raw)")
    axes[2].axhline(0,    color="black", ls="--", lw=0.8)
    axes[2].axhline(0.15, color="gray",  ls=":",  lw=0.8, label="floor=0.15 (if gate fix applied)")
    axes[2].set_title("Gate @ I=0A"); axes[2].legend(); axes[2].grid(alpha=0.4)
    plt.suptitle("REN v4.0 — Direct SOC (no soc_cc, directional+slope loss, ZC disabled)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {SAVE_DIR}/ren_soc_best.pth")
    print(f"  Saved: {SAVE_DIR}/training_curves.png")


if __name__ == "__main__":
    main()