"""
REN Training Script — VRFB SOC Estimator  (v4, dual-mode)
==========================================================

Trains TWO models in a single run and compares both against CC:

  standalone  [5 features]
    V, I, T_stack, T_tank, Q
    No prior estimator needed. Minimum hardware.
    Scientific claim: REN can estimate SOC from raw sensors alone.

  hybrid      [8 features]
    V, I, T_stack, T_tank, Q, soc_cc, I_limit_approx, transport_ratio_approx
    I_limit_approx and transport_ratio_approx are derived from Q and I
    using a fixed mid-SOC concentration — no extra sensors required.
    Scientific claim: REN corrects CC drift using voltage + hydraulics.

OUTPUT:
  ren/standalone/scaler.pkl + ren_soc_best.pth + training_log.csv
  ren/hybrid/scaler.pkl    + ren_soc_best.pth + training_log.csv
  ren/training_curves_standalone.png
  ren/training_curves_hybrid.png
  ren/final_comparison.txt

USAGE:
  python -m ren.train_ren              # trains both
  python -m ren.train_ren --mode standalone
  python -m ren.train_ren --mode hybrid
"""

import os
import sys
import time
import pickle
import math
import argparse
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
# DUAL-MODE CONFIGURATION
# =============================================================================

BASE_FEATURES = [
    "voltage",
    "current",
    "temperature_stack",
    "temperature_tank",
    "flow_rate",
]

CONFIGS = {
    "standalone": {
        "feature_cols": BASE_FEATURES,
        "input_dim":    5,
        "description":  "5 raw sensors — no CC, no derived features",
        "save_dir":     "ren/standalone",
    },
    "hybrid": {
        "feature_cols": BASE_FEATURES + [
            "soc_cc",
            "I_limit_approx",
            "transport_ratio_approx",
        ],
        "input_dim":    8,
        "description":  "5 sensors + CC output + approx hydraulics",
        "save_dir":     "ren/hybrid",
    },
}

TARGET_COL = "SOC_true"

# Model (shared)
HIDDEN_DIM      = 128
ALPHA           = 0.5
DROPOUT         = 0.1
N_POWER_ITERS   = 10

# Training (shared)
SEQ_LEN          = 256
BATCH_SIZE       = 32
EPOCHS           = 80
LR               = 3e-4
LR_WARMUP_EPOCHS = 5
WEIGHT_DECAY     = 1e-5
PATIENCE         = 25
GRAD_CLIP        = 1.0
WARMUP_STEPS     = 32
P_Z0_RESET       = 0.30
MSE_WEIGHT       = 0.7
MAE_WEIGHT       = 0.3
BOUNDARY_SCALE   = 3.0
SMOOTH_WEIGHT    = 0.10
SMOOTH_I_REF     = 0.3    # scaled units after StandardScaler
HARD_CASE_WEIGHT = 3.0
HARD_CASE_THRESH = 0.10
SOC_MIN          = 0.05
SOC_MAX          = 0.95

TRAIN_CSV = "datasets/vrfb_train.csv"
TEST_CSV  = "datasets/vrfb_test.csv"
SAVE_DIR  = "ren"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATA
# =============================================================================

def load_episodes(df, scaler, feat_cols, fit_scaler=False):
    X_all = df[feat_cols].values.astype(np.float32)
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

def composite_loss(pred, target, current=None):
    err = pred - target
    with torch.no_grad():
        w_boundary = 1.0 + BOUNDARY_SCALE * (
            torch.exp(-50.0 * (target - SOC_MIN).clamp(min=0.0)) +
            torch.exp(-50.0 * (SOC_MAX - target).clamp(min=0.0))
        )
        w_hard = torch.where(
            err.abs() > HARD_CASE_THRESH,
            torch.full_like(err, HARD_CASE_WEIGHT),
            torch.ones_like(err),
        )
        w = w_boundary * w_hard
    base = MSE_WEIGHT * (w * err**2).mean() + MAE_WEIGHT * F.l1_loss(pred, target)
    if current is not None and SMOOTH_WEIGHT > 0 and pred.shape[1] > 1:
        delta = (pred[:, 1:] - pred[:, :-1]).abs()
        with torch.no_grad():
            iw = torch.exp(-current[:, 1:].abs() / SMOOTH_I_REF)
        return base + SMOOTH_WEIGHT * (iw * delta).mean()
    return base


# =============================================================================
# STATEFUL EPOCH
# =============================================================================

def run_stateful_epoch(model, episodes, optimiser=None):
    is_train = optimiser is not None
    model.train() if is_train else model.eval()
    perm       = torch.randperm(len(episodes)).tolist()
    total_loss = 0.0
    sq_errors, abs_errors = [], []
    n_batches  = 0
    ctx = torch.enable_grad if is_train else torch.no_grad

    for batch_start in range(0, len(perm), BATCH_SIZE):
        batch_idxs = perm[batch_start : batch_start + BATCH_SIZE]
        batch_eps  = [episodes[i] for i in batch_idxs]
        B          = len(batch_eps)
        min_len    = min(ep[0].shape[0] for ep in batch_eps)
        n_chunks   = min_len // SEQ_LEN
        if n_chunks == 0:
            continue

        z = model.z0.expand(B, -1).contiguous().to(DEVICE)
        if is_train:
            z = z.detach()

        batch_loss = 0.0
        for ci in range(n_chunks):
            s = ci * SEQ_LEN
            e = s + SEQ_LEN
            X_c = torch.stack([ep[0][s:e] for ep in batch_eps]).to(DEVICE)
            y_c = torch.stack([ep[1][s:e] for ep in batch_eps]).to(DEVICE)

            # z0 reset: pass z=None so forward() uses self.z0 with gradient
            use_z0 = is_train and ci > 0 and torch.rand(1).item() < P_Z0_RESET

            with ctx():
                y_pred, z_next = model(X_c, z=None if use_z0 else z)
                if WARMUP_STEPS > 0 and SEQ_LEN > WARMUP_STEPS:
                    yp = y_pred[:, WARMUP_STEPS:, :]
                    yc = y_c[:,    WARMUP_STEPS:, :]
                    ic = X_c[:,    WARMUP_STEPS:, 1:2]
                else:
                    yp, yc, ic = y_pred, y_c, X_c[:, :, 1:2]

                loss = composite_loss(yp, yc, current=ic)
                batch_loss += loss.item()
                err = (yp - yc).detach().cpu().numpy().ravel()
                sq_errors.append(err**2)
                abs_errors.append(np.abs(err))

                if is_train:
                    optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimiser.step()

            z = z_next.detach()

        total_loss += batch_loss / max(n_chunks, 1)
        n_batches  += 1

    sq_all  = np.concatenate(sq_errors)
    abs_all = np.concatenate(abs_errors)
    return (total_loss/max(n_batches,1), math.sqrt(sq_all.mean()),
            abs_all.mean(), abs_all.max())


# =============================================================================
# LR SCHEDULE
# =============================================================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup, total, min_ratio=0.01):
        self.warmup = warmup; self.total = total; self.min_ratio = min_ratio
        super().__init__(opt)
    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            scale = (e+1)/self.warmup
        else:
            p = (e-self.warmup)/max(self.total-self.warmup,1)
            scale = self.min_ratio + 0.5*(1-self.min_ratio)*(1+math.cos(math.pi*p))
        return [b*scale for b in self.base_lrs]


# =============================================================================
# SINGLE MODE TRAINING
# =============================================================================

def train_one_mode(mode_name, cfg, df_train, df_test, cc_rmse, cc_mae, cc_max):
    feat_cols = cfg["feature_cols"]
    input_dim = cfg["input_dim"]
    save_dir  = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  MODE: {mode_name.upper()}  —  {cfg['description']}")
    print(f"  Features ({input_dim}): {feat_cols}")
    print(f"{'='*65}")

    missing = [c for c in feat_cols + [TARGET_COL, "episode_id"]
               if c not in df_train.columns]
    if missing:
        raise ValueError(f"Missing columns for {mode_name}: {missing}\n"
                         f"Available: {list(df_train.columns)}\n"
                         f"Re-run dataset_gen.py v4.")

    scaler    = StandardScaler()
    train_eps = load_episodes(df_train, scaler, feat_cols, fit_scaler=True)
    test_eps  = load_episodes(df_test,  scaler, feat_cols, fit_scaler=False)

    scaler_path = os.path.join(save_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler  → {scaler_path}")
    print(f"  Train episodes: {len(train_eps)}   chunks/ep: ~{train_eps[0][0].shape[0]//SEQ_LEN}")

    model = REN(
        input_dim     = input_dim,
        hidden_dim    = HIDDEN_DIM,
        output_dim    = 1,
        alpha         = ALPHA,
        dropout       = DROPOUT,
        n_power_iters = N_POWER_ITERS,
        use_feedthrough = False,   # D(x_t) removed from output path
    ).to(DEVICE)

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimiser, LR_WARMUP_EPOCHS, EPOCHS)

    print(f"\n  {'Ep':>4}  {'Trn':>10}  {'Val':>10}  {'EMA':>10}  "
          f"{'RMSE':>8}  {'MAE':>8}  {'CR':>7}  {'z0':>6}  {'LR':>9}")
    print(f"  {'-'*85}")

    best_loss = math.inf; ema_loss = math.inf; patience_ct = 0
    log = []; t_losses = []; v_losses = []; v_rmses = []; lrs = []
    t0 = time.time()

    for epoch in range(1, EPOCHS+1):
        t_ep = time.time()
        tl, tr, tm, _ = run_stateful_epoch(model, train_eps, optimiser)
        vl, vr, vm, vx = run_stateful_epoch(model, test_eps)
        scheduler.step()

        ema_loss = vl if epoch==1 else 0.7*ema_loss + 0.3*vl
        cr = model.contraction_rate()
        z0n = model.z0_norm()
        lr = scheduler.get_last_lr()[0]

        t_losses.append(tl); v_losses.append(vl)
        v_rmses.append(vr);  lrs.append(lr)
        log.append(dict(epoch=epoch, trn_loss=tl, val_loss=vl, ema_loss=ema_loss,
                        val_rmse=vr, val_mae=vm, val_max=vx,
                        contraction_rate=cr, z0_norm=z0n, lr=lr))

        marker = ""
        if ema_loss < best_loss:
            best_loss = ema_loss; patience_ct = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "ren_soc_best.pth"))
            marker = " ◀"
        else:
            patience_ct += 1

        print(f"  {epoch:>4d}  {tl:>10.6f}  {vl:>10.6f}  {ema_loss:>10.6f}  "
              f"{vr:>8.5f}  {vm:>8.5f}  {cr:>7.4f}  {z0n:>6.3f}  {lr:>9.2e}"
              f"  [{time.time()-t_ep:.0f}s]{marker}")

        if patience_ct >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, "ren_soc_last.pth"))
    pd.DataFrame(log).to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

    # Final eval with best model
    model.load_state_dict(torch.load(os.path.join(save_dir, "ren_soc_best.pth"),
                                     map_location=DEVICE))
    _, ren_rmse, ren_mae, ren_max = run_stateful_epoch(model, test_eps)

    print(f"\n  {'─'*55}")
    print(f"  {mode_name.upper()} FINAL  vs CC baseline")
    print(f"  {'Metric':<16} {'REN':>10}  {'CC':>10}  {'Improv':>10}")
    print(f"  {'─'*50}")
    for val, base, label in [(ren_rmse,cc_rmse,"RMSE"),
                              (ren_mae, cc_mae, "MAE"),
                              (ren_max, cc_max, "Max Error")]:
        imp = (1-val/base)*100
        print(f"  {label:<16} {val:>10.5f}  {base:>10.5f}  {imp:>+9.1f}%")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  Best EMA val loss: {best_loss:.6f}")
    print(f"  Final z0 norm: {model.z0_norm():.4f}")

    # Training curve
    ep_x = list(range(1, len(t_losses)+1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep_x, t_losses, label="Train", color="steelblue", lw=1.5)
    axes[0].plot(ep_x, v_losses, label="Val",   color="tomato",    lw=1.5)
    axes[0].set_title(f"{mode_name} — Loss"); axes[0].legend(); axes[0].grid(alpha=0.4)
    axes[1].plot(ep_x, v_rmses, color="purple", lw=2, label="REN RMSE")
    axes[1].axhline(cc_rmse, color="orange", ls="--", lw=1.5, label=f"CC={cc_rmse:.4f}")
    axes[1].set_title("Val RMSE vs CC"); axes[1].legend(); axes[1].grid(alpha=0.4)
    axes[2].plot(ep_x, lrs, color="green", lw=1.5)
    axes[2].set_title("LR Schedule"); axes[2].grid(alpha=0.4)
    plt.suptitle(f"REN {mode_name} — VRFB SOC", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"training_curves_{mode_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → ren/training_curves_{mode_name}.png")

    return {"rmse": ren_rmse, "mae": ren_mae, "max": ren_max}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone","hybrid","both"],
                        default="both")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  REN Training v4 — Dual Mode (standalone + hybrid)")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS}  Patience: {PATIENCE}  Batch: {BATCH_SIZE}")
    print(f"  Loss: {MSE_WEIGHT}×MSE(boundary+hard-case) + {MAE_WEIGHT}×MAE + {SMOOTH_WEIGHT}×smooth")
    print(f"{'='*65}")

    print("\nLoading data...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    print(f"  Train: {len(df_train):,} rows  ({df_train['episode_id'].nunique()} eps)")
    print(f"  Test:  {len(df_test):,} rows  ({df_test['episode_id'].nunique()} eps)")

    cc_errors = np.abs(df_test[TARGET_COL].values - df_test["soc_cc"].values)
    cc_rmse = math.sqrt(np.mean(cc_errors**2))
    cc_mae  = cc_errors.mean()
    cc_max  = cc_errors.max()
    print(f"\n  CC baseline — RMSE: {cc_rmse:.5f}  MAE: {cc_mae:.5f}  Max: {cc_max:.5f}")

    modes_to_run = (["standalone","hybrid"] if args.mode=="both"
                    else [args.mode])
    results = {}
    for mode in modes_to_run:
        results[mode] = train_one_mode(
            mode, CONFIGS[mode], df_train, df_test, cc_rmse, cc_mae, cc_max
        )

    # Final comparison table
    if len(results) == 2:
        print(f"\n{'='*65}")
        print(f"  FINAL COMPARISON — CC vs Standalone vs Hybrid")
        print(f"{'='*65}")
        print(f"  {'Metric':<16} {'CC':>10}  {'Standalone':>12}  {'Hybrid':>10}")
        print(f"  {'─'*55}")
        for metric, label in [("rmse","RMSE"),("mae","MAE"),("max","Max Error")]:
            cc_v  = {"rmse":cc_rmse,"mae":cc_mae,"max":cc_max}[metric]
            sa_v  = results["standalone"][metric]
            hy_v  = results["hybrid"][metric]
            print(f"  {label:<16} {cc_v:>10.5f}  "
                  f"{sa_v:>10.5f}({(1-sa_v/cc_v)*100:>+5.1f}%)  "
                  f"{hy_v:>8.5f}({(1-hy_v/cc_v)*100:>+5.1f}%)")

        summary = (
            f"CC RMSE={cc_rmse:.5f} MAE={cc_mae:.5f}\n"
            f"Standalone RMSE={results['standalone']['rmse']:.5f} "
            f"MAE={results['standalone']['mae']:.5f}\n"
            f"Hybrid RMSE={results['hybrid']['rmse']:.5f} "
            f"MAE={results['hybrid']['mae']:.5f}\n"
        )
        with open(os.path.join(SAVE_DIR, "final_comparison.txt"), "w") as f:
            f.write(summary)
        print(f"\n  Saved → ren/final_comparison.txt")


if __name__ == "__main__":
    main()