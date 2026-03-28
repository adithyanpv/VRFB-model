"""
Hybrid REN Inference & Diagnostic Script
=========================================
Proves that CC + REN correction > CC alone.

Mirrors the exact inference pipeline in server.py:
  - 7 features: voltage, current, T_stack, T_tank, flow_rate, soc_cc, tr_approx
  - REN predicts correction (SOC_true - soc_cc) trained as residual
  - Final SOC = clip(soc_cc + correction, 0, 1)
  - EMA smoothing tau=30 (matching server.py)
  - z carried forward per episode from model.z0

OUTPUTS
-------
  ren/hybrid_plots/episode_XX.png        per-episode 5-panel plots
  ren/hybrid_plots/aggregate_scatter.png predicted vs true scatter
  ren/hybrid_plots/error_distribution.png error histogram CC vs CC+REN
  ren/hybrid_plots/soc_band_errors.png   RMSE by SOC band
  ren/hybrid_plots/episode_rmse_bar.png  per-episode RMSE bar chart
  ren/hybrid_plots/zero_current_check.png gate/ZC constraint validation
  ren/hybrid_plots/correction_analysis.png correction magnitude analysis
  ren/hybrid_metrics.csv                 per-episode numeric table
  ren/hybrid_summary.txt                 overall proof summary

USAGE
-----
  python infer_hybrid.py
  python infer_hybrid.py --episodes 6   # plot first 6 episodes only
  python infer_hybrid.py --no-plots     # metrics only, skip plot generation
"""

import argparse
import math
import os
import pickle
import warnings
import builtins

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from ren.ren_model import REN

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# MUST MATCH train_ren.py EXACTLY
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "voltage",
    "current",
    "temperature_stack",
    "temperature_tank",
    "flow_rate",
    "soc_cc",
    "transport_ratio_approx",
]
TARGET_COL   = "SOC_true"        # absolute SOC ground truth
CURRENT_IDX  = 1                 # index of "current" in FEATURE_COLS
INPUT_DIM    = len(FEATURE_COLS) # 7
HIDDEN_DIM   = 128
ALPHA        = 0.5
EMA_TAU      = 30                # matches server.py

# I_limit approx constants — matches server.py and dataset_gen.py
_IL_CONST = 1 * 96485.0 * 2e-5 * 0.15 * 1600.0 * 0.5  # ~231.6 A at Q_ref
_Q_REF    = 20.0 / 60000.0  # 20 LPM in m³/s

TEST_CSV    = "datasets/vrfb_test.csv"
SCALER_PATH = "ren/scaler.pkl"
MODEL_PATH  = "ren/ren_soc_best.pth"
OUT_DIR     = "ren"
PLOT_DIR    = os.path.join(OUT_DIR, "hybrid_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plot colours — consistent with dashboard
C_TRUE = "#00e5ff"   # cyan  — ground truth
C_CC   = "#ff8c42"   # orange — Coulomb counter
C_REN  = "#00ff9d"   # green  — CC + REN hybrid

plt.rcParams.update({
    "font.size":   11,
    "axes.grid":   True,
    "grid.alpha":  0.3,
    "figure.dpi":  130,
})


# =============================================================================
# METRICS
# =============================================================================

def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    return {
        "rmse"     : math.sqrt(np.mean(err ** 2)),
        "mae"      : np.mean(np.abs(err)),
        "max_err"  : np.max(np.abs(err)),
        "mean_bias": np.mean(err),
        "std_err"  : np.std(err),
    }


def improvement(base_val: float, new_val: float) -> str:
    pct = (1.0 - new_val / base_val) * 100.0
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


# =============================================================================
# INFERENCE — mirrors server.py _ren_step exactly
# =============================================================================

def run_episode(model, scaler, X_ep_raw: np.ndarray,
                soc_cc_ep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one full episode carrying z forward from model.z0.

    Parameters
    ----------
    X_ep_raw  : (T, 7) unscaled features matching FEATURE_COLS
    soc_cc_ep : (T,) Coulomb counter SOC

    Returns
    -------
    soc_hybrid : (T,) CC + REN correction, EMA-smoothed, clipped [0,1]
    correction : (T,) raw REN correction output (pre-EMA, pre-clip)
    soc_hybrid_raw : (T,) hybrid before EMA (for diagnostics)
    """
    model.eval()

    X_scaled = scaler.transform(X_ep_raw).astype(np.float32)
    T = X_scaled.shape[0]

    # Start from learned z0 — matches server._reset_ren_state()
    with torch.no_grad():
        z = model.z0.clone()  # (1, hidden_dim)

    EMA_ALPHA = 1.0 / EMA_TAU
    ema_soc   = float(soc_cc_ep[0])   # warm-started at first CC value

    soc_hybrid     = np.zeros(T, dtype=np.float32)
    soc_hybrid_raw = np.zeros(T, dtype=np.float32)
    corrections    = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for t in range(T):
            x_t = torch.tensor(X_scaled[t:t+1], dtype=torch.float32).unsqueeze(0)
            # x_raw = scaled X (matches train_ren.py: Xc_raw = Xc.clone())
            y_t, z = model(x_t, z=z, x_raw=x_t)
            corr = float(y_t.squeeze())

            raw_hybrid = float(np.clip(soc_cc_ep[t] + corr, 0.0, 1.0))
            ema_soc    = (1.0 - EMA_ALPHA) * ema_soc + EMA_ALPHA * raw_hybrid

            corrections[t]    = corr
            soc_hybrid_raw[t] = raw_hybrid
            soc_hybrid[t]     = float(np.clip(ema_soc, 0.0, 1.0))

    return soc_hybrid, corrections, soc_hybrid_raw


# =============================================================================
# PLOTS — per episode
# =============================================================================

def plot_episode(ep_id, time_s, soc_true, soc_cc, soc_hybrid,
                 corrections, m_cc, m_hybrid, save_path):
    """5-panel diagnostic plot for one episode."""
    time_h   = time_s / 3600.0
    err_cc   = soc_cc      - soc_true
    err_hyb  = soc_hybrid  - soc_true

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    # ── Panel 1: SOC traces (full width) ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_h, soc_true,   color=C_TRUE, lw=2.0,  label="Ground truth (SOC_true)", zorder=3)
    ax1.plot(time_h, soc_cc,     color=C_CC,   lw=1.2,  alpha=0.85,
             label=f"Coulomb Counter   RMSE={m_cc['rmse']:.4f}", zorder=2)
    ax1.plot(time_h, soc_hybrid, color=C_REN,  lw=1.5,  alpha=0.90,
             label=f"CC + REN hybrid   RMSE={m_hybrid['rmse']:.4f}", zorder=2)
    ax1.set_ylabel("State of Charge")
    ax1.set_title(f"Episode {ep_id}  —  SOC Comparison",
                  fontweight="bold", fontsize=13)
    ax1.legend(loc="best", fontsize=9)
    ax1.set_ylim(-0.03, 1.03)

    # ── Panel 2: Absolute error ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_h, np.abs(err_cc),  color=C_CC,  lw=1.0, alpha=0.85,
             label=f"CC   MAE={m_cc['mae']:.4f}")
    ax2.plot(time_h, np.abs(err_hyb), color=C_REN, lw=1.0, alpha=0.85,
             label=f"Hybrid MAE={m_hybrid['mae']:.4f}")
    ax2.set_ylabel("|Error|  (SOC units)")
    ax2.set_title("Absolute Error Over Time")
    ax2.legend(fontsize=9)
    ax2.set_xlabel("Time (hours)")

    # ── Panel 3: Signed error ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax3.plot(time_h, err_cc,  color=C_CC,  lw=1.0, alpha=0.85,
             label=f"CC    bias={m_cc['mean_bias']:+.4f}")
    ax3.plot(time_h, err_hyb, color=C_REN, lw=1.0, alpha=0.85,
             label=f"Hybrid bias={m_hybrid['mean_bias']:+.4f}")
    ax3.set_ylabel("Signed Error  (pred − true)")
    ax3.set_title("Signed Error  (bias analysis)")
    ax3.legend(fontsize=9)
    ax3.set_xlabel("Time (hours)")

    # ── Panel 4: Running MAE ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    n = np.arange(1, len(err_cc) + 1)
    ax4.plot(time_h, np.cumsum(np.abs(err_cc))  / n,
             color=C_CC,  lw=1.2, label="CC  running MAE")
    ax4.plot(time_h, np.cumsum(np.abs(err_hyb)) / n,
             color=C_REN, lw=1.2, label="Hybrid running MAE")
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Running MAE")
    ax4.set_title("Cumulative MAE  (does error grow?)")
    ax4.legend(fontsize=9)

    # ── Panel 5: REN correction magnitude ─────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.plot(time_h, corrections, color="#b388ff", lw=0.8, alpha=0.85)
    ax5.axhline(0, color="white", lw=0.8, ls="--", alpha=0.4)
    ax5.fill_between(time_h, corrections, 0,
                     where=corrections > 0, color="#b388ff", alpha=0.2,
                     label="Positive correction (REN > CC)")
    ax5.fill_between(time_h, corrections, 0,
                     where=corrections < 0, color="#ff3d5a", alpha=0.2,
                     label="Negative correction (REN < CC)")
    ax5.set_xlabel("Time (hours)")
    ax5.set_ylabel("REN correction  (SOC units)")
    ax5.set_title("REN correction applied to CC")
    ax5.legend(fontsize=9)

    # ── Panel 6: Metrics table ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    rows = []
    for lbl, mk, mc in [("RMSE", "rmse", "rmse"),
                          ("MAE",  "mae",  "mae"),
                          ("Max Err", "max_err", "max_err"),
                          ("Bias", "mean_bias", "mean_bias")]:
        rv = m_hybrid[mk]; cv = m_cc[mc]
        if lbl != "Bias":
            imp = f"{improvement(cv, rv)}"
            fc  = "#d4edda" if float(imp.replace('%','')) > 0 else "#f8d7da"
        else:
            imp = "—"; fc = "white"
        rows.append([lbl, f"{rv:.5f}", f"{cv:.5f}", imp, fc])

    tbl = ax6.table(
        cellText  = [[r[0], r[1], r[2], r[3]] for r in rows],
        colLabels = ["Metric", "CC+REN", "CC", "Improvement"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.8)
    for j in range(4):
        tbl[(0, j)].set_facecolor("#0d1525")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, r in enumerate(rows, start=1):
        if r[3] != "—":
            tbl[(i, 3)].set_facecolor(r[4])
    ax6.set_title("Episode Metrics", fontweight="bold", pad=8)

    plt.suptitle(f"VRFB SOC Estimation — Episode {ep_id}  |  "
                 f"CC+REN vs CC vs Ground Truth",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# =============================================================================
# AGGREGATE PLOTS
# =============================================================================

def plot_scatter(true, pred_hybrid, pred_cc, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, pred, lbl, col in [
        (axes[0], pred_cc,     "Coulomb Counter", C_CC),
        (axes[1], pred_hybrid, "CC + REN Hybrid", C_REN),
    ]:
        m = metrics(true, pred)
        ax.scatter(true, pred, c=col, s=1, alpha=0.2, rasterized=True)
        lo, hi = true.min(), true.max()
        ax.plot([lo, hi], [lo, hi], "w--", lw=1.2, label="Perfect")
        ax.set_xlabel("SOC True")
        ax.set_ylabel("SOC Predicted")
        ax.set_title(f"{lbl}\nRMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}")
        ax.legend(fontsize=9)
        ax.set_xlim(lo - 0.02, hi + 0.02)
        ax.set_ylim(lo - 0.02, hi + 0.02)
    plt.suptitle("Predicted vs True SOC — Scatter", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_error_distribution(err_hybrid, err_cc, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Signed error histograms
    bins = np.linspace(-0.35, 0.35, 100)
    axes[0].hist(err_cc,     bins=bins, color=C_CC,  alpha=0.65, label="CC",          density=True)
    axes[0].hist(err_hybrid, bins=bins, color=C_REN, alpha=0.65, label="CC + REN",    density=True)
    axes[0].axvline(0, color="white", lw=1.2, ls="--")
    axes[0].set_xlabel("Signed error  (pred − true)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Signed Error Distribution")
    axes[0].legend()

    # Absolute error CDF
    abs_cc  = np.sort(np.abs(err_cc))
    abs_hyb = np.sort(np.abs(err_hybrid))
    cdf = np.linspace(0, 1, len(abs_cc))
    axes[1].plot(abs_cc,  cdf, color=C_CC,  lw=2, label=f"CC   (MAE={abs_cc.mean():.4f})")
    cdf2 = np.linspace(0, 1, len(abs_hyb))
    axes[1].plot(abs_hyb, cdf2, color=C_REN, lw=2, label=f"CC+REN (MAE={abs_hyb.mean():.4f})")
    axes[1].set_xlabel("|Error|  (SOC units)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative Error Distribution\n(curve to the LEFT = better)")
    axes[1].legend()

    plt.suptitle("Error Analysis — CC+REN vs CC", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_episode_rmse_bar(ep_ids, rmse_cc, rmse_hybrid, save_path):
    x   = np.arange(len(ep_ids))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(ep_ids) * 0.6), 5))
    bars_cc  = ax.bar(x - w/2, rmse_cc,     w, color=C_CC,  alpha=0.85, label="CC")
    bars_hyb = ax.bar(x + w/2, rmse_hybrid, w, color=C_REN, alpha=0.85, label="CC + REN")
    ax.axhline(np.mean(rmse_cc),     color=C_CC,  ls="--", lw=1.2,
               label=f"CC mean = {np.mean(rmse_cc):.4f}")
    ax.axhline(np.mean(rmse_hybrid), color=C_REN, ls="--", lw=1.2,
               label=f"Hybrid mean = {np.mean(rmse_hybrid):.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in ep_ids], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Episode ID")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-Episode RMSE — CC vs CC+REN Hybrid", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_soc_band_errors(true_all, pred_hybrid, pred_cc, save_path):
    bands  = np.arange(0, 1.0, 0.1)
    hyb_b  = []; cc_b = []; labels = []
    for lo in bands:
        hi   = lo + 0.1
        mask = (true_all >= lo) & (true_all < hi)
        if mask.sum() < 100:
            continue
        hyb_b.append(math.sqrt(np.mean((pred_hybrid[mask] - true_all[mask]) ** 2)))
        cc_b.append( math.sqrt(np.mean((pred_cc[mask]     - true_all[mask]) ** 2)))
        labels.append(f"{lo:.1f}–{hi:.1f}")

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, cc_b,  w, color=C_CC,  alpha=0.85, label="CC")
    ax.bar(x + w/2, hyb_b, w, color=C_REN, alpha=0.85, label="CC + REN")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("SOC Band")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by SOC Band — CC vs CC+REN\n"
                 "(shows where hybrid improves most)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_zero_current_check(df_test, true_all, pred_hybrid, pred_cc, save_path):
    """
    Validates the zero-current invariance constraint.
    At I=0, dSOC/dt must be 0 — SOC must not change.
    Plots: step-to-step SOC change vs current magnitude for both estimators.
    """
    I_vals    = df_test["current"].values
    zero_mask = np.abs(I_vals) < 2.0   # threshold: 2A dead-band

    # Step-to-step changes
    delta_hyb = np.abs(np.diff(pred_hybrid, prepend=pred_hybrid[0]))
    delta_cc  = np.abs(np.diff(pred_cc,     prepend=pred_cc[0]))
    delta_true= np.abs(np.diff(true_all,    prepend=true_all[0]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: |ΔSOC| distribution at I=0
    bins = np.linspace(0, 0.01, 60)
    axes[0].hist(delta_cc[zero_mask],  bins=bins, color=C_CC,   alpha=0.7,
                 label=f"CC   mean={delta_cc[zero_mask].mean():.5f}", density=True)
    axes[0].hist(delta_hyb[zero_mask], bins=bins, color=C_REN,  alpha=0.7,
                 label=f"CC+REN mean={delta_hyb[zero_mask].mean():.5f}", density=True)
    axes[0].hist(delta_true[zero_mask],bins=bins, color=C_TRUE, alpha=0.5,
                 label=f"True  mean={delta_true[zero_mask].mean():.5f}", density=True)
    axes[0].set_xlabel("|ΔSOC| per step")
    axes[0].set_ylabel("Density")
    axes[0].set_title("SOC change distribution when |I| < 2A\n"
                      "(should be near zero — gate constraint)", fontweight="bold")
    axes[0].legend(fontsize=9)

    # Right: mean |ΔSOC| vs current bin
    I_bins  = np.arange(0, 200, 10)
    hyb_mu  = []; cc_mu = []; true_mu = []
    for lo in I_bins:
        mask = (np.abs(I_vals) >= lo) & (np.abs(I_vals) < lo + 10)
        if mask.sum() < 50:
            hyb_mu.append(np.nan); cc_mu.append(np.nan); true_mu.append(np.nan)
        else:
            hyb_mu.append(delta_hyb[mask].mean())
            cc_mu.append(delta_cc[mask].mean())
            true_mu.append(delta_true[mask].mean())

    axes[1].plot(I_bins + 5, true_mu, color=C_TRUE, lw=2,   label="Ground truth")
    axes[1].plot(I_bins + 5, cc_mu,   color=C_CC,   lw=1.5, label="CC")
    axes[1].plot(I_bins + 5, hyb_mu,  color=C_REN,  lw=1.5, label="CC + REN")
    axes[1].set_xlabel("|Current| bin (A)")
    axes[1].set_ylabel("Mean |ΔSOC| per step")
    axes[1].set_title("SOC step-change vs current magnitude\n"
                      "(physics: larger I → larger ΔSOC)", fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.suptitle("Zero-Current Invariance Constraint Validation", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_correction_analysis(df_test, corrections_all, true_all, pred_cc, save_path):
    """
    Shows what the REN correction is doing:
    - Correction vs CC error (should be positively correlated — REN corrects CC)
    - Correction distribution by SOC band
    - Correction magnitude over time (do corrections grow or shrink?)
    """
    cc_err   = pred_cc - true_all   # signed CC error
    corr     = corrections_all       # REN correction
    I_vals   = df_test["current"].values
    soc_true = true_all

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: REN correction vs CC error scatter
    sample = np.random.choice(len(corr), size=min(50000, len(corr)), replace=False)
    axes[0].scatter(cc_err[sample], corr[sample], s=1, alpha=0.15,
                    c=C_REN, rasterized=True)
    lim = max(np.abs(cc_err).max(), np.abs(corr).max()) * 0.8
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].axhline(0, color="white", lw=0.8, ls="--", alpha=0.4)
    axes[0].axvline(0, color="white", lw=0.8, ls="--", alpha=0.4)
    axes[0].plot([-lim, lim], [-lim, lim], "w--", lw=1, alpha=0.6,
                 label="Perfect correction")
    # Compute correlation
    corr_coef = np.corrcoef(cc_err[sample], corr[sample])[0, 1]
    axes[0].set_xlabel("CC signed error  (CC − true)")
    axes[0].set_ylabel("REN correction")
    axes[0].set_title(f"REN correction vs CC error\n"
                      f"Pearson r = {corr_coef:.3f}  "
                      f"(+1 = perfect, 0 = no info)", fontweight="bold")
    axes[0].legend(fontsize=8)

    # Middle: Correction distribution by SOC band
    soc_bands   = [(0.05,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,0.95)]
    band_labels = ["5–20%", "20–40%", "40–60%", "60–80%", "80–95%"]
    band_corrs  = []
    for lo, hi in soc_bands:
        mask = (soc_true >= lo) & (soc_true < hi)
        band_corrs.append(corr[mask])
    axes[1].boxplot(band_corrs, labels=band_labels, patch_artist=True,
                    boxprops=dict(facecolor=C_REN, alpha=0.4),
                    medianprops=dict(color="white", lw=2))
    axes[1].axhline(0, color="white", lw=1, ls="--", alpha=0.5)
    axes[1].set_xlabel("SOC band")
    axes[1].set_ylabel("REN correction")
    axes[1].set_title("Correction distribution by SOC band\n"
                      "(positive = REN raises CC estimate)", fontweight="bold")

    # Right: correction at I=0 vs I≠0
    zero = np.abs(I_vals) < 2.0
    axes[2].hist(np.abs(corr[zero]),          bins=50, color=C_CC,  alpha=0.75, density=True,
                 label=f"|I|<2A   mean={np.abs(corr[zero]).mean():.5f}")
    axes[2].hist(np.abs(corr[~zero]),         bins=50, color=C_REN, alpha=0.75, density=True,
                 label=f"|I|≥2A   mean={np.abs(corr[~zero]).mean():.5f}")
    axes[2].set_xlabel("|correction| magnitude")
    axes[2].set_ylabel("Density")
    axes[2].set_title("|Correction| at rest vs active current\n"
                      "(gate: smaller correction at I=0)", fontweight="bold")
    axes[2].legend(fontsize=9)

    plt.suptitle("REN Correction Behaviour Analysis", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main(max_ep_plots: int = 999, no_plots: bool = False):

    print(f"\n{'='*68}")
    print(f"  Hybrid REN Inference  —  CC + REN vs CC")
    print(f"  Proving: CC + REN correction > CC alone")
    print(f"{'='*68}")

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun train_ren.py first."
        )
    model = REN(
        input_dim        = INPUT_DIM,
        hidden_dim       = HIDDEN_DIM,
        output_dim       = 1,
        alpha            = ALPHA,
        dropout          = 0.0,          # disabled at inference
        n_power_iters    = 10,
        use_feedthrough  = False,
        use_current_gate = True,
        current_feat_idx = CURRENT_IDX,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    print(f"\n  Model        : {MODEL_PATH}")
    print(f"  Parameters   : {model.count_parameters():,}")
    print(f"  σ(A_bar)     : {model.contraction_rate():.4f}  (< {1-ALPHA:.2f})")
    print(f"  z0 norm      : {model.z0_norm():.4f}")

    # ── Load scaler ───────────────────────────────────────────────────────
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    I_mean = scaler.mean_[CURRENT_IDX]
    I_std  = scaler.scale_[CURRENT_IDX]
    gate_at_zero = model.gate_value_at(abs(-I_mean / I_std))
    print(f"  Gate @ I=0A  : {gate_at_zero:.4f}  (target: < 0.1)")
    print(f"  Scaler       : {SCALER_PATH}")

    # ── Load test data ────────────────────────────────────────────────────
    print(f"\n  Loading {TEST_CSV} ...")
    df = pd.read_csv(TEST_CSV)

    # Validate all required columns exist
    missing = [c for c in FEATURE_COLS + ["SOC_true", "soc_cc", "episode_id", "time"]
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in test CSV: {missing}\n"
            f"Re-run dataset_gen.py — these columns must be present."
        )

    ep_ids = sorted(df["episode_id"].unique())
    print(f"  Episodes     : {len(ep_ids)}")
    print(f"  Total rows   : {len(df):,}")

    # ── Per-episode inference ─────────────────────────────────────────────
    print(f"\n  {'Ep':>4}  {'CC RMSE':>10}  {'Hyb RMSE':>10}  "
          f"{'CC MAE':>9}  {'Hyb MAE':>9}  {'Δ RMSE':>9}  {'Improved':>8}")
    print(f"  {'-'*70}")

    all_true = []; all_cc = []; all_hyb = []; all_corr = []
    ep_log   = []
    rmse_cc_list = []; rmse_hyb_list = []

    for i, ep_id in enumerate(ep_ids):
        ep_df = df[df["episode_id"] == ep_id].reset_index(drop=True)

        time_s   = ep_df["time"].values
        soc_true = ep_df["SOC_true"].values
        soc_cc   = ep_df["soc_cc"].values
        X_ep     = ep_df[FEATURE_COLS].values.astype(np.float32)

        soc_hybrid, corrections, soc_hybrid_raw = run_episode(
            model, scaler, X_ep, soc_cc
        )

        m_cc  = metrics(soc_true, soc_cc)
        m_hyb = metrics(soc_true, soc_hybrid)
        delta_rmse = m_hyb["rmse"] - m_cc["rmse"]
        improved   = "✓" if delta_rmse < 0 else "✗"

        print(f"  {ep_id:>4d}  {m_cc['rmse']:>10.5f}  {m_hyb['rmse']:>10.5f}  "
              f"{m_cc['mae']:>9.5f}  {m_hyb['mae']:>9.5f}  "
              f"{delta_rmse:>+9.5f}  {improved}")

        all_true.append(soc_true); all_cc.append(soc_cc)
        all_hyb.append(soc_hybrid); all_corr.append(corrections)
        rmse_cc_list.append(m_cc["rmse"])
        rmse_hyb_list.append(m_hyb["rmse"])

        ep_log.append({
            "episode_id"       : ep_id,
            "cc_rmse"          : m_cc["rmse"],
            "hyb_rmse"         : m_hyb["rmse"],
            "cc_mae"           : m_cc["mae"],
            "hyb_mae"          : m_hyb["mae"],
            "cc_max_err"       : m_cc["max_err"],
            "hyb_max_err"      : m_hyb["max_err"],
            "cc_bias"          : m_cc["mean_bias"],
            "hyb_bias"         : m_hyb["mean_bias"],
            "rmse_improvement" : (1 - m_hyb["rmse"] / m_cc["rmse"]) * 100,
            "mae_improvement"  : (1 - m_hyb["mae"]  / m_cc["mae"])  * 100,
            "n_improved_steps" : int((np.abs(soc_hybrid - soc_true) <
                                      np.abs(soc_cc      - soc_true)).sum()),
            "frac_steps_improved": float((np.abs(soc_hybrid - soc_true) <
                                          np.abs(soc_cc      - soc_true)).mean()),
            "corr_vs_cc_err_r" : float(np.corrcoef(
                                       soc_cc - soc_true, corrections)[0, 1]),
        })

        if not no_plots and i < max_ep_plots:
            plot_episode(
                ep_id, time_s, soc_true, soc_cc, soc_hybrid,
                corrections, m_cc, m_hyb,
                os.path.join(PLOT_DIR, f"episode_{ep_id:03d}.png")
            )

    # ── Aggregate ─────────────────────────────────────────────────────────
    true_all = np.concatenate(all_true)
    cc_all   = np.concatenate(all_cc)
    hyb_all  = np.concatenate(all_hyb)
    corr_all = np.concatenate(all_corr)
    err_cc   = cc_all  - true_all
    err_hyb  = hyb_all - true_all

    ov_cc  = metrics(true_all, cc_all)
    ov_hyb = metrics(true_all, hyb_all)

    n_eps_improved = sum(1 for r in ep_log if r["rmse_improvement"] > 0)

    # ── Overall results table ─────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  OVERALL TEST SET RESULTS")
    print(f"{'='*68}")
    print(f"  {'Metric':<20} {'CC+REN':>12}  {'CC':>12}  {'Improvement':>13}")
    print(f"  {'-'*62}")
    for key, lbl in [("rmse","RMSE"), ("mae","MAE"),
                     ("max_err","Max Error"), ("mean_bias","Mean Bias"),
                     ("std_err", "Error Std")]:
        rv = ov_hyb[key]; cv = ov_cc[key]
        imp = improvement(cv, rv) if key not in ("mean_bias","std_err") else "—"
        print(f"  {lbl:<20} {rv:>12.5f}  {cv:>12.5f}  {imp:>13}")

    print(f"\n  Episodes improved (RMSE)   : {n_eps_improved} / {len(ep_ids)}"
          f"  ({100*n_eps_improved/len(ep_ids):.0f}%)")
    print(f"  Steps with lower error     : "
          f"{(np.abs(hyb_all-true_all) < np.abs(cc_all-true_all)).mean()*100:.1f}% of all steps")
    print(f"  Mean |correction|          : {np.abs(corr_all).mean():.5f} SOC units")
    print(f"  Corr(correction, CC error) : "
          f"{np.corrcoef(cc_all-true_all, corr_all)[0,1]:.4f}  (+1 = perfect)")
    print(f"  Gate @ I=0A                : {gate_at_zero:.4f}  (target: << 0.1)")

    # ── Verdict ───────────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    rmse_imp_pct = (1 - ov_hyb["rmse"] / ov_cc["rmse"]) * 100
    mae_imp_pct  = (1 - ov_hyb["mae"]  / ov_cc["mae"])  * 100
    verdict = "CC + REN HYBRID IS BETTER" if rmse_imp_pct > 0 else "CC IS STILL BETTER"
    print(f"  VERDICT: {verdict}")
    print(f"  RMSE improvement: {rmse_imp_pct:+.2f}%")
    print(f"  MAE  improvement: {mae_imp_pct:+.2f}%")
    print(f"  {'='*60}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(OUT_DIR, "hybrid_metrics.csv")
    pd.DataFrame(ep_log).to_csv(metrics_path, index=False)
    print(f"\n  Saved → {metrics_path}")

    # ── Save summary ──────────────────────────────────────────────────────
    summary_lines = [
        "CC + REN Hybrid vs Coulomb Counter — Test Set Summary",
        "=" * 56,
        f"Test episodes    : {len(ep_ids)}",
        f"Total rows       : {len(true_all):,}",
        f"Gate @ I=0A      : {gate_at_zero:.4f}  (target < 0.1)",
        "",
        f"{'Metric':<20} {'CC+REN':>10}  {'CC':>10}  {'Improvement':>13}",
        "-" * 58,
    ]
    for key, lbl in [("rmse","RMSE"), ("mae","MAE"),
                     ("max_err","Max Error"), ("mean_bias","Mean Bias")]:
        rv = ov_hyb[key]; cv = ov_cc[key]
        imp = improvement(cv, rv) if key != "mean_bias" else "—"
        summary_lines.append(f"{lbl:<20} {rv:>10.5f}  {cv:>10.5f}  {imp:>13}")
    summary_lines += [
        "",
        f"Episodes improved (RMSE) : {n_eps_improved}/{len(ep_ids)} "
        f"({100*n_eps_improved/len(ep_ids):.0f}%)",
        f"Steps with lower error   : "
        f"{(np.abs(hyb_all-true_all) < np.abs(cc_all-true_all)).mean()*100:.1f}%",
        f"Corr(correction, CC err) : "
        f"{np.corrcoef(cc_all-true_all, corr_all)[0,1]:.4f}",
        "",
        "Per-episode RMSE stats:",
        f"  CC+REN  mean={np.mean(rmse_hyb_list):.5f}  "
        f"min={np.min(rmse_hyb_list):.5f}  max={np.max(rmse_hyb_list):.5f}",
        f"  CC      mean={np.mean(rmse_cc_list):.5f}  "
        f"min={np.min(rmse_cc_list):.5f}  max={np.max(rmse_cc_list):.5f}",
        "",
        f"VERDICT: {verdict}",
        f"RMSE improvement: {rmse_imp_pct:+.2f}%",
        f"MAE  improvement: {mae_imp_pct:+.2f}%",
    ]
    summary_path = os.path.join(OUT_DIR, "hybrid_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Saved → {summary_path}")

    # ── Aggregate plots ───────────────────────────────────────────────────
    if not no_plots:
        print(f"\n  Generating aggregate plots...")

        plot_scatter(
            true_all, hyb_all, cc_all,
            os.path.join(PLOT_DIR, "aggregate_scatter.png")
        )
        print(f"    ✓ aggregate_scatter.png")

        plot_error_distribution(
            err_hyb, err_cc,
            os.path.join(PLOT_DIR, "error_distribution.png")
        )
        print(f"    ✓ error_distribution.png")

        plot_episode_rmse_bar(
            ep_ids, rmse_cc_list, rmse_hyb_list,
            os.path.join(PLOT_DIR, "episode_rmse_bar.png")
        )
        print(f"    ✓ episode_rmse_bar.png")

        plot_soc_band_errors(
            true_all, hyb_all, cc_all,
            os.path.join(PLOT_DIR, "soc_band_errors.png")
        )
        print(f"    ✓ soc_band_errors.png")

        plot_zero_current_check(
            df, true_all, hyb_all, cc_all,
            os.path.join(PLOT_DIR, "zero_current_check.png")
        )
        print(f"    ✓ zero_current_check.png")

        plot_correction_analysis(
            df, corr_all, true_all, cc_all,
            os.path.join(PLOT_DIR, "correction_analysis.png")
        )
        print(f"    ✓ correction_analysis.png")

        print(f"\n  Episode plots: {min(len(ep_ids), max_ep_plots)} files in {PLOT_DIR}/")
        print(f"\n  Done.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid REN inference — proves CC+REN > CC"
    )
    parser.add_argument(
        "--episodes", type=int, default=999,
        help="Max number of per-episode plots to generate (default: all)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip all plot generation, print metrics only"
    )
    args = parser.parse_args()
    main(max_ep_plots=args.episodes, no_plots=args.no_plots)