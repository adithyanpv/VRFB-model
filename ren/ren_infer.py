"""
Hybrid REN Inference  v4  (Pure Observer Architecture)
======================================================
CHANGES FROM v3
---------------
  - INPUT_DIM = 6  (pure observer: removed bias_est, cumulative_ah_norm, tr)
  - No PI observer / bias integrator in run_episode
  - Feature vector: voltage, current, T_stack, T_tank, flow, soc_cc
  - final_soc = clip(soc_cc + correction, 0, 1)
  - Dual EMA retained (fast tau=30, display tau=120)
  - Correlation metric: target is -1.0 (perfect correction = -(CC_error))

USAGE
-----
  python -m ren.infer
  python -m ren.infer --episodes 6
  python -m ren.infer --no-plots

OUTPUTS
-------
  ren/hybrid_plots/episode_XXX.png
  ren/hybrid_plots/aggregate_scatter.png
  ren/hybrid_plots/error_distribution.png
  ren/hybrid_plots/soc_band_errors.png
  ren/hybrid_plots/episode_rmse_bar.png
  ren/hybrid_plots/correction_analysis.png
  ren/hybrid_metrics.csv
  ren/hybrid_summary.txt
"""

import argparse
import math
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from ren.ren_model import REN

warnings.filterwarnings("ignore")

# ── Must match train_ren.py FEATURE_COLS exactly ─────────────────────────────
FEATURE_COLS = [
    "voltage",
    "current",
    "temperature_stack",
    "temperature_tank",
    "flow_rate",
    "soc_cc",
    "v_ocv_approx",    # Ohmic-corrected Nernst voltage
]
TARGET_COL   = "SOC_true"
CURRENT_IDX  = 1
INPUT_DIM    = len(FEATURE_COLS)   # 7
HIDDEN_DIM   = 128
ALPHA        = 0.5
EMA_TAU_FAST = 30
EMA_TAU_DISP = 120

TEST_CSV    = "datasets/vrfb_test.csv"
SCALER_PATH = "ren/scaler.pkl"
MODEL_PATH  = "ren/ren_soc_best.pth"
OUT_DIR     = "ren"
PLOT_DIR    = os.path.join(OUT_DIR, "hybrid_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_TRUE = "#00e5ff"
C_CC   = "#ff8c42"
C_REN  = "#00ff9d"

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130})


# =============================================================================
# METRICS
# =============================================================================

def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    return {
        "rmse"     : math.sqrt(np.mean(err**2)),
        "mae"      : np.mean(np.abs(err)),
        "max_err"  : np.max(np.abs(err)),
        "mean_bias": np.mean(err),
        "std_err"  : np.std(err),
    }


def improvement_str(base: float, new: float) -> str:
    return f"{(1.0 - new / base) * 100:+.1f}%"


# =============================================================================
# INFERENCE — pure observer, no feedback loop
# =============================================================================

def run_episode(
    model,
    scaler,
    X_ep_raw:  np.ndarray,    # (T, 6) unscaled features
    soc_cc_ep: np.ndarray,    # (T,) raw CC SOC
    I_mean:    float,
    I_std:     float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure observer stepwise inference.

    X_ep_raw must contain exactly FEATURE_COLS = [voltage, current,
    T_stack, T_tank, flow_rate, soc_cc] in that order.

    No PI observer. No bias integrator. No cumulative features.
    The hidden state z is the sole long-horizon integrator.

    Returns
    -------
    soc_hybrid  : (T,) display EMA (tau=120) of clip(soc_cc + correction, 0, 1)
    corrections : (T,) raw REN correction output
    """
    model.eval()
    X_scaled = scaler.transform(X_ep_raw).astype(np.float32)
    T        = X_scaled.shape[0]

    with torch.no_grad():
        z = model.z0.clone()

    ema_fast   = float(soc_cc_ep[0])
    ema_disp   = float(soc_cc_ep[0])
    EMA_A_FAST = 1.0 / EMA_TAU_FAST
    EMA_A_DISP = 1.0 / EMA_TAU_DISP

    soc_hybrid  = np.zeros(T, dtype=np.float32)
    corrections = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for t in range(T):
            x_t = torch.tensor(
                X_scaled[t:t+1], dtype=torch.float32
            ).unsqueeze(0)

            # Reconstruct raw Amps for gate (scaler shifted current)
            I_raw = torch.tensor(
                [[[X_scaled[t, CURRENT_IDX] * I_std + I_mean]]],
                dtype=torch.float32,
            )

            y_t, z = model(x_t, z=z, x_raw=I_raw)
            corr   = float(y_t.squeeze())

            raw_hyb  = float(np.clip(soc_cc_ep[t] + corr, 0.0, 1.0))
            ema_fast = (1.0 - EMA_A_FAST) * ema_fast + EMA_A_FAST * raw_hyb
            ema_disp = (1.0 - EMA_A_DISP) * ema_disp + EMA_A_DISP * raw_hyb

            corrections[t] = corr
            soc_hybrid[t]  = float(np.clip(ema_disp, 0.0, 1.0))

    return soc_hybrid, corrections


# =============================================================================
# EPISODE PLOT
# =============================================================================

def plot_episode(ep_id, time_s, soc_true, soc_cc, soc_hybrid,
                 corrections, m_cc, m_hyb, save_path):
    time_h  = time_s / 3600.0
    err_cc  = soc_cc     - soc_true
    err_hyb = soc_hybrid - soc_true

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_h, soc_true,   color=C_TRUE, lw=2.0, label="Ground truth", zorder=3)
    ax1.plot(time_h, soc_cc,     color=C_CC,   lw=1.2, alpha=0.85,
             label=f"CC   RMSE={m_cc['rmse']:.4f}")
    ax1.plot(time_h, soc_hybrid, color=C_REN,  lw=1.5, alpha=0.90,
             label=f"CC+REN  RMSE={m_hyb['rmse']:.4f}")
    ax1.set_ylabel("SOC"); ax1.set_ylim(-0.03, 1.03)
    ax1.set_title(f"Episode {ep_id} — SOC Comparison  (pure observer, 6 features)",
                  fontweight="bold")
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_h, np.abs(err_cc),  color=C_CC,  lw=1.0, alpha=0.85,
             label=f"CC   MAE={m_cc['mae']:.4f}")
    ax2.plot(time_h, np.abs(err_hyb), color=C_REN, lw=1.0, alpha=0.85,
             label=f"Hybrid MAE={m_hyb['mae']:.4f}")
    ax2.set_ylabel("|Error|"); ax2.set_title("Absolute Error")
    ax2.legend(fontsize=9); ax2.set_xlabel("Time (h)")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax3.plot(time_h, err_cc,  color=C_CC,  lw=1.0, alpha=0.85,
             label=f"CC   bias={m_cc['mean_bias']:+.4f}")
    ax3.plot(time_h, err_hyb, color=C_REN, lw=1.0, alpha=0.85,
             label=f"Hybrid bias={m_hyb['mean_bias']:+.4f}")
    ax3.set_ylabel("Signed Error"); ax3.set_title("Signed Error")
    ax3.legend(fontsize=9); ax3.set_xlabel("Time (h)")

    ax4 = fig.add_subplot(gs[1, 2])
    n = np.arange(1, len(err_cc)+1)
    ax4.plot(time_h, np.cumsum(np.abs(err_cc))  / n, color=C_CC,  lw=1.2, label="CC")
    ax4.plot(time_h, np.cumsum(np.abs(err_hyb)) / n, color=C_REN, lw=1.2, label="CC+REN")
    ax4.set_xlabel("Time (h)"); ax4.set_ylabel("Running MAE")
    ax4.set_title("Running MAE"); ax4.legend(fontsize=9)

    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.plot(time_h, corrections, color="#b388ff", lw=0.8, alpha=0.85)
    ax5.axhline(0, color="white", lw=0.8, ls="--", alpha=0.4)
    ax5.fill_between(time_h, corrections, 0,
                     where=corrections > 0, color="#b388ff", alpha=0.2, label="Positive")
    ax5.fill_between(time_h, corrections, 0,
                     where=corrections < 0, color="#ff3d5a", alpha=0.2, label="Negative")
    ax5.set_xlabel("Time (h)"); ax5.set_ylabel("REN correction")
    ax5.set_title("REN Correction Signal"); ax5.legend(fontsize=9)

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    rows = []
    for lbl, mk in [("RMSE","rmse"),("MAE","mae"),("Max Err","max_err"),("Bias","mean_bias")]:
        rv = m_hyb[mk]; cv = m_cc[mk]
        imp = improvement_str(cv, rv) if lbl != "Bias" else "---"
        fc  = "#d4edda" if lbl != "Bias" and float(imp.replace('%','')) > 0 else "#f8d7da"
        if lbl == "Bias": fc = "white"
        rows.append([lbl, f"{rv:.5f}", f"{cv:.5f}", imp, fc])
    tbl = ax6.table(
        cellText  = [[r[0],r[1],r[2],r[3]] for r in rows],
        colLabels = ["Metric","CC+REN","CC","Improvement"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.8)
    for j in range(4):
        tbl[(0,j)].set_facecolor("#0d1525")
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")
    for i, r in enumerate(rows, 1):
        if r[3] != "---":
            tbl[(i,3)].set_facecolor(r[4])
    ax6.set_title("Metrics", fontweight="bold", pad=8)

    plt.suptitle(f"Episode {ep_id}  |  CC+REN vs CC  (pure observer)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# =============================================================================
# AGGREGATE PLOTS
# =============================================================================

def plot_scatter(true, pred_hyb, pred_cc, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, pred, lbl, col in [(axes[0], pred_cc, "Coulomb Counter", C_CC),
                                (axes[1], pred_hyb, "CC + REN", C_REN)]:
        m = metrics(true, pred)
        ax.scatter(true, pred, c=col, s=1, alpha=0.15, rasterized=True)
        lo, hi = true.min(), true.max()
        ax.plot([lo,hi],[lo,hi],"w--",lw=1.2)
        ax.set_xlabel("SOC True"); ax.set_ylabel("SOC Predicted")
        ax.set_title(f"{lbl}\nRMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}")
    plt.suptitle("Predicted vs True SOC  (pure observer)", fontweight="bold")
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight", dpi=120); plt.close(fig)


def plot_error_distribution(err_hyb, err_cc, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(-0.40, 0.40, 100)
    axes[0].hist(err_cc,  bins=bins, color=C_CC,  alpha=0.65, label="CC",     density=True)
    axes[0].hist(err_hyb, bins=bins, color=C_REN, alpha=0.65, label="CC+REN", density=True)
    axes[0].axvline(0, color="white", lw=1.2, ls="--")
    axes[0].set_xlabel("Signed error"); axes[0].set_ylabel("Density")
    axes[0].set_title("Error Distribution"); axes[0].legend()
    abs_cc  = np.sort(np.abs(err_cc))
    abs_hyb = np.sort(np.abs(err_hyb))
    axes[1].plot(abs_cc,  np.linspace(0,1,len(abs_cc)),  color=C_CC,  lw=2,
                 label=f"CC   MAE={abs_cc.mean():.4f}")
    axes[1].plot(abs_hyb, np.linspace(0,1,len(abs_hyb)), color=C_REN, lw=2,
                 label=f"CC+REN MAE={abs_hyb.mean():.4f}")
    axes[1].set_xlabel("|Error|"); axes[1].set_ylabel("CDF")
    axes[1].set_title("Error CDF  (left = better)"); axes[1].legend()
    plt.suptitle("Error Analysis", fontweight="bold")
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight", dpi=120); plt.close(fig)


def plot_episode_rmse_bar(ep_ids, rmse_cc, rmse_hyb, save_path):
    x = np.arange(len(ep_ids)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(ep_ids)*0.6), 5))
    ax.bar(x - w/2, rmse_cc,  w, color=C_CC,  alpha=0.85, label="CC")
    ax.bar(x + w/2, rmse_hyb, w, color=C_REN, alpha=0.85, label="CC+REN")
    ax.axhline(np.mean(rmse_cc),  color=C_CC,  ls="--", lw=1.2,
               label=f"CC mean={np.mean(rmse_cc):.4f}")
    ax.axhline(np.mean(rmse_hyb), color=C_REN, ls="--", lw=1.2,
               label=f"Hybrid mean={np.mean(rmse_hyb):.4f}")
    ax.set_xticks(x); ax.set_xticklabels([str(e) for e in ep_ids],
                                          rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Episode"); ax.set_ylabel("RMSE")
    ax.set_title("Per-Episode RMSE", fontweight="bold"); ax.legend(fontsize=9)
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight", dpi=120); plt.close(fig)


def plot_soc_band_errors(true, pred_hyb, pred_cc, save_path):
    bands = np.arange(0, 1.0, 0.1)
    hyb_b = []; cc_b = []; labels = []
    for lo in bands:
        hi   = lo + 0.1
        mask = (true >= lo) & (true < hi)
        if mask.sum() < 100: continue
        hyb_b.append(math.sqrt(np.mean((pred_hyb[mask]-true[mask])**2)))
        cc_b.append( math.sqrt(np.mean((pred_cc[mask] -true[mask])**2)))
        labels.append(f"{lo:.1f}-{hi:.1f}")
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, cc_b,  w, color=C_CC,  alpha=0.85, label="CC")
    ax.bar(x + w/2, hyb_b, w, color=C_REN, alpha=0.85, label="CC+REN")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("SOC Band"); ax.set_ylabel("RMSE")
    ax.set_title("RMSE by SOC Band", fontweight="bold"); ax.legend()
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight", dpi=120); plt.close(fig)


def plot_correction_analysis(df, corrections_all, true_all, pred_cc, save_path):
    cc_err   = pred_cc - true_all
    corr     = corrections_all
    I_vals   = df["current"].values
    soc_true = true_all

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sample = np.random.choice(len(corr), size=min(50000, len(corr)), replace=False)
    axes[0].scatter(cc_err[sample], corr[sample], s=1, alpha=0.15, c=C_REN, rasterized=True)
    lim = max(np.abs(cc_err).max(), np.abs(corr).max()) * 0.8
    axes[0].set_xlim(-lim, lim); axes[0].set_ylim(-lim, lim)
    axes[0].axhline(0, color="white", lw=0.8, ls="--", alpha=0.4)
    axes[0].axvline(0, color="white", lw=0.8, ls="--", alpha=0.4)
    axes[0].plot([-lim,lim],[lim,-lim],"w--",lw=1,alpha=0.6,label="Perfect (slope=-1)")
    r = float(np.corrcoef(cc_err[sample], corr[sample])[0,1])
    axes[0].set_xlabel("CC error (CC - true)"); axes[0].set_ylabel("REN correction")
    axes[0].set_title(f"Correction vs CC error\nPearson r={r:.3f}  (target -1.0)",
                      fontweight="bold")
    axes[0].legend(fontsize=8)

    bands  = [(0.05,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,0.95)]
    labels = ["5-20%","20-40%","40-60%","60-80%","80-95%"]
    data   = [corr[(soc_true >= lo) & (soc_true < hi)] for lo, hi in bands]
    axes[1].boxplot(data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor=C_REN, alpha=0.4),
                    medianprops=dict(color="white", lw=2))
    axes[1].axhline(0, color="white", lw=1, ls="--", alpha=0.5)
    axes[1].set_xlabel("SOC band"); axes[1].set_ylabel("Correction")
    axes[1].set_title("Correction by SOC band", fontweight="bold")

    zero = np.abs(I_vals) < 2.0
    axes[2].hist(np.abs(corr[zero]),  bins=50, color=C_CC,  alpha=0.75, density=True,
                 label=f"|I|<2A   mean={np.abs(corr[zero]).mean():.5f}")
    axes[2].hist(np.abs(corr[~zero]), bins=50, color=C_REN, alpha=0.75, density=True,
                 label=f"|I|>=2A  mean={np.abs(corr[~zero]).mean():.5f}")
    axes[2].set_xlabel("|Correction|"); axes[2].set_ylabel("Density")
    axes[2].set_title("|Correction| at rest vs active", fontweight="bold")
    axes[2].legend(fontsize=9)

    plt.suptitle("REN Correction Analysis  (Pearson r target: -1.0)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120); plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main(max_ep_plots: int = 999, no_plots: bool = False):

    print(f"\n{'='*68}")
    print(f"  Hybrid REN Inference  v4  (Pure Observer Architecture)")
    print(f"  Features ({INPUT_DIM}): {FEATURE_COLS}")
    print(f"  PI observer : REMOVED — z is the sole integrator")
    print(f"{'='*68}")

    model = REN(
        input_dim        = INPUT_DIM,
        hidden_dim       = HIDDEN_DIM,
        output_dim       = 1,
        alpha            = ALPHA,
        dropout          = 0.0,
        n_power_iters    = 10,
        use_feedthrough  = False,
        use_current_gate = True,
        current_feat_idx = CURRENT_IDX,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    I_mean       = float(scaler.mean_[CURRENT_IDX])
    I_std        = float(scaler.scale_[CURRENT_IDX])
    gate_at_zero = model.gate_value_at_raw_amps(0.0)

    print(f"\n  Model        : {MODEL_PATH}")
    print(f"  Parameters   : {model.count_parameters():,}")
    print(f"  sigma(A_bar) : {model.contraction_rate():.4f}  (< {1-ALPHA:.2f})")
    print(f"  z0 norm      : {model.z0_norm():.4f}")
    print(f"  Gate @ I=0A  : {gate_at_zero:.6f}  (must be 0.0)")
    if gate_at_zero > 0.01:
        print(f"  [WARN] Gate not closed — retrain")
    else:
        print(f"  [OK] Gate constraint enforced")

    print(f"\n  Loading {TEST_CSV} ...")
    df = pd.read_csv(TEST_CSV)

    # Validate columns
    needed  = FEATURE_COLS + ["SOC_true", "soc_cc", "episode_id", "time"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\nRe-run dataset_gen.py (v5 pure observer)."
        )

    ep_ids = sorted(df["episode_id"].unique())
    print(f"  Episodes  : {len(ep_ids)}")
    print(f"  Rows      : {len(df):,}")
    print(f"\n  Current scaler : mean={I_mean:.2f}A  std={I_std:.2f}A")

    print(f"\n  {'Ep':>4}  {'CC RMSE':>10}  {'Hyb RMSE':>10}  "
          f"{'CC MAE':>9}  {'Hyb MAE':>9}  {'dRMSE':>9}  {'OK?':>5}")
    print(f"  {'-'*70}")

    all_true = []; all_cc = []; all_hyb = []; all_corr = []
    ep_log = []; rmse_cc_list = []; rmse_hyb_list = []

    for i, ep_id in enumerate(ep_ids):
        ep_df    = df[df["episode_id"] == ep_id].reset_index(drop=True)
        time_s   = ep_df["time"].values
        soc_true = ep_df["SOC_true"].values
        soc_cc   = ep_df["soc_cc"].values
        X_ep     = ep_df[FEATURE_COLS].values.astype(np.float32)

        soc_hyb, corr = run_episode(model, scaler, X_ep, soc_cc, I_mean, I_std)

        m_cc   = metrics(soc_true, soc_cc)
        m_hyb  = metrics(soc_true, soc_hyb)
        d_rmse = m_hyb["rmse"] - m_cc["rmse"]
        ok     = "OK" if d_rmse < 0 else "X"

        print(f"  {ep_id:>4d}  {m_cc['rmse']:>10.5f}  {m_hyb['rmse']:>10.5f}  "
              f"{m_cc['mae']:>9.5f}  {m_hyb['mae']:>9.5f}  "
              f"{d_rmse:>+9.5f}  {ok}")

        all_true.append(soc_true); all_cc.append(soc_cc)
        all_hyb.append(soc_hyb);  all_corr.append(corr)
        rmse_cc_list.append(m_cc["rmse"])
        rmse_hyb_list.append(m_hyb["rmse"])

        ep_log.append({
            "episode_id"          : ep_id,
            "cc_rmse"             : m_cc["rmse"],
            "hyb_rmse"            : m_hyb["rmse"],
            "cc_mae"              : m_cc["mae"],
            "hyb_mae"             : m_hyb["mae"],
            "cc_max_err"          : m_cc["max_err"],
            "hyb_max_err"         : m_hyb["max_err"],
            "cc_bias"             : m_cc["mean_bias"],
            "hyb_bias"            : m_hyb["mean_bias"],
            "rmse_improvement_pct": (1 - m_hyb["rmse"] / m_cc["rmse"]) * 100,
            "mae_improvement_pct" : (1 - m_hyb["mae"]  / m_cc["mae"])  * 100,
            "frac_steps_improved" : float(
                (np.abs(soc_hyb - soc_true) < np.abs(soc_cc - soc_true)).mean()
            ),
            "corr_cc_err_corr": float(
                np.corrcoef(soc_cc - soc_true, corr)[0, 1]
            ),
        })

        if not no_plots and i < max_ep_plots:
            plot_episode(ep_id, time_s, soc_true, soc_cc, soc_hyb, corr,
                         m_cc, m_hyb,
                         os.path.join(PLOT_DIR, f"episode_{ep_id:03d}.png"))

    true_all = np.concatenate(all_true)
    cc_all   = np.concatenate(all_cc)
    hyb_all  = np.concatenate(all_hyb)
    corr_all = np.concatenate(all_corr)
    err_cc   = cc_all  - true_all
    err_hyb  = hyb_all - true_all

    ov_cc    = metrics(true_all, cc_all)
    ov_hyb   = metrics(true_all, hyb_all)
    n_improved = sum(1 for r in ep_log if r["rmse_improvement_pct"] > 0)
    r_overall  = float(np.corrcoef(cc_all - true_all, corr_all)[0, 1])
    rmse_imp   = (1 - ov_hyb["rmse"] / ov_cc["rmse"]) * 100
    mae_imp    = (1 - ov_hyb["mae"]  / ov_cc["mae"])  * 100

    print(f"\n{'='*68}  OVERALL RESULTS")
    print(f"  {'Metric':<22} {'CC+REN':>12}  {'CC':>12}  {'Improvement':>12}")
    print(f"  {'-'*62}")
    for key, lbl in [("rmse","RMSE"),("mae","MAE"),("max_err","Max Error"),
                     ("mean_bias","Mean Bias"),("std_err","Error Std")]:
        rv  = ov_hyb[key]; cv = ov_cc[key]
        imp = improvement_str(cv, rv) if key not in ("mean_bias","std_err") else "---"
        print(f"  {lbl:<22} {rv:>12.5f}  {cv:>12.5f}  {imp:>12}")

    print(f"\n  Episodes improved (RMSE)    : {n_improved}/{len(ep_ids)} "
          f"({100*n_improved/len(ep_ids):.0f}%)")
    print(f"  Steps with lower |error|    : "
          f"{(np.abs(hyb_all-true_all) < np.abs(cc_all-true_all)).mean()*100:.1f}%")
    print(f"  Mean |correction|           : {np.abs(corr_all).mean():.5f}")
    print(f"\n  Corr(CC_error, correction)  : {r_overall:.4f}")
    print(f"  (target: -1.0 = perfect,  0 = no info,  +1 = opposite)")
    print(f"  Gate @ I=0A (raw Amps)      : {gate_at_zero:.6f}  (must be 0.0)")

    verdict = "CC + REN IS BETTER" if rmse_imp > 0 else "CC IS STILL BETTER"
    print(f"\n  {'='*56}")
    print(f"  VERDICT  : {verdict}")
    print(f"  RMSE imp : {rmse_imp:+.2f}%")
    print(f"  MAE  imp : {mae_imp:+.2f}%")
    print(f"  {'='*56}")

    pd.DataFrame(ep_log).to_csv(
        os.path.join(OUT_DIR, "hybrid_metrics.csv"), index=False
    )

    summary = [
        "CC + REN Hybrid vs CC  (Pure Observer, 6 features)",
        "=" * 56,
        f"Features : {FEATURE_COLS}",
        f"PI observer : REMOVED",
        f"Gate @ I=0A : {gate_at_zero:.6f}  (must be 0.0)",
        f"Corr(CC_err, correction): {r_overall:.4f}  (target -1.0)",
        "",
        f"{'Metric':<22} {'CC+REN':>10}  {'CC':>10}  {'Improvement':>12}",
        "-" * 58,
    ]
    for key, lbl in [("rmse","RMSE"),("mae","MAE"),("max_err","Max Err"),("mean_bias","Bias")]:
        rv = ov_hyb[key]; cv = ov_cc[key]
        imp = improvement_str(cv, rv) if key != "mean_bias" else "---"
        summary.append(f"{lbl:<22} {rv:>10.5f}  {cv:>10.5f}  {imp:>12}")
    summary += [
        "",
        f"Episodes improved (RMSE): {n_improved}/{len(ep_ids)}",
        f"Steps improved: {(np.abs(hyb_all-true_all)<np.abs(cc_all-true_all)).mean()*100:.1f}%",
        "",
        f"VERDICT : {verdict}",
        f"RMSE improvement : {rmse_imp:+.2f}%",
        f"MAE  improvement : {mae_imp:+.2f}%",
    ]
    with open(os.path.join(OUT_DIR, "hybrid_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"\n  Saved -> ren/hybrid_metrics.csv  and  ren/hybrid_summary.txt")

    if not no_plots:
        print(f"\n  Generating aggregate plots...")
        plot_scatter(true_all, hyb_all, cc_all,
                     os.path.join(PLOT_DIR, "aggregate_scatter.png"))
        plot_error_distribution(err_hyb, err_cc,
                     os.path.join(PLOT_DIR, "error_distribution.png"))
        plot_episode_rmse_bar(ep_ids, rmse_cc_list, rmse_hyb_list,
                     os.path.join(PLOT_DIR, "episode_rmse_bar.png"))
        plot_soc_band_errors(true_all, hyb_all, cc_all,
                     os.path.join(PLOT_DIR, "soc_band_errors.png"))
        plot_correction_analysis(df, corr_all, true_all, cc_all,
                     os.path.join(PLOT_DIR, "correction_analysis.png"))
        print(f"    OK 5 aggregate plots")
        print(f"    OK {min(len(ep_ids), max_ep_plots)} episode plots in {PLOT_DIR}/")
    print(f"\n  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=999)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    main(max_ep_plots=args.episodes, no_plots=args.no_plots)