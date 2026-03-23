# ren/infer_ren.py
"""
REN Inference & Comparison Script  (v3 — 9 features)
===================================
Loads the trained REN model and runs it episode-by-episode on the test set.
Produces a full comparison of:

  SOC_true  — physics ground truth from the digital twin
  soc_cc    — Coulomb counter (with injected initial error + sensor bias)
  soc_ren   — REN prediction from 8 observable signals

OUTPUTS:
  ren/comparison_metrics.csv       — per-episode error table (REN vs CC)
  ren/comparison_summary.txt       — overall metrics summary
  ren/plots/episode_XX.png         — time-series plot per episode
  ren/plots/scatter_ren.png        — REN predicted vs true scatter
  ren/plots/scatter_cc.png         — CC predicted vs true scatter
  ren/plots/error_distribution.png — error histogram REN vs CC
  ren/plots/metrics_by_episode.png — RMSE per episode bar chart
  ren/plots/soc_band_errors.png    — RMSE broken down by SOC band

USAGE:
  python -m ren.infer_ren
  python -m ren.infer_ren --episodes 4   # plot only first 4 episodes
"""

import os
import argparse
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from ren.ren_model import REN

# ── Config — must match train_ren.py exactly ─────────────────────────────────
FEATURE_COLS = [
    "voltage", "current", "temperature_stack", "temperature_tank",
    "flow_rate",
]
TARGET_COL  = "SOC_true"
INPUT_DIM   = 5
HIDDEN_DIM  = 128
ALPHA       = 0.5    # match train_ren.py v3
DROPOUT     = 0.0     # disable dropout at inference

TEST_CSV    = "datasets/vrfb_test.csv"
SCALER_PATH = "ren/scaler.pkl"
MODEL_PATH  = "ren/ren_soc_best.pth"
OUT_DIR     = "ren"
PLOT_DIR    = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plot style
plt.rcParams.update({
    "font.family":  "Arial",
    "font.size":    11,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "figure.dpi":   130,
})

COLOUR_TRUE = "#2E4057"   # dark blue  — ground truth
COLOUR_CC   = "#E07B39"   # orange     — Coulomb counter
COLOUR_REN  = "#2ECC71"   # green      — REN


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    return {
        "rmse":     math.sqrt(np.mean(err**2)),
        "mae":      np.mean(np.abs(err)),
        "max_err":  np.max(np.abs(err)),
        "mean_err": np.mean(err),   # signed — reveals systematic bias
    }


def run_ren_on_episode(model, scaler, X_ep: np.ndarray) -> np.ndarray:
    """
    Run REN on one full episode carrying hidden state z forward.
    The entire episode is passed as a single sequence so z builds
    context continuously — this is the correct deployment mode.
    """
    model.eval()
    X_scaled = scaler.transform(X_ep).astype(np.float32)
    X_t      = torch.tensor(X_scaled).unsqueeze(0).to(DEVICE)  # [1, T, 8]

    with torch.no_grad():
        # z=None -> use model's learned z0 initial state
        y_seq, _ = model(X_t, z=None)   # [1, T, 1]

    return y_seq.squeeze().cpu().numpy()   # [T]


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE TIME-SERIES PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_episode(ep_id, time_s, soc_true, soc_cc, soc_ren,
                 ren_m, cc_m, save_path):
    """5-panel plot: SOC traces, absolute error, signed error,
       running MAE, and metrics table."""
    time_h  = time_s / 3600.0
    err_ren = soc_ren - soc_true
    err_cc  = soc_cc  - soc_true

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    # Panel 1 — SOC traces (full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_h, soc_true, color=COLOUR_TRUE, lw=2.0,
             label="Ground Truth  (SOC_true)", zorder=3)
    ax1.plot(time_h, soc_cc,   color=COLOUR_CC,   lw=1.2, alpha=0.85,
             label=f"Coulomb Counter   RMSE={cc_m['rmse']:.4f}", zorder=2)
    ax1.plot(time_h, soc_ren,  color=COLOUR_REN,  lw=1.2, alpha=0.85,
             label=f"REN Estimate      RMSE={ren_m['rmse']:.4f}", zorder=2)
    ax1.set_ylabel("State of Charge (SOC)")
    ax1.set_title(f"Episode {ep_id}  —  SOC Comparison",
                  fontweight="bold", fontsize=13)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(-0.02, 1.02)

    # Panel 2 — Absolute error
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_h, np.abs(err_cc),  color=COLOUR_CC,  lw=1.0, alpha=0.85,
             label=f"CC   MAE={cc_m['mae']:.4f}")
    ax2.plot(time_h, np.abs(err_ren), color=COLOUR_REN, lw=1.0, alpha=0.85,
             label=f"REN  MAE={ren_m['mae']:.4f}")
    ax2.set_ylabel("|Error|  (SOC units)")
    ax2.set_title("Absolute Error Over Time")
    ax2.legend(fontsize=9)

    # Panel 3 — Signed error
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.plot(time_h, err_cc,  color=COLOUR_CC,  lw=1.0, alpha=0.85,
             label=f"CC   bias={cc_m['mean_err']:+.4f}")
    ax3.plot(time_h, err_ren, color=COLOUR_REN, lw=1.0, alpha=0.85,
             label=f"REN  bias={ren_m['mean_err']:+.4f}")
    ax3.set_ylabel("Signed Error  (pred − true)")
    ax3.set_title("Signed Error  (reveals systematic bias)")
    ax3.legend(fontsize=9)

    # Panel 4 — Running MAE
    ax4 = fig.add_subplot(gs[2, 0])
    n = np.arange(1, len(err_cc) + 1)
    ax4.plot(time_h, np.cumsum(np.abs(err_cc))  / n,
             color=COLOUR_CC,  lw=1.2, label="CC  running MAE")
    ax4.plot(time_h, np.cumsum(np.abs(err_ren)) / n,
             color=COLOUR_REN, lw=1.2, label="REN running MAE")
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Running MAE")
    ax4.set_title("Running MAE  (does error grow over time?)")
    ax4.legend(fontsize=9)

    # Panel 5 — Metrics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    imp_rmse = (1 - ren_m["rmse"]    / cc_m["rmse"])    * 100
    imp_mae  = (1 - ren_m["mae"]     / cc_m["mae"])     * 100
    imp_max  = (1 - ren_m["max_err"] / cc_m["max_err"]) * 100
    rows = [
        ["RMSE",      f"{ren_m['rmse']:.5f}",    f"{cc_m['rmse']:.5f}",    f"{imp_rmse:+.1f}%"],
        ["MAE",       f"{ren_m['mae']:.5f}",     f"{cc_m['mae']:.5f}",     f"{imp_mae:+.1f}%"],
        ["Max Error", f"{ren_m['max_err']:.5f}", f"{cc_m['max_err']:.5f}", f"{imp_max:+.1f}%"],
        ["Mean Bias", f"{ren_m['mean_err']:+.5f}", f"{cc_m['mean_err']:+.5f}", "—"],
    ]
    tbl = ax5.table(cellText=rows,
                    colLabels=["Metric", "REN", "CC", "Improvement"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.7)
    for j in range(4):
        tbl[(0, j)].set_facecolor("#2E4057")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(rows, start=1):
        val = row[3]
        if val != "—":
            c = "#d4edda" if float(val.replace("%","")) > 0 else "#f8d7da"
            tbl[(i, 3)].set_facecolor(c)
    ax5.set_title("Episode Metrics", fontweight="bold", pad=8)

    for ax in [ax2, ax3]:
        ax.set_xlabel("Time (hours)")

    plt.suptitle(f"VRFB SOC Estimation — Episode {ep_id}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATE PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scatter(true_all, pred_all, label, colour, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    m = metrics(true_all, pred_all)
    ax.scatter(true_all, pred_all, alpha=0.04, s=1,
               color=colour, rasterized=True)
    lo, hi = true_all.min(), true_all.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect prediction")
    ax.set_xlabel("SOC True")
    ax.set_ylabel(f"SOC {label}")
    ax.set_title(f"{label} vs Ground Truth\n"
                 f"RMSE={m['rmse']:.5f}   MAE={m['mae']:.5f}   "
                 f"MaxErr={m['max_err']:.5f}")
    ax.legend(fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_error_distribution(err_ren, err_cc, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, err, colour, label in [
        (axes[0], err_cc,  COLOUR_CC,  "Coulomb Counter"),
        (axes[1], err_ren, COLOUR_REN, "REN"),
    ]:
        ax.hist(err, bins=120, color=colour, alpha=0.75,
                edgecolor="none", density=True)
        ax.axvline(0,             color="black", lw=1.2, ls="--",
                   label="Zero error")
        ax.axvline(np.mean(err),  color="red",   lw=1.2,
                   label=f"Mean bias = {np.mean(err):+.4f}")
        ax.axvline( np.std(err),  color="grey",  lw=0.8, ls=":")
        ax.axvline(-np.std(err),  color="grey",  lw=0.8, ls=":",
                   label=f"±1σ = {np.std(err):.4f}")
        ax.set_xlabel("Signed Error (pred − true)")
        ax.set_ylabel("Density")
        rmse = math.sqrt(np.mean(err**2))
        ax.set_title(f"{label}\nRMSE={rmse:.5f}   MAE={np.mean(np.abs(err)):.5f}")
        ax.legend(fontsize=8)
    plt.suptitle("Error Distribution: CC vs REN",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_by_episode(ep_ids, ren_rmses, cc_rmses, path):
    x = np.arange(len(ep_ids))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(ep_ids) * 0.7), 5))
    ax.bar(x - w/2, cc_rmses,  w, color=COLOUR_CC,  alpha=0.85, label="CC RMSE")
    ax.bar(x + w/2, ren_rmses, w, color=COLOUR_REN, alpha=0.85, label="REN RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ep {e}" for e in ep_ids],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE per Test Episode — CC vs REN", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_soc_band_errors(true_all, pred_ren, pred_cc, path):
    """RMSE broken down into 0.1-wide SOC bands."""
    bands = np.arange(0, 1.0, 0.1)
    ren_b, cc_b, labels = [], [], []
    for lo in bands:
        hi   = lo + 0.1
        mask = (true_all >= lo) & (true_all < hi)
        if mask.sum() < 50:
            continue
        ren_b.append(math.sqrt(np.mean((pred_ren[mask] - true_all[mask])**2)))
        cc_b.append( math.sqrt(np.mean((pred_cc[mask]  - true_all[mask])**2)))
        labels.append(f"{lo:.1f}–{hi:.1f}")

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, cc_b,  w, color=COLOUR_CC,  alpha=0.85, label="CC RMSE")
    ax.bar(x + w/2, ren_b, w, color=COLOUR_REN, alpha=0.85, label="REN RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("SOC Band")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by SOC Band — CC vs REN\n"
                 "(shows where each estimator struggles most)",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(max_episode_plots: int = 999):

    print("\n" + "=" * 65)
    print("  REN Inference — VRFB SOC Comparison")
    print("  Ground Truth  vs  Coulomb Counter  vs  REN")
    print("=" * 65)

    # Load model
    model = REN(
        input_dim     = INPUT_DIM,
        hidden_dim    = HIDDEN_DIM,
        output_dim    = 1,
        alpha         = 0.5,       # match train_ren.py v3 default
        dropout       = DROPOUT,
        n_power_iters = 10,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"  Model    : {MODEL_PATH}")
    print(f"  σ(A_bar) : {model.contraction_rate():.4f}  (contractivity)")

    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"  Scaler   : {SCALER_PATH}")

    # Load test data
    print(f"\n  Loading {TEST_CSV} ...")
    df     = pd.read_csv(TEST_CSV)
    ep_ids = sorted(df["episode_id"].unique())
    print(f"  Episodes : {len(ep_ids)}    Rows : {len(df):,}")

    # Per-episode inference
    print(f"\n  {'Ep':>4}  {'CC RMSE':>10}  {'REN RMSE':>10}  "
          f"{'CC MAE':>9}  {'REN MAE':>9}  {'Improv':>8}")
    print("  " + "-" * 60)

    all_true, all_cc, all_ren = [], [], []
    ep_log = []
    ren_rmses, cc_rmses = [], []

    for i, ep_id in enumerate(ep_ids):
        ep_df    = df[df["episode_id"] == ep_id].reset_index(drop=True)
        time_s   = ep_df["time"].values
        soc_true = ep_df[TARGET_COL].values
        soc_cc   = ep_df["soc_cc"].values
        X_ep     = ep_df[FEATURE_COLS].values.astype(np.float32)

        soc_ren  = run_ren_on_episode(model, scaler, X_ep)

        ren_m  = metrics(soc_true, soc_ren)
        cc_m   = metrics(soc_true, soc_cc)
        improv = (1 - ren_m["rmse"] / cc_m["rmse"]) * 100

        print(f"  {ep_id:>4d}  {cc_m['rmse']:>10.5f}  {ren_m['rmse']:>10.5f}  "
              f"{cc_m['mae']:>9.5f}  {ren_m['mae']:>9.5f}  {improv:>+7.1f}%")

        all_true.append(soc_true)
        all_cc.append(soc_cc)
        all_ren.append(soc_ren)
        ren_rmses.append(ren_m["rmse"])
        cc_rmses.append(cc_m["rmse"])

        ep_log.append({
            "episode_id":           ep_id,
            "cc_rmse":              cc_m["rmse"],
            "ren_rmse":             ren_m["rmse"],
            "cc_mae":               cc_m["mae"],
            "ren_mae":              ren_m["mae"],
            "cc_max_err":           cc_m["max_err"],
            "ren_max_err":          ren_m["max_err"],
            "cc_mean_bias":         cc_m["mean_err"],
            "ren_mean_bias":        ren_m["mean_err"],
            "rmse_improvement_pct": improv,
        })

        # Per-episode time-series plot
        if i < max_episode_plots:
            plot_episode(
                ep_id, time_s, soc_true, soc_cc, soc_ren, ren_m, cc_m,
                os.path.join(PLOT_DIR, f"episode_{ep_id:02d}.png")
            )

    # Aggregate
    true_all = np.concatenate(all_true)
    cc_all   = np.concatenate(all_cc)
    ren_all  = np.concatenate(all_ren)
    err_ren  = ren_all - true_all
    err_cc   = cc_all  - true_all
    ov_ren   = metrics(true_all, ren_all)
    ov_cc    = metrics(true_all, cc_all)

    # Overall table
    print(f"\n{'=' * 65}")
    print(f"  OVERALL TEST SET RESULTS")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<18} {'REN':>12}  {'CC':>12}  {'Improvement':>12}")
    print(f"  {'-' * 58}")
    for key, label in [("rmse","RMSE"), ("mae","MAE"),
                       ("max_err","Max Error"), ("mean_err","Mean Bias")]:
        rv = ov_ren[key]; cv = ov_cc[key]
        imp = f"{(1-rv/cv)*100:>+.1f}%" if key != "mean_err" else "—"
        print(f"  {label:<18} {rv:>12.5f}  {cv:>12.5f}  {imp:>12}")
    print(f"{'=' * 65}")

    # Save CSV + summary
    pd.DataFrame(ep_log).to_csv(
        os.path.join(OUT_DIR, "comparison_metrics.csv"), index=False)

    summary = [
        "REN vs CC vs Ground Truth — Test Set Summary",
        "=" * 50,
        f"Test episodes : {len(ep_ids)}",
        f"Total rows    : {len(true_all):,}",
        "",
        f"{'Metric':<18} {'REN':>10}  {'CC':>10}  {'Improvement':>12}",
        "-" * 54,
    ]
    for key, label in [("rmse","RMSE"), ("mae","MAE"),
                       ("max_err","Max Error"), ("mean_err","Mean Bias")]:
        rv = ov_ren[key]; cv = ov_cc[key]
        imp = f"{(1-rv/cv)*100:+.1f}%" if key != "mean_err" else "—"
        summary.append(f"{label:<18} {rv:>10.5f}  {cv:>10.5f}  {imp:>12}")
    summary += [
        "",
        "Per-episode RMSE:",
        f"  REN  mean={np.mean(ren_rmses):.5f}  "
        f"min={np.min(ren_rmses):.5f}  max={np.max(ren_rmses):.5f}",
        f"  CC   mean={np.mean(cc_rmses):.5f}  "
        f"min={np.min(cc_rmses):.5f}  max={np.max(cc_rmses):.5f}",
    ]
    with open(os.path.join(OUT_DIR, "comparison_summary.txt"), "w") as f:
        f.write("\n".join(summary))

    # Aggregate plots
    print(f"\n  Saving aggregate plots...")
    plot_scatter(true_all, ren_all, "REN", COLOUR_REN,
                 os.path.join(PLOT_DIR, "scatter_ren.png"))
    plot_scatter(true_all, cc_all, "Coulomb Counter", COLOUR_CC,
                 os.path.join(PLOT_DIR, "scatter_cc.png"))
    plot_error_distribution(err_ren, err_cc,
                 os.path.join(PLOT_DIR, "error_distribution.png"))
    plot_metrics_by_episode(ep_ids, ren_rmses, cc_rmses,
                 os.path.join(PLOT_DIR, "metrics_by_episode.png"))
    plot_soc_band_errors(true_all, ren_all, cc_all,
                 os.path.join(PLOT_DIR, "soc_band_errors.png"))

    print(f"\n  Saved:")
    print(f"    ren/comparison_metrics.csv")
    print(f"    ren/comparison_summary.txt")
    print(f"    ren/plots/episode_XX.png  ({len(ep_ids)} files)")
    print(f"    ren/plots/scatter_ren.png  /  scatter_cc.png")
    print(f"    ren/plots/error_distribution.png")
    print(f"    ren/plots/metrics_by_episode.png")
    print(f"    ren/plots/soc_band_errors.png")
    print(f"\n  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=999,
                        help="Max episode plots to generate (default: all)")
    args = parser.parse_args()
    main(max_episode_plots=args.episodes)