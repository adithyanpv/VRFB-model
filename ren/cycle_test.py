"""
Continuous Charge/Discharge Cycle Test  v4  (Pure Observer Architecture)
========================================================================
CHANGES FROM v3
---------------
  - input_dim = 6 (pure observer: no bias_est, no cumulative_ah_norm, no tr)
  - PI observer completely removed from simulation loop
  - Feature vector: voltage, current, T_stack, T_tank, flow, soc_cc
  - final_soc = clip(soc_cc + ren_correction, 0, 1)
  - Dual EMA retained (fast tau=30, display tau=120)
  - This eliminates the runaway feedback that caused Cycle 4 explosion

USAGE
-----
  python -m ren.cycle_test                             # 5 cycles, 150A
  python -m ren.cycle_test --cycles 8 --current 200
  python -m ren.cycle_test --soc-low 0.10 --soc-high 0.90
  python -m ren.cycle_test --cc-bias 5.0
  python -m ren.cycle_test --no-plots

OUTPUTS
-------
  ren/cycle_test/cycle_test_main.png
  ren/cycle_test/cycle_details.png
  ren/cycle_test/cycle_metrics.csv
  ren/cycle_test/cycle_summary.txt
"""

import argparse
import math
import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from vrfb.config          import VRFBConfig
from vrfb.vrfb_core       import VRFB
from vrfb.bms_controller  import BMSController
from vrfb.sensor_model    import SensorModel
from vrfb.coulomb_counter import CoulombCounter
from ren.ren_model        import REN

SCALER_PATH  = "ren/scaler.pkl"
MODEL_PATH   = "ren/ren_soc_best.pth"
OUT_DIR      = "ren/cycle_test"
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN_DIM   = 128
ALPHA        = 0.5
EMA_TAU_FAST = 30
EMA_TAU_DISP = 120
CURRENT_IDX  = 1

# 6 strictly observable features — must match train_ren.py exactly
_R_STACK_NOMINAL = (0.0015 + 0.0005) * 40   # 0.08 Ω — from config.py

FEATURE_COLS = [
    "voltage", "current", "temperature_stack",
    "temperature_tank", "flow_rate", "soc_cc", "v_ocv_approx",
]
DEVICE = torch.device("cpu")
C_TRUE = "#00e5ff"
C_CC   = "#ff8c42"
C_REN  = "#00ff9d"


# =============================================================================
# METRICS
# =============================================================================

def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    return {
        "rmse"     : math.sqrt(np.mean(err**2)) if len(err) > 0 else 0.0,
        "mae"      : np.mean(np.abs(err))        if len(err) > 0 else 0.0,
        "max_err"  : np.max(np.abs(err))         if len(err) > 0 else 0.0,
        "mean_bias": np.mean(err)                if len(err) > 0 else 0.0,
    }


# =============================================================================
# CYCLE PROFILE
# =============================================================================

def compute_halfcycle_steps(Q_nominal_Ah, I_A, soc_swing, dt, margin_s=600.0):
    return int(math.ceil((soc_swing * Q_nominal_Ah / I_A * 3600.0 + margin_s) / dt))


def build_profile(n_cycles, steps_per_half, rest_steps,
                  I_discharge, I_charge, Q_m3s, T_base=298.15):
    rng   = np.random.default_rng(42)
    total = n_cycles * 2 * steps_per_half + (n_cycles * 2 + 1) * rest_steps + 100

    I_arr = np.zeros(total, dtype=np.float32)
    Q_arr = np.full(total, Q_m3s, dtype=np.float32)
    T_arr = np.clip(T_base + np.cumsum(np.concatenate([[0.0],
                    rng.uniform(-0.001, 0.001, size=total-1)])),
                    293.0, 315.0).astype(np.float32)

    phase_log: list[dict] = []
    pos = rest_steps

    for c in range(n_cycles):
        d_s = pos; d_e = min(pos + steps_per_half, total)
        I_arr[d_s:d_e] = I_discharge
        phase_log.append({"cycle": c, "phase": "discharge", "start": d_s, "end": d_e})
        pos = d_e + rest_steps

        c_s = pos; c_e = min(pos + steps_per_half, total)
        I_arr[c_s:c_e] = -I_charge
        phase_log.append({"cycle": c, "phase": "charge", "start": c_s, "end": c_e})
        pos = c_e + rest_steps
        if pos >= total:
            break

    return I_arr, Q_arr, T_arr, phase_log, pos


# =============================================================================
# SIMULATION
# =============================================================================

def run_cycle_test(args):
    cfg          = VRFBConfig()
    dt           = cfg.dt_default
    Q_nominal_Ah = cfg.n * cfg.F * cfg.C_total * cfg.V_tank / 3600.0
    soc_swing    = args.soc_high - args.soc_low
    Q_m3s        = args.flow * cfg.LPM_to_m3s

    steps_per_half = compute_halfcycle_steps(Q_nominal_Ah, args.current, soc_swing, dt)
    rest_steps     = int(args.rest / dt)
    drift_per_half = args.cc_bias * steps_per_half * dt / (Q_nominal_Ah * 3600.0)

    print(f"\n  Q_nominal      : {Q_nominal_Ah:.0f} Ah")
    print(f"  SOC window     : {args.soc_low:.2f} -> {args.soc_high:.2f}  (swing={soc_swing:.2f})")
    print(f"  Half-cycle     : {steps_per_half:,} steps  ({steps_per_half*dt/3600:.1f} h)")
    print(f"  Expected drift : {drift_per_half:.4f} SOC/half-cycle")
    print(f"  PI observer    : REMOVED — pure observer inference")

    # Load model — 6 features
    model = REN(
        input_dim        = 7,
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

    print(f"\n  Gate @ I=0A  : {model.gate_value_at_raw_amps(0.0):.6f}  (must be 0.0)")
    print(f"  Gate @ {args.current:.0f}A : {model.gate_value_at_raw_amps(args.current):.4f}")

    # Physics objects
    cfg.initial_soc = args.soc_high
    battery = VRFB(cfg)
    bms     = BMSController(cfg)
    sensor  = SensorModel(cfg)
    cc      = CoulombCounter(cfg)

    cc_init = float(np.clip(args.soc_high + args.cc_init_err, 0.06, 0.94))
    cc.initialize(cc_init)
    cc.set_current_bias(args.cc_bias)
    sensor.set_current_bias(args.cc_bias)

    # REN state — no PI observer state needed
    with torch.no_grad():
        z_ren = model.z0.detach().clone()
    ema_fast   = float(args.soc_high)
    ema_disp   = float(args.soc_high)
    EMA_A_FAST = 1.0 / EMA_TAU_FAST
    EMA_A_DISP = 1.0 / EMA_TAU_DISP

    # Build profiles
    I_arr, Q_arr, T_arr, phase_log, total_steps = build_profile(
        args.cycles, steps_per_half, rest_steps,
        args.current, args.current, Q_m3s,
    )
    step_phase = {}
    step_cycle = {}
    for entry in phase_log:
        for s in range(entry["start"], min(entry["end"], total_steps)):
            step_phase[s] = entry["phase"]
            step_cycle[s] = entry["cycle"]

    print(f"\n  Running {total_steps:,} steps ({total_steps*dt/3600:.1f} sim-hours) ...",
          flush=True)
    t_wall = time.time()

    rows = []
    for step in range(total_steps):
        I_cmd = float(I_arr[step])
        Q_cmd = float(Q_arr[step])
        T_amb = float(T_arr[step])

        out = battery.get_outputs()
        bms.update_mode(I_cmd, out, dt)
        flow_override = bms.get_flow_override()
        Q_eff  = flow_override if flow_override is not None else Q_cmd
        I_safe = bms.apply_protection(I_cmd, out)

        battery.step(I_safe, Q_eff, T_amb, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        soc_cc   = float(cc.update(measured["current"], dt, out["capacity_nominal"]))

        # Pure observer: 6-feature vector, no PI observer
        v_ocv = measured["voltage"] - measured["current"] * _R_STACK_NOMINAL
        x = np.array([[
            measured["voltage"],
            measured["current"],
            measured["temperature"],
            measured["temperature_tank"],
            measured["flow_rate"],
            soc_cc,
            float(v_ocv),
        ]], dtype=np.float32)

        x_s   = scaler.transform(x).astype(np.float32)
        x_t   = torch.tensor(x_s).unsqueeze(0)
        I_raw = torch.tensor([[[measured["current"]]]], dtype=torch.float32)

        with torch.no_grad():
            y_t, z_ren = model(x_t, z=z_ren, x_raw=I_raw)

        corr    = float(y_t.squeeze())
        raw_hyb = float(np.clip(soc_cc + corr, 0.0, 1.0))

        # Dual EMA
        ema_fast = (1.0 - EMA_A_FAST) * ema_fast + EMA_A_FAST * raw_hyb
        ema_disp = (1.0 - EMA_A_DISP) * ema_disp + EMA_A_DISP * raw_hyb
        soc_ren  = float(np.clip(ema_disp, 0.0, 1.0))

        rows.append({
            "step"       : step,
            "time_h"     : step * dt / 3600.0,
            "cycle"      : step_cycle.get(step, args.cycles - 1),
            "phase"      : step_phase.get(step, "rest"),
            "bms_mode"   : bms.mode_name,
            "I_cmd"      : I_cmd,
            "I_safe"     : I_safe,
            "soc_true"   : out["soc_true"],
            "soc_cc"     : soc_cc,
            "soc_ren"    : soc_ren,
            "correction" : corr,
            "err_cc"     : soc_cc  - out["soc_true"],
            "abs_err_cc" : abs(soc_cc  - out["soc_true"]),
            "abs_err_ren": abs(soc_ren - out["soc_true"]),
            "voltage"    : measured["voltage"],
            "current"    : measured["current"],
        })

    elapsed = time.time() - t_wall
    print(f"  Done. ({elapsed:.1f}s,  {total_steps/elapsed:.0f} steps/sec)")

    df = pd.DataFrame(rows)

    # Per-cycle metrics
    cyc_records = []
    for c_idx in range(args.cycles):
        grp    = df[df["cycle"] == c_idx]
        active = grp[grp["phase"] != "rest"]
        if len(active) == 0:
            active = grp
        m_cc  = metrics(active["soc_true"].values, active["soc_cc"].values)
        m_ren = metrics(active["soc_true"].values, active["soc_ren"].values)
        cyc_records.append({
            "cycle"           : c_idx,
            "cc_rmse"         : m_cc["rmse"],
            "ren_rmse"        : m_ren["rmse"],
            "cc_mae"          : m_cc["mae"],
            "ren_mae"         : m_ren["mae"],
            "cc_max_err"      : m_cc["max_err"],
            "ren_max_err"     : m_ren["max_err"],
            "cc_bias"         : m_cc["mean_bias"],
            "ren_bias"        : m_ren["mean_bias"],
            "rmse_improvement": (1 - m_ren["rmse"] / m_cc["rmse"]) * 100
                                 if m_cc["rmse"] > 1e-6 else 0.0,
            "end_cc_err"      : grp.iloc[-1]["err_cc"],
            "end_ren_err"     : grp.iloc[-1]["abs_err_ren"],
        })

    cyc_df = pd.DataFrame(cyc_records)
    ov_cc  = metrics(df["soc_true"].values, df["soc_cc"].values)
    ov_ren = metrics(df["soc_true"].values, df["soc_ren"].values)
    r      = float(np.corrcoef(df["err_cc"].values, df["correction"].values)[0, 1])
    cc_trend  = np.polyfit(cyc_df["cycle"].values, cyc_df["cc_rmse"].values,  1)[0]
    ren_trend = np.polyfit(cyc_df["cycle"].values, cyc_df["ren_rmse"].values, 1)[0]

    info = {
        "ov_cc": ov_cc, "ov_ren": ov_ren, "r": r,
        "cc_trend": cc_trend, "ren_trend": ren_trend,
        "Q_nominal_Ah": Q_nominal_Ah,
        "drift_per_half": drift_per_half,
        "gate_at_zero": model.gate_value_at_raw_amps(0.0),
    }
    return df, cyc_df, info


# =============================================================================
# PLOTS
# =============================================================================

def plot_main(df, cyc_df, info, args, save_path):
    n   = len(cyc_df)
    th  = df["time_h"].values
    fig = plt.figure(figsize=(20, 16))
    gs  = plt.GridSpec(3, 2, figure=fig, hspace=0.44, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(th, df["soc_true"], color=C_TRUE, lw=2.0, label="Ground truth", zorder=4)
    ax1.plot(th, df["soc_cc"],   color=C_CC,   lw=1.0, alpha=0.75, label="CC", zorder=3)
    ax1.plot(th, df["soc_ren"],  color=C_REN,  lw=1.4, alpha=0.90,
             label="CC+REN (pure observer)", zorder=3)
    ax1.set_ylabel("SOC"); ax1.set_xlabel("Time (hours)")
    ax1.set_ylim(-0.04, 1.08)
    ax1.set_title(
        f"Continuous Cycles — Pure Observer Architecture\n"
        f"CC bias={args.cc_bias:.1f}A  I={args.current:.0f}A  "
        f"SOC {args.soc_low:.2f}->{args.soc_high:.2f}",
        fontweight="bold", fontsize=12)
    ax1.legend(loc="lower right", fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(th, df["abs_err_cc"],  alpha=0.25, color=C_CC)
    ax2.fill_between(th, df["abs_err_ren"], alpha=0.25, color=C_REN)
    ax2.plot(th, df["abs_err_cc"],  color=C_CC,  lw=0.9, alpha=0.85, label="CC |error|")
    ax2.plot(th, df["abs_err_ren"], color=C_REN, lw=0.9, alpha=0.85, label="CC+REN |error|")
    ax2.set_ylabel("|Error|"); ax2.set_xlabel("Time (hours)")
    ax2.set_title("Absolute Error (no runaway = no feedback loop)", fontweight="bold")
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(n); w = 0.38
    ax3.bar(x - w/2, cyc_df["cc_rmse"],  w, color=C_CC,  alpha=0.85, label="CC")
    ax3.bar(x + w/2, cyc_df["ren_rmse"], w, color=C_REN, alpha=0.85, label="CC+REN")
    ax3.set_xticks(x); ax3.set_xticklabels([f"C{i+1}" for i in range(n)])
    ax3.set_xlabel("Cycle"); ax3.set_ylabel("RMSE")
    ax3.set_title("RMSE per Cycle", fontweight="bold"); ax3.legend(fontsize=9)
    for i, row in cyc_df.iterrows():
        imp = row["rmse_improvement"]
        ax3.text(i, max(row["cc_rmse"], row["ren_rmse"]) + 0.001,
                 f"{imp:+.0f}%", ha="center", fontsize=7,
                 color="#00ff9d" if imp > 0 else "#ff3d5a", fontweight="bold")

    ax4 = fig.add_subplot(gs[2, 0])
    cx = cyc_df["cycle"].values
    ax4.plot(cx+1, cyc_df["cc_rmse"],  "o-", color=C_CC,  lw=2, ms=7, label="CC RMSE")
    ax4.plot(cx+1, cyc_df["ren_rmse"], "s-", color=C_REN, lw=2, ms=7, label="CC+REN RMSE")
    if len(cx) >= 3:
        z_cc  = np.polyfit(cx, cyc_df["cc_rmse"].values,  1)
        z_ren = np.polyfit(cx, cyc_df["ren_rmse"].values, 1)
        ax4.plot(cx+1, np.poly1d(z_cc)(cx),  "--", color=C_CC,  lw=1.3, alpha=0.6,
                 label=f"CC trend ({z_cc[0]:+.5f}/cycle)")
        ax4.plot(cx+1, np.poly1d(z_ren)(cx), "--", color=C_REN, lw=1.3, alpha=0.6,
                 label=f"REN trend ({z_ren[0]:+.5f}/cycle)")
    ax4.set_xlabel("Cycle"); ax4.set_ylabel("RMSE")
    ax4.set_title("Error Trend — key proof\nCC slope > 0 (drift); REN slope bounded",
                  fontweight="bold")
    ax4.legend(fontsize=8); ax4.set_xticks(cx+1)

    ax5 = fig.add_subplot(gs[2, 1])
    signed_cc  = df["err_cc"].values
    signed_ren = df["soc_ren"].values - df["soc_true"].values
    window = max(int(len(df)/200), 1)
    cc_sm  = pd.Series(signed_cc).rolling(window,  min_periods=1).mean()
    ren_sm = pd.Series(signed_ren).rolling(window, min_periods=1).mean()
    ax5.plot(th, cc_sm,  color=C_CC,  lw=1.4, alpha=0.9, label="CC signed error")
    ax5.plot(th, ren_sm, color=C_REN, lw=1.4, alpha=0.9, label="CC+REN signed error")
    ax5.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax5.fill_between(th, cc_sm,  0, color=C_CC,  alpha=0.2)
    ax5.fill_between(th, ren_sm, 0, color=C_REN, alpha=0.2)
    ax5.set_xlabel("Time (hours)"); ax5.set_ylabel("Signed Error (pred - true)")
    ax5.set_title("Systematic Bias Accumulation", fontweight="bold")
    ax5.legend(fontsize=9)

    ov  = info["ov_cc"]; ovr = info["ov_ren"]
    n_improved = (cyc_df["ren_rmse"] < cyc_df["cc_rmse"]).sum()
    box = (
        f"OVERALL ({n} cycles)\n"
        f"CC     RMSE={ov['rmse']:.4f}  MAE={ov['mae']:.4f}  bias={ov['mean_bias']:+.4f}\n"
        f"CC+REN RMSE={ovr['rmse']:.4f}  MAE={ovr['mae']:.4f}  bias={ovr['mean_bias']:+.4f}\n"
        f"RMSE imp: {(1-ovr['rmse']/ov['rmse'])*100:+.1f}%  |  "
        f"Cycles improved: {n_improved}/{n}  |  Corr: {info['r']:.3f}"
    )
    fig.text(0.5, 0.01, box, ha="center", va="bottom", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1525",
                       edgecolor="#2a4060", alpha=0.95))
    plt.suptitle(
        f"VRFB Cycle Test — Pure Observer (6-feature REN)\n"
        f"{n} cycles  I={args.current:.0f}A  Q={args.flow:.0f} LPM  "
        f"CC bias={args.cc_bias:.1f}A",
        fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("    OK cycle_test_main.png")


def plot_cycle_details(df, cyc_df, save_path):
    n    = len(cyc_df)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4*n), squeeze=False)
    for c_idx, row in cyc_df.iterrows():
        grp = df[df["cycle"] == row["cycle"]]
        th  = grp["time_h"].values
        axes[c_idx, 0].plot(th, grp["soc_true"], color=C_TRUE, lw=1.8, label="True")
        axes[c_idx, 0].plot(th, grp["soc_cc"],   color=C_CC,   lw=1.0, alpha=0.8,
                            label=f"CC   RMSE={row['cc_rmse']:.4f}")
        axes[c_idx, 0].plot(th, grp["soc_ren"],  color=C_REN,  lw=1.3, alpha=0.85,
                            label=f"REN  RMSE={row['ren_rmse']:.4f}")
        for ph, col in [("discharge","#ff8c42"),("charge","#00e5ff")]:
            p = grp[grp["phase"] == ph]
            if len(p) > 0:
                axes[c_idx, 0].axvspan(p["time_h"].iloc[0], p["time_h"].iloc[-1],
                                       color=col, alpha=0.07)
        axes[c_idx, 0].set_ylabel("SOC"); axes[c_idx, 0].set_ylim(-0.02, 1.02)
        axes[c_idx, 0].set_title(
            f"Cycle {int(row['cycle'])+1}  |  {row['rmse_improvement']:+.1f}% RMSE  |  "
            f"end CC err: {row['end_cc_err']:+.4f}", fontweight="bold")
        axes[c_idx, 0].legend(fontsize=8)

        axes[c_idx, 1].plot(th, grp["abs_err_cc"],  color=C_CC,  lw=0.9, alpha=0.75,
                            label=f"CC  max={row['cc_max_err']:.4f}")
        axes[c_idx, 1].plot(th, grp["abs_err_ren"], color=C_REN, lw=0.9, alpha=0.85,
                            label=f"REN max={row['ren_max_err']:.4f}")
        axes[c_idx, 1].set_ylabel("|Error|"); axes[c_idx, 1].set_xlabel("Time (hours)")
        axes[c_idx, 1].set_title(f"|Error| — Cycle {int(row['cycle'])+1}", fontweight="bold")
        axes[c_idx, 1].legend(fontsize=8)

    plt.suptitle("Per-Cycle Detail  (Pure Observer)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("    OK cycle_details.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cycle test — Pure Observer REN")
    parser.add_argument("--cycles",      type=int,   default=5)
    parser.add_argument("--current",     type=float, default=150.0)
    parser.add_argument("--flow",        type=float, default=20.0)
    parser.add_argument("--soc-low",     type=float, default=0.30)
    parser.add_argument("--soc-high",    type=float, default=0.70)
    parser.add_argument("--rest",        type=int,   default=300)
    parser.add_argument("--cc-bias",     type=float, default=3.0)
    parser.add_argument("--cc-init-err", type=float, default=0.0)
    parser.add_argument("--no-plots",    action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*64}")
    print(f"  Continuous Cycle Test v4  (Pure Observer)")
    print(f"{'='*64}")
    print(f"  Cycles        : {args.cycles}")
    print(f"  Current       : {args.current:.0f} A")
    print(f"  Flow          : {args.flow:.0f} LPM")
    print(f"  SOC window    : {args.soc_low:.2f} -> {args.soc_high:.2f}")
    print(f"  CC bias       : {args.cc_bias:.2f} A")
    print(f"  CC init error : {args.cc_init_err*100:.1f}%")
    print(f"  Features      : {len(FEATURE_COLS)}  {FEATURE_COLS}")

    df, cyc_df, info = run_cycle_test(args)

    ov  = info["ov_cc"]; ovr = info["ov_ren"]
    n_improved = (cyc_df["ren_rmse"] < cyc_df["cc_rmse"]).sum()
    n          = len(cyc_df)

    print(f"\n{'='*64}  PER-CYCLE")
    print(f"  {'Cycle':>6}  {'CC RMSE':>10}  {'REN RMSE':>10}  "
          f"{'Improvement':>12}  {'End CC err':>12}")
    print(f"  {'-'*60}")
    for _, row in cyc_df.iterrows():
        print(f"  {int(row['cycle'])+1:>6}  {row['cc_rmse']:>10.5f}  "
              f"{row['ren_rmse']:>10.5f}  {row['rmse_improvement']:>+11.1f}%  "
              f"{row['end_cc_err']:>+12.5f}")

    print(f"\n{'='*64}  OVERALL")
    print(f"  CC     RMSE={ov['rmse']:.5f}  MAE={ov['mae']:.5f}  bias={ov['mean_bias']:+.5f}")
    print(f"  CC+REN RMSE={ovr['rmse']:.5f}  MAE={ovr['mae']:.5f}  bias={ovr['mean_bias']:+.5f}")
    print(f"  RMSE improvement : {(1-ovr['rmse']/ov['rmse'])*100:+.1f}%")
    print(f"  Cycles improved  : {n_improved}/{n}")
    print(f"  Corr(CC_err,corr): {info['r']:.4f}  (target -1.0)")
    print(f"  CC trend         : {info['cc_trend']:+.6f}/cycle")
    print(f"  REN trend        : {info['ren_trend']:+.6f}/cycle")
    if info["cc_trend"] > 0.0002:
        print(f"  [CONFIRMED] CC error growing — drift detected")
    if abs(info["ren_trend"]) < abs(info["cc_trend"]):
        print(f"  [CONFIRMED] REN trend flatter — correction is bounded")
    if abs(info["ren_trend"]) > abs(info["cc_trend"]) * 0.9:
        print(f"  [NOTE] REN drift similar to CC — z may need longer training")

    cyc_df.to_csv(os.path.join(OUT_DIR, "cycle_metrics.csv"), index=False)

    lines = [
        "Cycle Test v4 — Pure Observer Architecture",
        "=" * 60,
        f"Features : {FEATURE_COLS}",
        f"PI observer : REMOVED",
        f"Gate @ I=0A : {info['gate_at_zero']:.6f}",
        f"CC bias     : {args.cc_bias:.2f} A",
        "",
        f"{'Metric':<20} {'CC+REN':>10}  {'CC':>10}  {'Improvement':>12}",
        "-" * 56,
    ]
    for key, lbl in [("rmse","RMSE"),("mae","MAE"),("max_err","Max Err"),("mean_bias","Bias")]:
        rv = ovr[key]; cv = ov[key]
        imp = f"{(1-rv/cv)*100:+.1f}%" if key != "mean_bias" else "---"
        lines.append(f"{lbl:<20} {rv:>10.5f}  {cv:>10.5f}  {imp:>12}")
    lines += [
        "", f"Cycles improved : {n_improved}/{n}",
        f"Corr(CC_err, correction) : {info['r']:.4f}  (target -1.0)",
        f"CC RMSE trend  : {info['cc_trend']:+.6f}/cycle",
        f"REN RMSE trend : {info['ren_trend']:+.6f}/cycle",
    ]
    with open(os.path.join(OUT_DIR, "cycle_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved -> {OUT_DIR}/cycle_metrics.csv  and  cycle_summary.txt")

    if not args.no_plots:
        print("\n  Generating plots...")
        plot_main(df, cyc_df, info, args, os.path.join(OUT_DIR, "cycle_test_main.png"))
        plot_cycle_details(df, cyc_df, os.path.join(OUT_DIR, "cycle_details.png"))
    print("\n  Done.")


if __name__ == "__main__":
    main()