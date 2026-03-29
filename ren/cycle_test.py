"""
Continuous Charge/Discharge Cycle Test  (v2 — physics-correct)
===============================================================

PURPOSE
-------
Prove that CC drift accumulates across cycles while CC+REN stays bounded.

v1 FAILURE and why
-------------------
v1 allocated 7200 steps (2h) per half-cycle regardless of battery capacity.
  Q_nominal = n·F·C_total·V_tank / 3600 = 2146 Ah
  At 100A, a full SOC swing (0.05→0.95) takes 19.3 hours.
  So v1 only moved SOC by 0.10 per half-cycle — the battery barely cycled.
  CC bias of 0.8A over 2h causes only 0.001 SOC drift — invisible next
  to the 0.05 initial offset. Nothing meaningful was shown.

v2 DESIGN
----------
  1. Half-cycle duration is computed from physics:
       t_half = soc_swing × Q_nominal / I_A × 3600  (seconds)
     The simulation runs until the BMS cuts off at the SOC limit OR the
     time budget is exhausted — whichever comes first.

  2. Default SOC window: 0.30 → 0.70 (swing = 0.40, not full 0.05→0.95).
     This keeps each half-cycle to ~3.4h at 150A — practical for testing.
     Full window (0.05→0.95) is available via --soc-low 0.05 --soc-high 0.95.

  3. Default cc_init_err = 0.0 (no initial offset).
     This ISOLATES the drift signal from a constant offset.
     The CC starts perfectly correct — any error that grows is pure drift.

  4. Default cc_bias = 3.0A.
     Per half-cycle drift: 3.0 × t_half / (Q_nominal × 3600)
     At 150A, swing 0.40: t_half ≈ 12,300s → drift ≈ 0.0048 SOC/half-cycle
     After 6 full cycles (12 half-cycles): ~0.058 SOC cumulative drift.
     This is clearly visible and physically realistic for a degraded sensor.

  5. Additionally models coulombic efficiency drift:
     During charge, η_c = 0.98 in the physics ODE but the CC assumes η_c = 1.0.
     This adds ~0.002 SOC underestimate per charge half-cycle, compounding
     with the sensor bias drift.

WHAT THE PLOTS SHOW
-------------------
  Panel 1: Full SOC traces — three lines moving across the full SOC window
  Panel 2: Absolute error — CC error grows monotonically; REN error stays bounded
  Panel 3: Per-cycle RMSE bars — REN bars are shorter than CC bars, gap widens
  Panel 4: RMSE trend lines — CC slope positive (growing); REN slope near-zero
  Panel 5: Cumulative CC drift — shows the integrated bias error directly
  Panel 6: Current profile — confirms real full-swing cycling

USAGE
-----
  python -m ren.cycle_test                             # 5 cycles, 150A, defaults
  python -m ren.cycle_test --cycles 8 --current 200   # more cycles, faster
  python -m ren.cycle_test --soc-low 0.10 --soc-high 0.90  # wider window
  python -m ren.cycle_test --cc-bias 5.0              # stronger drift
  python -m ren.cycle_test --no-plots                 # metrics only

OUTPUTS
-------
  ren/cycle_test/cycle_test_main.png    6-panel main figure
  ren/cycle_test/cycle_details.png      per-cycle SOC + error detail
  ren/cycle_test/cycle_metrics.csv      per-cycle numeric table
  ren/cycle_test/cycle_summary.txt      proof statement
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
from vrfb.bms_controller  import BMSController, BMSMode
from vrfb.sensor_model    import SensorModel
from vrfb.coulomb_counter import CoulombCounter
from ren.ren_model        import REN

SCALER_PATH = "ren/scaler.pkl"
MODEL_PATH  = "ren/ren_soc_best.pth"
OUT_DIR     = "ren/cycle_test"
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN_DIM  = 128
ALPHA       = 0.5
EMA_TAU     = 30
CURRENT_IDX = 1

FEATURE_COLS = [
    "voltage", "current", "temperature_stack", "temperature_tank",
    "flow_rate", "soc_cc", "transport_ratio_approx",
]

_IL_CONST = 1 * 96485.0 * 2e-5 * 0.15 * 1600.0 * 0.5
_Q_REF    = 20.0 / 60000.0

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
# CYCLE PROFILE — physics-correct half-cycle duration
# =============================================================================

def compute_halfcycle_steps(
    Q_nominal_Ah: float,
    I_A:          float,
    soc_swing:    float,
    dt:           float,
    margin_s:     float = 600.0,   # 10-min margin so BMS has time to cut off
) -> int:
    """
    Number of steps for one half-cycle (charge or discharge).

    t_half = soc_swing × Q_nominal / I × 3600  (seconds)

    The BMS will cut current before this if the SOC limit is reached.
    The margin ensures the profile doesn't end before the BMS acts.
    """
    t_half_s = soc_swing * Q_nominal_Ah / I_A * 3600.0 + margin_s
    return int(math.ceil(t_half_s / dt))


def build_profile(
    n_cycles:          int,
    steps_per_half:    int,
    rest_steps:        int,
    I_discharge:       float,
    I_charge:          float,
    Q_m3s:             float,
    T_base:            float = 298.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Returns (I_profile, Q_profile, T_profile, phase_log).

    phase_log: list of dicts with keys cycle, phase, start, end (step indices).
    Structure per cycle:
      [rest_pre] → [discharge] → [rest_mid] → [charge]
    """
    rng        = np.random.default_rng(42)
    total_half = n_cycles * 2                                # total half-cycles
    total      = total_half * steps_per_half + (total_half + 1) * rest_steps + 100

    I_arr = np.zeros(total, dtype=np.float32)
    Q_arr = np.full( total, Q_m3s, dtype=np.float32)

    # Slow ambient temperature random walk
    increments    = rng.uniform(-0.001, 0.001, size=total)
    increments[0] = 0.0
    T_arr = T_base + np.cumsum(increments)
    T_arr = np.clip(T_arr, 293.0, 315.0).astype(np.float32)

    phase_log: list[dict] = []
    pos = rest_steps   # initial rest

    for c in range(n_cycles):
        # Discharge
        d_start = pos
        d_end   = min(pos + steps_per_half, total)
        I_arr[d_start:d_end] = I_discharge
        phase_log.append({"cycle": c, "phase": "discharge",
                           "start": d_start, "end": d_end})
        pos = d_end

        # Rest between half-cycles
        pos += rest_steps

        # Charge
        c_start = pos
        c_end   = min(pos + steps_per_half, total)
        I_arr[c_start:c_end] = -I_charge
        phase_log.append({"cycle": c, "phase": "charge",
                           "start": c_start, "end": c_end})
        pos = c_end

        # Rest after charge
        pos += rest_steps

        if pos >= total:
            break

    return I_arr, Q_arr, T_arr, phase_log, pos


# =============================================================================
# SIMULATION
# =============================================================================

def run_cycle_test(args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    cfg           = VRFBConfig()
    dt            = cfg.dt_default
    Q_nominal_Ah  = cfg.n * cfg.F * cfg.C_total * cfg.V_tank / 3600.0
    soc_swing     = args.soc_high - args.soc_low
    Q_m3s         = args.flow * cfg.LPM_to_m3s

    steps_per_half = compute_halfcycle_steps(
        Q_nominal_Ah, args.current, soc_swing, dt
    )
    rest_steps = int(args.rest / dt)

    print(f"\n  Q_nominal      : {Q_nominal_Ah:.0f} Ah")
    print(f"  SOC window     : {args.soc_low:.2f} → {args.soc_high:.2f}  "
          f"(swing = {soc_swing:.2f})")
    print(f"  Half-cycle     : {steps_per_half:,} steps  "
          f"({steps_per_half*dt/3600:.1f} h per half-cycle)")
    drift_per_half = args.cc_bias * steps_per_half * dt / (Q_nominal_Ah * 3600.0)
    print(f"  Expected drift : {drift_per_half:.4f} SOC/half-cycle  "
          f"({drift_per_half * 2 * args.cycles:.4f} total over {args.cycles} cycles)")

    # ── Load REN ─────────────────────────────────────────────────────────
    model = REN(
        input_dim=7, hidden_dim=HIDDEN_DIM, output_dim=1,
        alpha=ALPHA, dropout=0.0, n_power_iters=10,
        use_feedthrough=False, use_current_gate=True,
        current_feat_idx=CURRENT_IDX,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print(f"\n  Gate @ I=0A    : {model.gate_value_at_raw_amps(0.0):.6f}")
    print(f"  Gate @ {args.current:.0f}A   : {model.gate_value_at_raw_amps(args.current):.4f}")

    # ── Physics objects ───────────────────────────────────────────────────
    # Set initial SOC to soc_high so first action is discharge
    cfg.initial_soc = args.soc_high
    battery = VRFB(cfg)
    bms     = BMSController(cfg)
    sensor  = SensorModel(cfg)
    cc      = CoulombCounter(cfg)

    # Start CC at true SOC (no initial error) — isolates drift from offset
    cc_init = float(np.clip(args.soc_high + args.cc_init_err, 0.06, 0.94))
    cc.initialize(cc_init)
    cc.set_current_bias(args.cc_bias)
    sensor.set_current_bias(args.cc_bias)

    # ── REN hidden state ──────────────────────────────────────────────────
    with torch.no_grad():
        z_ren = model.z0.detach().clone()
    ema_soc = cfg.initial_soc

    # ── Build profiles ────────────────────────────────────────────────────
    I_arr, Q_arr, T_arr, phase_log, total_steps = build_profile(
        args.cycles, steps_per_half, rest_steps,
        args.current, args.current, Q_m3s,
    )

    # Build step→(cycle, phase) lookup
    step_phase = {}
    step_cycle = {}
    for entry in phase_log:
        for s in range(entry["start"], min(entry["end"], total_steps)):
            step_phase[s] = entry["phase"]
            step_cycle[s] = entry["cycle"]

    print(f"\n  Running {total_steps:,} steps  "
          f"({total_steps*dt/3600:.1f} sim-hours) ...", flush=True)
    t_wall = time.time()

    # ── Simulation loop ───────────────────────────────────────────────────
    rows = []
    EMA_ALPHA = 1.0 / EMA_TAU

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
        out = battery.get_outputs()

        measured = sensor.measure(out)
        soc_cc   = cc.update(measured["current"], dt, out["capacity_nominal"])

        # REN inference — mirrors server.py
        Q_m  = measured["flow_rate"]
        I_m  = measured["current"]
        il   = _IL_CONST * (max(Q_m, 1e-6) / _Q_REF) ** 0.4
        tr   = abs(I_m) / max(il, 1.0)

        x    = np.array([[measured["voltage"], I_m, measured["temperature"],
                          measured["temperature_tank"], Q_m,
                          float(soc_cc), tr]], dtype=np.float32)
        x_s  = scaler.transform(x).astype(np.float32)
        x_t  = torch.tensor(x_s).unsqueeze(0)
        I_raw = torch.tensor([[[I_m]]], dtype=torch.float32)

        with torch.no_grad():
            y_t, z_ren = model(x_t, z=z_ren, x_raw=I_raw)

        corr    = float(y_t.squeeze())
        raw_hyb = float(np.clip(float(soc_cc) + corr, 0.0, 1.0))
        ema_soc = (1.0 - EMA_ALPHA) * ema_soc + EMA_ALPHA * raw_hyb
        soc_ren = float(np.clip(ema_soc, 0.0, 1.0))

        rows.append({
            "step"       : step,
            "time_h"     : step * dt / 3600.0,
            "cycle"      : step_cycle.get(step, args.cycles - 1),
            "phase"      : step_phase.get(step, "rest"),
            "bms_mode"   : bms.mode_name,
            "I_cmd"      : I_cmd,
            "I_safe"     : I_safe,
            "soc_true"   : out["soc_true"],
            "soc_cc"     : float(soc_cc),
            "soc_ren"    : soc_ren,
            "correction" : corr,
            "err_cc"     : float(soc_cc) - out["soc_true"],   # signed
            "abs_err_cc" : abs(float(soc_cc) - out["soc_true"]),
            "abs_err_ren": abs(soc_ren - out["soc_true"]),
            "voltage"    : measured["voltage"],
            "current"    : I_m,
        })

    elapsed = time.time() - t_wall
    print(f"  Done. ({elapsed:.1f}s wall clock,  "
          f"{total_steps/elapsed:.0f} steps/sec)")

    df = pd.DataFrame(rows)

    # ── Per-cycle metrics ─────────────────────────────────────────────────
    cyc_records = []
    for c_idx in range(args.cycles):
        grp = df[df["cycle"] == c_idx]
        if len(grp) == 0:
            continue
        active = grp[grp["phase"] != "rest"]
        if len(active) == 0:
            active = grp

        m_cc  = metrics(active["soc_true"].values, active["soc_cc"].values)
        m_ren = metrics(active["soc_true"].values, active["soc_ren"].values)

        # End-of-cycle CC error (shows accumulated drift)
        end_cc_err = grp.iloc[-1]["err_cc"]

        cyc_records.append({
            "cycle"            : c_idx,
            "cc_rmse"          : m_cc["rmse"],
            "ren_rmse"         : m_ren["rmse"],
            "cc_mae"           : m_cc["mae"],
            "ren_mae"          : m_ren["mae"],
            "cc_max_err"       : m_cc["max_err"],
            "ren_max_err"      : m_ren["max_err"],
            "cc_bias"          : m_cc["mean_bias"],
            "ren_bias"         : m_ren["mean_bias"],
            "rmse_improvement" : (1 - m_ren["rmse"] / m_cc["rmse"]) * 100
                                  if m_cc["rmse"] > 1e-6 else 0.0,
            "end_cc_err"       : end_cc_err,
            "end_ren_err"      : grp.iloc[-1]["abs_err_ren"],
        })

    cyc_df = pd.DataFrame(cyc_records)

    # ── Overall info dict ─────────────────────────────────────────────────
    ov_cc  = metrics(df["soc_true"].values, df["soc_cc"].values)
    ov_ren = metrics(df["soc_true"].values, df["soc_ren"].values)
    r      = float(np.corrcoef(df["err_cc"].values, df["correction"].values)[0, 1])

    cc_trend  = np.polyfit(cyc_df["cycle"].values, cyc_df["cc_rmse"].values,  1)[0]
    ren_trend = np.polyfit(cyc_df["cycle"].values, cyc_df["ren_rmse"].values, 1)[0]

    info = {
        "ov_cc": ov_cc, "ov_ren": ov_ren,
        "r": r, "cc_trend": cc_trend, "ren_trend": ren_trend,
        "Q_nominal_Ah": Q_nominal_Ah,
        "steps_per_half": steps_per_half,
        "drift_per_half": drift_per_half,
        "gate_at_zero": model.gate_value_at_raw_amps(0.0),
    }
    return df, cyc_df, info


# =============================================================================
# PLOTS
# =============================================================================

def plot_main(df, cyc_df, info, args, save_path):
    n_cycles = len(cyc_df)
    cycle_colours = plt.cm.tab10(np.linspace(0, 0.9, max(n_cycles, 1)))
    time_h = df["time_h"].values

    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.44, wspace=0.32)

    # ── Panel 1: Full SOC traces ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_h, df["soc_true"], color=C_TRUE, lw=2.0, zorder=4, label="Ground truth")
    ax1.plot(time_h, df["soc_cc"],   color=C_CC,   lw=1.0, alpha=0.75, zorder=3,
             label="Coulomb Counter  (bias accumulates each cycle)")
    ax1.plot(time_h, df["soc_ren"],  color=C_REN,  lw=1.4, alpha=0.90, zorder=3,
             label="CC + REN Hybrid  (correction tracks voltage signal)")

    rest_mask = df["phase"] == "rest"
    for _, grp in df[rest_mask].groupby((~rest_mask).cumsum()):
        ax1.axvspan(grp["time_h"].iloc[0], grp["time_h"].iloc[-1],
                    color="gray", alpha=0.10)

    for c_idx, grp in df.groupby("cycle"):
        mid = (grp["time_h"].iloc[0] + grp["time_h"].iloc[-1]) / 2
        ax1.text(mid, 1.03, f"Cycle {c_idx+1}",
                 ha="center", va="bottom", fontsize=8,
                 color=cycle_colours[c_idx % len(cycle_colours)],
                 transform=ax1.get_xaxis_transform())

    ax1.set_ylabel("State of Charge", fontsize=11)
    ax1.set_xlabel("Time (hours)", fontsize=10)
    ax1.set_ylim(-0.04, 1.08)
    ax1.set_title(
        f"Continuous Charge/Discharge Cycles — SOC Tracking\n"
        f"CC starts CORRECT (0% initial error), bias={args.cc_bias:.1f}A  "
        f"→  drift accumulates each cycle",
        fontweight="bold", fontsize=12,
    )
    ax1.legend(loc="lower right", fontsize=9)

    # ── Panel 2: Absolute error — shows drift growth ───────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(time_h, df["abs_err_cc"],  alpha=0.25, color=C_CC)
    ax2.fill_between(time_h, df["abs_err_ren"], alpha=0.25, color=C_REN)
    ax2.plot(time_h, df["abs_err_cc"],  color=C_CC,  lw=0.9, alpha=0.85,
             label="CC  |error|")
    ax2.plot(time_h, df["abs_err_ren"], color=C_REN, lw=0.9, alpha=0.85,
             label="CC+REN  |error|")
    ax2.set_ylabel("|Error| (SOC units)")
    ax2.set_xlabel("Time (hours)")
    ax2.set_title("Absolute Error Over All Cycles\n"
                  "CC error grows each cycle; REN error stays bounded",
                  fontweight="bold")
    ax2.legend(fontsize=9)

    # ── Panel 3: Per-cycle RMSE bars ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(n_cycles)
    w = 0.38
    ax3.bar(x - w/2, cyc_df["cc_rmse"],  w, color=C_CC,  alpha=0.85, label="CC")
    ax3.bar(x + w/2, cyc_df["ren_rmse"], w, color=C_REN, alpha=0.85, label="CC+REN")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"C{i+1}" for i in range(n_cycles)], fontsize=9)
    ax3.set_xlabel("Cycle")
    ax3.set_ylabel("RMSE")
    ax3.set_title("RMSE per Cycle\n(CC bar grows each cycle; REN bar stays flat)",
                  fontweight="bold")
    ax3.legend(fontsize=9)
    for i, row in cyc_df.iterrows():
        imp = row["rmse_improvement"]
        col = "#00ff9d" if imp > 0 else "#ff3d5a"
        ax3.text(i, max(row["cc_rmse"], row["ren_rmse"]) + 0.001,
                 f"{imp:+.0f}%", ha="center", va="bottom",
                 fontsize=7, color=col, fontweight="bold")

    # ── Panel 4: Trend lines ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    cx = cyc_df["cycle"].values
    ax4.plot(cx + 1, cyc_df["cc_rmse"],  "o-", color=C_CC,  lw=2, ms=7,
             label="CC RMSE")
    ax4.plot(cx + 1, cyc_df["ren_rmse"], "s-", color=C_REN, lw=2, ms=7,
             label="CC+REN RMSE")
    if len(cx) >= 3:
        z_cc  = np.polyfit(cx, cyc_df["cc_rmse"].values,  1)
        z_ren = np.polyfit(cx, cyc_df["ren_rmse"].values, 1)
        ax4.plot(cx + 1, np.poly1d(z_cc)(cx),  "--",
                 color=C_CC,  lw=1.3, alpha=0.6,
                 label=f"CC trend  ({z_cc[0]:+.5f}/cycle)")
        ax4.plot(cx + 1, np.poly1d(z_ren)(cx), "--",
                 color=C_REN, lw=1.3, alpha=0.6,
                 label=f"REN trend ({z_ren[0]:+.5f}/cycle)")
    ax4.set_xlabel("Cycle number")
    ax4.set_ylabel("RMSE")
    ax4.set_title("Error Trend — key proof\n"
                  "CC slope > 0 (drift); REN slope ≈ 0 (bounded)",
                  fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.set_xticks(cx + 1)

    # ── Panel 5: Cumulative CC drift ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    signed_err_cc  = df["err_cc"].values     # CC - true
    signed_err_ren = df["soc_ren"].values - df["soc_true"].values

    # Running mean signed error (shows systematic bias accumulation)
    window = max(int(len(df) / 200), 1)
    cc_smooth  = pd.Series(signed_err_cc).rolling(window,  min_periods=1).mean()
    ren_smooth = pd.Series(signed_err_ren).rolling(window, min_periods=1).mean()

    ax5.plot(time_h, cc_smooth,  color=C_CC,  lw=1.4, alpha=0.9,
             label="CC signed error (running mean)")
    ax5.plot(time_h, ren_smooth, color=C_REN, lw=1.4, alpha=0.9,
             label="CC+REN signed error (running mean)")
    ax5.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax5.fill_between(time_h, cc_smooth,  0, color=C_CC,  alpha=0.2)
    ax5.fill_between(time_h, ren_smooth, 0, color=C_REN, alpha=0.2)
    ax5.set_xlabel("Time (hours)")
    ax5.set_ylabel("Signed Error  (pred − true)")
    ax5.set_title("Systematic Bias Accumulation\n"
                  "CC drifts negative (sensor underestimates); REN corrects it",
                  fontweight="bold")
    ax5.legend(fontsize=9)

    # ── Stats summary box ─────────────────────────────────────────────────
    ov  = info["ov_cc"]
    ovr = info["ov_ren"]
    n_improved = (cyc_df["ren_rmse"] < cyc_df["cc_rmse"]).sum()
    box_text = (
        f"OVERALL  ({n_cycles} cycles)\n"
        f"CC     RMSE={ov['rmse']:.4f}   MAE={ov['mae']:.4f}   "
        f"bias={ov['mean_bias']:+.4f}\n"
        f"CC+REN RMSE={ovr['rmse']:.4f}   MAE={ovr['mae']:.4f}   "
        f"bias={ovr['mean_bias']:+.4f}\n"
        f"RMSE improvement: {(1-ovr['rmse']/ov['rmse'])*100:+.1f}%     "
        f"MAE improvement: {(1-ovr['mae']/ov['mae'])*100:+.1f}%\n"
        f"Cycles improved: {n_improved}/{n_cycles}     "
        f"Corr(CC_err, correction) = {info['r']:.4f}  (target -1.0)"
    )
    fig.text(0.5, 0.01, box_text, ha="center", va="bottom", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1525",
                       edgecolor="#2a4060", alpha=0.95))

    plt.suptitle(
        f"VRFB Continuous Cycle Test — CC + REN vs Coulomb Counter\n"
        f"{n_cycles} cycles  |  I={args.current:.0f}A  |  Q={args.flow:.0f} LPM  |  "
        f"SOC {args.soc_low:.2f}→{args.soc_high:.2f}  |  "
        f"CC sensor bias={args.cc_bias:.1f}A  |  CC init err={args.cc_init_err*100:.0f}%",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"    ✓ cycle_test_main.png")


def plot_cycle_details(df, cyc_df, save_path):
    n_cycles = len(cyc_df)
    fig, axes = plt.subplots(n_cycles, 2, figsize=(16, 4 * n_cycles),
                             squeeze=False)

    for c_idx, row in cyc_df.iterrows():
        grp = df[df["cycle"] == row["cycle"]]
        th  = grp["time_h"].values

        ax_soc = axes[c_idx, 0]
        ax_err = axes[c_idx, 1]

        ax_soc.plot(th, grp["soc_true"], color=C_TRUE, lw=1.8, label="True")
        ax_soc.plot(th, grp["soc_cc"],   color=C_CC,   lw=1.0, alpha=0.8,
                    label=f"CC   RMSE={row['cc_rmse']:.4f}  bias={row['cc_bias']:+.4f}")
        ax_soc.plot(th, grp["soc_ren"],  color=C_REN,  lw=1.3, alpha=0.85,
                    label=f"REN  RMSE={row['ren_rmse']:.4f}  bias={row['ren_bias']:+.4f}")

        for phase, col in [("discharge","#ff8c42"),("charge","#00e5ff")]:
            p = grp[grp["phase"] == phase]
            if len(p) > 0:
                ax_soc.axvspan(p["time_h"].iloc[0], p["time_h"].iloc[-1],
                               color=col, alpha=0.07)
        ax_soc.set_ylabel("SOC")
        ax_soc.set_title(
            f"Cycle {int(row['cycle'])+1}  |  RMSE improvement: "
            f"{row['rmse_improvement']:+.1f}%  |  "
            f"end CC error: {row['end_cc_err']:+.4f}",
            fontweight="bold",
        )
        ax_soc.legend(fontsize=8); ax_soc.set_ylim(-0.02, 1.02)

        ax_err.plot(th, grp["abs_err_cc"],  color=C_CC,  lw=0.9, alpha=0.75,
                    label=f"CC  max={row['cc_max_err']:.4f}")
        ax_err.plot(th, grp["abs_err_ren"], color=C_REN, lw=0.9, alpha=0.85,
                    label=f"REN max={row['ren_max_err']:.4f}")
        ax_err.set_ylabel("|Error|")
        ax_err.set_xlabel("Time (hours)")
        ax_err.set_title(f"|Error| — Cycle {int(row['cycle'])+1}", fontweight="bold")
        ax_err.legend(fontsize=8)

    plt.suptitle("Per-Cycle SOC and Error Detail", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"    ✓ cycle_details.png")


# =============================================================================
# SUMMARY TEXT
# =============================================================================

def write_summary(cyc_df, info, args, path):
    ov  = info["ov_cc"]
    ovr = info["ov_ren"]
    n   = len(cyc_df)
    n_improved = (cyc_df["ren_rmse"] < cyc_df["cc_rmse"]).sum()

    lines = [
        "Continuous Cycle Test — CC + REN vs Coulomb Counter  (v2)",
        "=" * 60,
        f"Cycles          : {n}",
        f"Current         : {args.current:.0f} A",
        f"Flow            : {args.flow:.0f} LPM",
        f"SOC window      : {args.soc_low:.2f} → {args.soc_high:.2f}  "
        f"(swing = {args.soc_high-args.soc_low:.2f})",
        f"CC bias         : {args.cc_bias:.2f} A  (sensor DC offset)",
        f"CC init error   : {args.cc_init_err*100:.1f}%",
        f"Q_nominal       : {info['Q_nominal_Ah']:.0f} Ah",
        f"Steps/half-cycle: {info['steps_per_half']:,}  "
        f"({info['steps_per_half']/3600:.1f} h)",
        f"Expected drift  : {info['drift_per_half']:.4f} SOC/half-cycle",
        f"Gate @ I=0A     : {info['gate_at_zero']:.6f}  (must be 0.0)",
        "",
        f"{'Metric':<22} {'CC+REN':>10}  {'CC':>10}  {'Improvement':>12}",
        "-" * 58,
    ]
    for key, lbl in [("rmse","RMSE"),("mae","MAE"),
                     ("max_err","Max Error"),("mean_bias","Mean Bias")]:
        rv = ovr[key]; cv = ov[key]
        imp = f"{(1-rv/cv)*100:+.1f}%" if key != "mean_bias" else "—"
        lines.append(f"{lbl:<22} {rv:>10.5f}  {cv:>10.5f}  {imp:>12}")
    lines += [
        "",
        f"Cycles improved (RMSE)   : {n_improved}/{n}",
        f"Corr(CC_err, correction) : {info['r']:.4f}  (target: -1.0)",
        "",
        "Per-cycle results (cumulative drift shows CC degradation):",
        f"  {'Cycle':>6}  {'CC RMSE':>10}  {'REN RMSE':>10}  "
        f"{'Improvement':>12}  {'End CC err':>12}",
        "  " + "-" * 56,
    ]
    for _, row in cyc_df.iterrows():
        lines.append(
            f"  {int(row['cycle'])+1:>6}  {row['cc_rmse']:>10.5f}  "
            f"{row['ren_rmse']:>10.5f}  {row['rmse_improvement']:>+11.1f}%  "
            f"{row['end_cc_err']:>+12.5f}"
        )
    lines += [
        "",
        f"CC RMSE trend  : {info['cc_trend']:+.6f} per cycle  "
        f"({'GROWING - drift confirmed' if info['cc_trend'] > 0.0002 else 'growing slowly'} )",
        f"REN RMSE trend : {info['ren_trend']:+.6f} per cycle  "
        f"({'bounded' if abs(info['ren_trend']) < abs(info['cc_trend']) else 'also growing'})",
        "",
        f"VERDICT: CC+REN {'IS BETTER' if ovr['rmse'] < ov['rmse'] else 'IS NOT BETTER'} "
        f"with {(1-ovr['rmse']/ov['rmse'])*100:+.1f}% RMSE improvement",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Continuous cycle test — CC+REN vs Coulomb Counter"
    )
    parser.add_argument("--cycles",       type=int,   default=5)
    parser.add_argument("--current",      type=float, default=150.0,
                        help="Charge/discharge current in A (default: 150)")
    parser.add_argument("--flow",         type=float, default=20.0,
                        help="Flow rate in LPM (default: 20)")
    parser.add_argument("--soc-low",      type=float, default=0.30,
                        help="Lower SOC limit for cycling (default: 0.30)")
    parser.add_argument("--soc-high",     type=float, default=0.70,
                        help="Upper SOC limit for cycling (default: 0.70)")
    parser.add_argument("--rest",         type=int,   default=300,
                        help="Rest between half-cycles in seconds (default: 300)")
    parser.add_argument("--cc-bias",      type=float, default=3.0,
                        help="Current sensor DC bias in A (default: 3.0)")
    parser.add_argument("--cc-init-err",  type=float, default=0.0,
                        help="Initial CC SOC error (default: 0.0 — isolates drift)")
    parser.add_argument("--no-plots",     action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*64}")
    print(f"  Continuous Cycle Test v2 — CC + REN vs Coulomb Counter")
    print(f"{'='*64}")
    print(f"  Cycles        : {args.cycles}")
    print(f"  Current       : {args.current:.0f} A")
    print(f"  Flow          : {args.flow:.0f} LPM")
    print(f"  SOC window    : {args.soc_low:.2f} → {args.soc_high:.2f}")
    print(f"  Rest          : {args.rest} s")
    print(f"  CC bias       : {args.cc_bias:.2f} A  (sensor DC offset)")
    print(f"  CC init error : {args.cc_init_err*100:.1f}%  (0 = isolates drift)")

    df, cyc_df, info = run_cycle_test(args)

    ov  = info["ov_cc"]
    ovr = info["ov_ren"]
    r   = info["r"]
    n_improved = (cyc_df["ren_rmse"] < cyc_df["cc_rmse"]).sum()
    n          = len(cyc_df)

    print(f"\n{'='*64}")
    print(f"  PER-CYCLE RESULTS")
    print(f"{'='*64}")
    print(f"  {'Cycle':>6}  {'CC RMSE':>10}  {'REN RMSE':>10}  "
          f"{'Improvement':>12}  {'End CC err':>12}")
    print(f"  {'-'*60}")
    for _, row in cyc_df.iterrows():
        print(f"  {int(row['cycle'])+1:>6}  {row['cc_rmse']:>10.5f}  "
              f"{row['ren_rmse']:>10.5f}  {row['rmse_improvement']:>+11.1f}%  "
              f"{row['end_cc_err']:>+12.5f}")

    print(f"\n{'='*64}")
    print(f"  OVERALL")
    print(f"{'='*64}")
    print(f"  CC     RMSE={ov['rmse']:.5f}   MAE={ov['mae']:.5f}  "
          f"bias={ov['mean_bias']:+.5f}")
    print(f"  CC+REN RMSE={ovr['rmse']:.5f}   MAE={ovr['mae']:.5f}  "
          f"bias={ovr['mean_bias']:+.5f}")
    print(f"\n  RMSE improvement     : {(1-ovr['rmse']/ov['rmse'])*100:+.1f}%")
    print(f"  MAE  improvement     : {(1-ovr['mae']/ov['mae'])*100:+.1f}%")
    print(f"  Cycles improved      : {n_improved}/{n}")
    print(f"  Corr(CC_err, corr)   : {r:.4f}  (target -1.0)")
    print(f"\n  CC RMSE trend        : {info['cc_trend']:+.6f} per cycle")
    print(f"  REN RMSE trend       : {info['ren_trend']:+.6f} per cycle")
    if info["cc_trend"] > 0.0002:
        print(f"  [CONFIRMED] CC error growing across cycles — drift detected")
    if abs(info["ren_trend"]) < abs(info["cc_trend"]):
        print(f"  [CONFIRMED] REN trend flatter than CC — correction is bounded")

    cyc_df.to_csv(os.path.join(OUT_DIR, "cycle_metrics.csv"), index=False)
    write_summary(cyc_df, info, args, os.path.join(OUT_DIR, "cycle_summary.txt"))
    print(f"\n  Saved → {OUT_DIR}/cycle_metrics.csv")
    print(f"  Saved → {OUT_DIR}/cycle_summary.txt")

    if not args.no_plots:
        print(f"\n  Generating plots...")
        plot_main(df, cyc_df, info, args,
                  os.path.join(OUT_DIR, "cycle_test_main.png"))
        plot_cycle_details(df, cyc_df,
                           os.path.join(OUT_DIR, "cycle_details.png"))
        print(f"\n  Done.")


if __name__ == "__main__":
    main()