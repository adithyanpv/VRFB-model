# experiments/test_6_flow_sweep_dynamic.py
"""
Flow Sweep Under Constant Discharge
=====================================
Steps flow through 4 levels while discharging at 120A.
Shows how flow rate affects:
  - Limiting current (I_limit)
  - Transport utilisation ratio
  - Voltage and temperature

FIX: Added warmup period so internal flow state matches the first
commanded flow before logging begins — eliminates the initial dip.
FIX: Removed print() inside loop (was flooding terminal with 10,000 lines).
"""

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.coulomb_counter import CoulombCounter
from vrfb.sensor_model import SensorModel
from vrfb.bms_controller import BMSController
from vrfb.utils import compute_metrics, save_plots, save_metrics

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Initialize
# ------------------------------------------------------------
cfg = VRFBConfig()
battery = VRFB(cfg)
bms = BMSController(cfg)
cc = CoulombCounter(cfg)
sensor = SensorModel(cfg)

cc.initialize(cfg.initial_soc)

dt = cfg.dt_default
sim_time = 10000
steps = int(sim_time / dt)

# ------------------------------------------------------------
# WARMUP — match internal flow state to first commanded flow
# ------------------------------------------------------------
# The battery initialises at cfg.initial_flow (20 LPM).
# The test starts at 2 LPM — without warmup the flow graph shows
# a downward transient for the first ~200 steps as the internal
# flow dynamics settle from 20 LPM → 2 LPM.
# Running a silent warmup at Q=2 LPM (I=0) removes this artefact.

Q_FIRST      = 2.0 * cfg.LPM_to_m3s
WARMUP_STEPS = 300

print("Running warmup (300 steps at 2 LPM, I=0)...")
for _ in range(WARMUP_STEPS):
    battery.step(0.0, Q_FIRST, 298.15, dt)

# Re-initialise CC after warmup so estimate starts clean
cc.initialize(cfg.initial_soc)
print("Warmup complete. Starting logged simulation...\n")

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
time_log        = []
true_soc        = []
est_soc         = []
voltage         = []
current         = []
temperature     = []
flow            = []
i_limit_log     = []
pump_power      = []
transport_ratio = []

# ------------------------------------------------------------
# Simulation Loop
# ------------------------------------------------------------
for step in range(steps):

    t = step * dt

    # Constant discharge current
    I_cmd = -120

    # Flow Sweep — 4 increasing levels
    if t < 2500:
        Q_cmd = 2.0  * cfg.LPM_to_m3s   # below flow_critical → BMS derates
    elif t < 5000:
        Q_cmd = 10.0 * cfg.LPM_to_m3s   # just above critical
    elif t < 7500:
        Q_cmd = 30.0 * cfg.LPM_to_m3s   # comfortable
    else:
        Q_cmd = 60.0 * cfg.LPM_to_m3s   # transport easily satisfied

    T = 298.15

    # BMS Protection
    prev_out = battery.get_outputs()
    I_safe   = bms.apply_protection(I_cmd, prev_out)

    # Physics step
    battery.step(I_safe, Q_cmd, T, dt)
    out = battery.get_outputs()

    # Sensor + CC
    measured = sensor.measure(out)
    soc_cc   = cc.update(
        measured_current=measured["current"],
        dt=dt,
        Q_nominal=out["capacity_nominal"]
    )

    # Logging
    time_log.append(t)
    true_soc.append(out["soc_true"])
    est_soc.append(soc_cc)
    voltage.append(out["voltage_stack"])
    current.append(out["current"])
    temperature.append(out["temperature_stack"])
    flow.append(out["flow_rate"])
    i_limit_log.append(out["i_limit"])
    pump_power.append(out["pump_power"])
    transport_ratio.append(out["transport_ratio"])


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
true_soc = np.array(true_soc)
est_soc  = np.array(est_soc)

rmse, mae, max_e, final_e = compute_metrics(true_soc, est_soc)

print("=== TEST 6: FLOW SWEEP UNDER DYNAMIC LOAD ===")
print(f"RMSE:        {rmse:.6f}")
print(f"MAE:         {mae:.6f}")
print(f"Max Error:   {max_e:.6f}")
print(f"Final Drift: {final_e:.6f}")

# Flow levels in LPM for readable axis
flow_lpm = np.array(flow) * 60000

# ------------------------------------------------------------
# Flow level shading helper
# ------------------------------------------------------------
flow_segments = [
    (0,    2500,  2,  "#ffe8e8"),
    (2500, 5000,  10, "#fff4e0"),
    (5000, 7500,  30, "#e8f4e8"),
    (7500, 10000, 60, "#e0eeff"),
]

# ------------------------------------------------------------
# Save standard results
# ------------------------------------------------------------
folder = "results/test_6_flow_sweep_dynamic"
save_plots(folder,
           time_log, true_soc, est_soc,
           voltage, current, temperature, flow, transport_ratio)
save_metrics(folder, rmse, mae, max_e, final_e)

# ------------------------------------------------------------
# Custom plot: Current vs I_limit with flow shading
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for ax in axes.flat:
    for t0, t1, lpm, col in flow_segments:
        ax.axvspan(t0, t1, color=col, alpha=0.7,
                   label=f"{lpm} LPM" if ax == axes[0, 0] else "")

# Current vs I_limit
axes[0, 0].plot(time_log, current,     lw=2,   label="Actual current (A)")
axes[0, 0].plot(time_log, i_limit_log, lw=1.5, ls="--", label="I_limit (A)")
axes[0, 0].axhline(120, color="gray", ls=":", lw=1, label="I_cmd=120A")
axes[0, 0].set_title("Actual Current vs Limiting Current")
axes[0, 0].set_ylabel("Current (A)")
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True)

# Transport ratio
axes[0, 1].plot(time_log, transport_ratio, color="purple", lw=2)
axes[0, 1].axhline(0.70, color="orange", ls="--", lw=1, label="0.70 safe limit")
axes[0, 1].axhline(0.90, color="red",    ls="--", lw=1, label="0.90 danger")
axes[0, 1].set_title("Transport Utilisation Ratio (|I| / I_limit)")
axes[0, 1].set_ylabel("Transport Ratio")
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True)

# Flow in LPM (clean axis — no m³/s)
axes[1, 0].plot(time_log, flow_lpm, color="steelblue", lw=2)
axes[1, 0].axhline(cfg.flow_critical * 60000, color="red", ls="--", lw=1,
                   label=f"flow_critical={cfg.flow_critical*60000:.0f} LPM")
axes[1, 0].set_title("Flow Rate (LPM) — clean axis after warmup fix")
axes[1, 0].set_ylabel("Flow (LPM)")
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True)

# Pump power
axes[1, 1].plot(time_log, pump_power, color="darkorange", lw=2)
axes[1, 1].set_title("Pump Power (W) — cubic in flow rate")
axes[1, 1].set_ylabel("Pump Power (W)")
axes[1, 1].grid(True)

for ax in axes.flat:
    ax.set_xlabel("Time (s)")

# Legend for flow segments (top-left only)
axes[0, 0].legend(fontsize=7, loc="upper right")

plt.suptitle(
    "Test 6 — Flow Sweep  |  I_cmd=120A  |  T=298K\n"
    "Pink=2LPM(derated)  Yellow=10LPM  Green=30LPM  Blue=60LPM",
    fontsize=10
)
plt.tight_layout()
plt.savefig(f"{folder}/test_6_flow_sweep_custom.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {folder}/test_6_flow_sweep_custom.png")

# ------------------------------------------------------------
# PLOT 2: Voltage and SOC across flow periods
# ------------------------------------------------------------
fig2, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

v_arr    = np.array(voltage)
soc_true = np.array(true_soc)
soc_cc   = np.array(est_soc)
cc_error = np.abs(soc_true - soc_cc)
flow_lpm = np.array(flow) * 60000

period_labels_top = ["2 LPM\n(BMS derated)", "10 LPM", "30 LPM", "60 LPM"]

for ax in axes:
    for i, (t0, t1, lpm, col) in enumerate(flow_segments):
        ax.axvspan(t0, t1, color=col, alpha=0.7)
    for t_trans in [2500, 5000, 7500]:
        ax.axvline(t_trans, color="black", ls="--", lw=0.8, alpha=0.4)

# ── Panel 1: Voltage ─────────────────────────────────────────────────────────
axes[0].plot(time_log, v_arr, color="darkorange", lw=2)
axes[0].axhline(cfg.V_min_stack, color="navy", ls=":", lw=1.2,
                label=f"V_min={cfg.V_min_stack}V")
axes[0].set_ylabel("Stack Voltage (V)", fontsize=11)
axes[0].set_title(
    "Stack Voltage  —  V = OCV(SOC) − I·R_ohmic − V_concentration",
    fontsize=10)
axes[0].legend(fontsize=8); axes[0].grid(True)

# Annotations
mid_pink   = int(1250 / cfg.dt_default)
mid_yellow = int(3750 / cfg.dt_default)
mid_green  = int(6250 / cfg.dt_default)

axes[0].annotate("Low I (60A derated)\n→ V = OCV − 60×0.08\n→ high voltage",
    xy=(1250, v_arr[mid_pink]), xytext=(300, v_arr[mid_pink]-2.5),
    fontsize=7, color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

axes[0].annotate("I jumps to 120A\n→ V = OCV − 120×0.08\n→ drops 4.8V",
    xy=(2700, v_arr[min(2700, len(v_arr)-1)]),
    xytext=(3000, v_arr[min(2700, len(v_arr)-1)]-2.0),
    fontsize=7, color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

axes[0].annotate("Higher flow reduces\nconc. overpotential\n→ V slowly recovers",
    xy=(6500, v_arr[min(6500, len(v_arr)-1)]),
    xytext=(6600, v_arr[min(6500, len(v_arr)-1)]+1.5),
    fontsize=7, color="darkgreen",
    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8))

# Period labels
for i, (t0, t1, lpm, col) in enumerate(flow_segments):
    ymax = axes[0].get_ylim()[1] if axes[0].get_ylim()[1] != 0 else 50
    axes[0].text((t0+t1)/2, axes[0].get_ylim()[1]*0.99,
                 period_labels_top[i],
                 ha="center", va="top", fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=col, alpha=0.9))

# ── Panel 2: True SOC vs CC SOC ───────────────────────────────────────────────
axes[1].plot(time_log, soc_true, color="steelblue", lw=2,   label="True SOC")
axes[1].plot(time_log, soc_cc,   color="tomato",    lw=1.5,
             ls="--", label=f"CC SOC  (RMSE={rmse:.4f})")
axes[1].set_ylabel("SOC", fontsize=11)
axes[1].set_title(
    "SOC  —  discharge rate changes at each flow step (current changes)",
    fontsize=10)
axes[1].legend(fontsize=8); axes[1].grid(True)

axes[1].annotate("Slow SOC drain\n(60A in pink zone)",
    xy=(1250, float(soc_true[mid_pink])),
    xytext=(300, float(soc_true[mid_pink])+0.02),
    fontsize=7, color="gray",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

axes[1].annotate("Fast drain\n(120A released)",
    xy=(3000, float(soc_true[mid_yellow])),
    xytext=(3200, float(soc_true[mid_yellow])+0.025),
    fontsize=7, color="gray",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# ── Panel 3: CC Error ─────────────────────────────────────────────────────────
axes[2].plot(time_log, cc_error, color="purple", lw=1.5)
axes[2].fill_between(time_log, 0, cc_error, color="purple", alpha=0.15)
axes[2].set_ylabel("|SOC_true − SOC_cc|", fontsize=11)
axes[2].set_xlabel("Time (s)", fontsize=11)
axes[2].set_title(
    "CC SOC Error  —  kink at t=2500 when current jumps; error grows throughout",
    fontsize=10)
axes[2].grid(True)

kink_idx = min(int(2500 / cfg.dt_default), len(cc_error)-1)
axes[2].annotate(
    f"Kink: CC suddenly\nintegrates 120A\ninstead of 60A",
    xy=(2500, cc_error[kink_idx]),
    xytext=(3500, cc_error[kink_idx]+0.0008),
    fontsize=7, color="purple",
    arrowprops=dict(arrowstyle="->", color="purple", lw=0.8))

plt.suptitle(
    "Test 6 — Voltage and SOC During Flow Sweep  |  I_cmd=120A  |  T=298K\n"
    "Pink=2LPM(derated)  Yellow=10LPM  Green=30LPM  Blue=60LPM",
    fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f"{folder}/test_6_voltage_soc.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {folder}/test_6_voltage_soc.png")