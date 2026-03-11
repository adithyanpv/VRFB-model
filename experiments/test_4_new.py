# experiments/test_4_flow_starvation.py
"""
Flow Starvation — CC Limitations Under Low Flow
=================================================
Shows two scenarios side by side:

SCENARIO A — With BMS flow derating (realistic operation):
  Flow drops to 5 LPM < flow_critical=8 LPM
  BMS Layer 6 DERATES current: I_safe = I_cmd × (5/8) = 62.5%
  Result: current drops, voltage RISES, temperature DROPS
  CC tracks current correctly but doesn't know WHY current dropped
  CC error is small BUT: CC cannot distinguish derating from real load change

SCENARIO B — BMS flow derating DISABLED (to show physics of starvation):
  Same 5 LPM flow, same I_cmd=80A, no flow derating
  Concentration overpotential builds → voltage SAGS
  CC integrates full 80A but electrochemical throughput is limited
  → CC overestimates SOC compared to true electrochemical state

This demonstrates the core CC limitation: CC integrates current regardless
of the electrochemical cause. Only an observer that sees voltage + flow
(like the REN) can distinguish derating from true delivery.

THESIS POINT:
  In Scenario A the CC error is small but for the WRONG reason — the BMS
  happened to reduce current. In real degraded systems where the BMS may
  not derate aggressively enough, the CC error from flow starvation grows.
  The REN learns the voltage-flow-SOC relationship directly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from vrfb.config       import VRFBConfig
from vrfb.vrfb_core    import VRFB
from vrfb.bms_controller import BMSController
from vrfb.sensor_model import SensorModel
from vrfb.coulomb_counter import CoulombCounter

os.makedirs("results", exist_ok=True)

SIM_TIME   = 36000   # 10 hours
I_CMD      = 80.0    # discharge current
T_AMB      = 298.15
STARVATION_START = 7200   # 2 hours in
STARVATION_END   = 28800  # 8 hours in
FLOW_NORMAL  = VRFBConfig().initial_flow          # 20 LPM
FLOW_STARVED = VRFBConfig().flow_min              # 5 LPM

print(f"Test 4: Flow Starvation")
print(f"  I_cmd          = {I_CMD} A")
print(f"  Normal flow    = {FLOW_NORMAL*60000:.0f} LPM")
print(f"  Starved flow   = {FLOW_STARVED*60000:.0f} LPM")
print(f"  flow_critical  = {VRFBConfig().flow_critical*60000:.0f} LPM")
print(f"  Starvation     = t={STARVATION_START}–{STARVATION_END}s "
      f"({(STARVATION_END-STARVATION_START)/3600:.0f} hrs)")
print("-" * 60)


def run_scenario(use_flow_derating: bool):
    """Run the simulation with or without BMS flow derating."""
    cfg             = VRFBConfig()
    cfg.initial_soc = 0.80   # start high to survive 10-hr run
    battery = VRFB(cfg)
    bms     = BMSController(cfg)
    sensor  = SensorModel(cfg)
    cc      = CoulombCounter(cfg)
    cc.initialize(cfg.initial_soc)

    dt    = cfg.dt_default
    steps = int(SIM_TIME / dt)

    t_log, soc_true_log, soc_cc_log = [], [], []
    v_log, i_log, temp_log, flow_log, err_log = [], [], [], [], []

    for step in range(steps):
        t     = step * dt
        Q_cmd = FLOW_STARVED if STARVATION_START < t < STARVATION_END else FLOW_NORMAL

        out    = battery.get_outputs()
        I_safe = bms.apply_protection(I_CMD, out)

        # Scenario B: undo BMS flow derating only, keep all other protections
        if not use_flow_derating and Q_cmd == FLOW_STARVED:
            # Recompute without Layer 6 (flow derating)
            # Simply re-apply at the pre-flow-derate level
            # We scale back up by the flow_ratio that was applied
            flow_ratio = Q_cmd / cfg.flow_critical
            if flow_ratio < 1.0 and abs(I_safe) > 0:
                I_safe = I_safe / max(flow_ratio, 1e-6)
                # Still clip to BMS global limits
                I_safe = np.clip(I_safe, -cfg.I_max, cfg.I_max)

        battery.step(I_safe, Q_cmd, T_AMB, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        soc_cc   = cc.update(measured_current=measured["current"],
                             dt=dt, Q_nominal=out["capacity_nominal"])

        t_log.append(t)
        soc_true_log.append(out["soc_true"])
        soc_cc_log.append(soc_cc)
        v_log.append(out["voltage_stack"])
        i_log.append(out["current"])
        temp_log.append(out["temperature_stack"])
        flow_log.append(Q_cmd * 60000)   # convert to LPM for display
        err_log.append(abs(out["soc_true"] - soc_cc))

        if out["soc_true"] < cfg.soc_min + 0.01:
            print(f"  SOC floor reached at t={t:.0f}s — stopping")
            break

    soc_true_arr = np.array(soc_true_log)
    soc_cc_arr   = np.array(soc_cc_log)
    rmse = np.sqrt(np.mean((soc_true_arr - soc_cc_arr)**2))
    mae  = np.mean(np.abs(soc_true_arr - soc_cc_arr))
    max_e = np.max(np.abs(soc_true_arr - soc_cc_arr))

    return {
        "t": np.array(t_log), "soc_true": soc_true_arr, "soc_cc": soc_cc_arr,
        "v": np.array(v_log), "i": np.array(i_log), "temp": np.array(temp_log),
        "flow": np.array(flow_log), "err": np.array(err_log),
        "rmse": rmse, "mae": mae, "max_e": max_e
    }


print("Running Scenario A — with BMS flow derating...")
A = run_scenario(use_flow_derating=True)
print(f"  RMSE={A['rmse']:.5f}  MAE={A['mae']:.5f}  MaxErr={A['max_e']:.5f}")

print("Running Scenario B — BMS flow derating DISABLED...")
B = run_scenario(use_flow_derating=False)
print(f"  RMSE={B['rmse']:.5f}  MAE={B['mae']:.5f}  MaxErr={B['max_e']:.5f}")

print(f"\n  CC error increase (B vs A): {B['rmse']/A['rmse']:.2f}× worse without flow protection")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Shade starvation region on all plots
for ax in axes.flat:
    ax.axvspan(STARVATION_START, STARVATION_END, alpha=0.08, color="red",
               label="Low flow period")

# SOC comparison
axes[0, 0].plot(A["t"], A["soc_true"], color="steelblue", lw=2, label="True SOC")
axes[0, 0].plot(A["t"], A["soc_cc"],   color="orange",    lw=1.5, ls="--", label="CC (with derating)")
axes[0, 0].plot(B["t"], B["soc_cc"],   color="tomato",    lw=1.5, ls=":",  label="CC (no derating)")
axes[0, 0].set_title("SOC — True vs CC\n(red shading = flow starvation zone)")
axes[0, 0].set_ylabel("SOC"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True)

# CC error
axes[0, 1].plot(A["t"], A["err"], color="steelblue", lw=1.5, label="With BMS derating")
axes[0, 1].plot(B["t"], B["err"], color="tomato",    lw=1.5, label="No flow derating")
axes[0, 1].set_title("CC SOC Error\n(no derating → error builds during starvation)")
axes[0, 1].set_ylabel("|SOC_true − SOC_cc|"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True)

# Voltage
axes[0, 2].plot(A["t"], A["v"], color="steelblue", lw=1.5, label="With derating")
axes[0, 2].plot(B["t"], B["v"], color="tomato",    lw=1.5, label="No derating")
axes[0, 2].axhline(VRFBConfig().V_min_stack, color="red", ls=":", lw=1, label=f"V_min={VRFBConfig().V_min_stack}V")
axes[0, 2].set_title("Stack Voltage\n(with derating: rises; without: sags)")
axes[0, 2].set_ylabel("Voltage (V)"); axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True)

# Current
axes[1, 0].plot(A["t"], A["i"], color="steelblue", lw=1.5, label="With derating")
axes[1, 0].plot(B["t"], B["i"], color="tomato",    lw=1.5, label="No derating")
axes[1, 0].axhline(I_CMD, color="gray", ls="--", lw=1, label=f"I_cmd={I_CMD}A", alpha=0.6)
axes[1, 0].set_title("Actual Current\n(BMS derates to 62.5% at 5 LPM)")
axes[1, 0].set_ylabel("Current (A)"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True)

# Temperature
axes[1, 1].plot(A["t"], A["temp"], color="steelblue", lw=1.5, label="With derating")
axes[1, 1].plot(B["t"], B["temp"], color="tomato",    lw=1.5, label="No derating")
axes[1, 1].axhline(VRFBConfig().T_derate, color="orange", ls=":", lw=1, label="T_derate")
axes[1, 1].set_title("Stack Temperature\n(less current = less heating = lower T)")
axes[1, 1].set_ylabel("Temperature (K)"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True)

# Flow
axes[1, 2].plot(A["t"], A["flow"], color="steelblue", lw=2)
axes[1, 2].axhline(VRFBConfig().flow_critical*60000, color="orange", ls=":",
                   label=f"flow_critical={VRFBConfig().flow_critical*60000:.0f} LPM")
axes[1, 2].set_title("Flow Rate (LPM)")
axes[1, 2].set_ylabel("Flow (LPM)"); axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True)

for ax in axes.flat:
    ax.set_xlabel("Time (s)")

plt.suptitle(
    f"Test 4 — Flow Starvation  |  I_cmd={I_CMD}A  |  "
    f"Starvation: {FLOW_STARVED*60000:.0f} LPM ({STARVATION_START//3600}–{STARVATION_END//3600}h)\n"
    f"Scenario A (BMS derating ON): RMSE={A['rmse']:.4f}  "
    f"Scenario B (derating OFF): RMSE={B['rmse']:.4f}",
    fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig("results/test_4_flow_starvation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → results/test_4_flow_starvation.png")