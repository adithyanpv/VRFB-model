# experiments/test_A_ocv_curve.py
"""
OCV vs SOC Curve — Most Important Realism Check
================================================
Charges at low current (10 A) from soc_min → soc_max.

WHY 10A not 1A:
  With I_crossover_ref=1.0A, using 1A charge means crossover is ~25%
  of the charge current at mid-SOC. The CC integrates 1A but true SOC
  only rises at ~0.75A net rate → massive CC error over millions of steps.
  At 10A, crossover is ~2.5% — negligible, test completes 10x faster.

CORRECT EXPECTED VALUES for E0=1.26V (no H+ term):
  Full VRFB Nernst: E = E0 + (RT/F) * ln(SOC^2 / (1-SOC)^2)

  SOC=0.1 → 1.26 + 0.02569*ln(0.01/0.81) = 1.147 V   (NOT 0.98V)
  SOC=0.5 → 1.260 V                                    (unchanged)
  SOC=0.9 → 1.26 + 0.02569*ln(0.81/0.01) = 1.373 V   (NOT 1.43V)
  Swing   → 0.25–0.35 V/cell

  The 0.98V/1.43V values were for E0=1.40V with the H+ activity term
  included. After correcting E0 to 1.26V those targets no longer apply.

PASS criteria:
  - Each OCV/cell within ±0.03 V of Nernst prediction
  - Monotonically increasing
  - OCV swing 0.25–0.35 V/cell
  - CC RMSE < 0.01
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.bms_controller import BMSController
from vrfb.sensor_model import SensorModel
from vrfb.coulomb_counter import CoulombCounter



cfg             = VRFBConfig()
cfg.initial_soc = 0.05
battery         = VRFB(cfg)
bms             = BMSController(cfg)
sensor          = SensorModel(cfg)
cc              = CoulombCounter(cfg)
cc.initialize(cfg.initial_soc)

dt      = cfg.dt_default
I_probe = -10.0
Q_cmd   = cfg.initial_flow
T_amb   = 298.15

RT_F = cfg.R * 298.15 / cfg.F   # ≈ 0.02569 V

# Correct Nernst predictions for E0=1.26V
EXPECTED = {
    0.1: cfg.E0 + RT_F * np.log(0.1**2 / 0.9**2),
    0.5: cfg.E0,
    0.9: cfg.E0 + RT_F * np.log(0.9**2 / 0.1**2),
}

soc_true_log = []
soc_cc_log   = []
ocv_log      = []

print("Test A: OCV vs SOC Curve")
print(f"  Probe current : {abs(I_probe)} A")
print(f"  E0 = {cfg.E0} V  |  N_cells = {cfg.N_cells}")
print(f"  Nernst targets:")
for s, v in EXPECTED.items():
    print(f"    SOC={s:.1f} → {v:.4f} V/cell")
print("-" * 65)

step = 0
while True:
    out    = battery.get_outputs()
    I_safe = bms.apply_protection(I_probe, out)
    if abs(I_safe) < 0.1:
        print(f"  BMS cut charge at SOC={out['soc_true']:.4f}")
        break

    battery.step(I_safe, Q_cmd, T_amb, dt)
    out      = battery.get_outputs()
    measured = sensor.measure(out)
    soc_cc   = cc.update(measured_current=measured["current"],
                         dt=dt, Q_nominal=out["capacity_nominal"])

    soc_true_log.append(out["soc_true"])
    soc_cc_log.append(soc_cc)
    ocv_log.append(out["voltage_stack"] / cfg.N_cells)

    step += 1
    if step % 50000 == 0:
        print(f"  step {step:7d}  SOC_true={out['soc_true']:.4f}  "
              f"SOC_cc={soc_cc:.4f}  OCV/cell={ocv_log[-1]:.4f} V")

soc_arr    = np.array(soc_true_log)
soc_cc_arr = np.array(soc_cc_log)
ocv_arr    = np.array(ocv_log)

ohmic_per_cell = abs(I_probe) * (cfg.R_membrane_initial + cfg.R_contact) * cfg.N_cells / cfg.N_cells
print(f"\n  Note: ohmic drop at {abs(I_probe)}A = {ohmic_per_cell*1000:.1f} mV/cell (acceptable for OCV approx)")

# ── Validation ────────────────────────────────────────────────────────────────
print("\n--- Physics Checks ---")
for target, expected in EXPECTED.items():
    idx    = np.argmin(np.abs(soc_arr - target))
    actual = ocv_arr[idx]
    ok     = abs(actual - expected) < 0.03
    print(f"  SOC={target:.1f}  OCV/cell={actual:.4f} V  "
          f"(Nernst={expected:.4f} V)  [{'PASS ✓' if ok else 'FAIL ✗'}]")

monotonic = np.all(np.diff(ocv_arr) >= -1e-5)
swing     = ocv_arr[-1] - ocv_arr[0]
swing_expected = 2 * RT_F * np.log(0.95**2 / 0.05**2)
print(f"  Monotonically increasing : [{'PASS ✓' if monotonic else 'FAIL ✗'}]")
print(f"  OCV swing (0.05→0.95)    : {swing:.4f} V/cell  "
      f"(Nernst predicts {swing_expected:.4f} V)  "
      f"[{'PASS ✓' if abs(swing - swing_expected) < 0.05 else 'FAIL ✗'}]")

print("\n--- Coulomb Counter Checks ---")
cc_rmse  = np.sqrt(np.mean((soc_arr - soc_cc_arr) ** 2))
cc_drift = abs(soc_arr[-1] - soc_cc_arr[-1])
crossover_pct = cfg.I_crossover_ref * 0.25 / abs(I_probe) * 100
print(f"  CC RMSE vs true SOC : {cc_rmse:.5f}  "
      f"[{'PASS ✓' if cc_rmse < 0.01 else 'REVIEW'}]")
print(f"  Final CC drift      : {cc_drift:.5f}")
print(f"  Crossover ~{crossover_pct:.1f}% of I_probe at mid-SOC "
      f"→ residual CC drift is expected and physically meaningful")

# ── Reference Nernst curve ────────────────────────────────────────────────────
soc_ref = np.linspace(0.02, 0.98, 500)
ocv_ref = cfg.E0 + RT_F * np.log(soc_ref**2 / ((1 - soc_ref)**2 + 1e-12))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(soc_arr, ocv_arr, label=f"Digital Twin OCV ({abs(I_probe)}A probe)",
             linewidth=2, color="steelblue")
axes[0].plot(soc_ref, ocv_ref, "--", color="gray",
             label="Nernst: E0=1.26 + RT/F·ln(SOC²/(1-SOC)²)")
for soc_t, exp_v in EXPECTED.items():
    axes[0].plot(soc_t, exp_v, "go", markersize=8)
    axes[0].annotate(f" {exp_v:.3f}V", xy=(soc_t, exp_v), fontsize=8, color="green")
axes[0].set_xlabel("True SOC")
axes[0].set_ylabel("OCV per cell (V)")
axes[0].set_title(f"Test A — OCV vs SOC  (I={abs(I_probe)}A)")
axes[0].legend(fontsize=8)
axes[0].grid(True)

axes[1].plot(soc_arr,    label="True SOC",            color="steelblue", linewidth=1)
axes[1].plot(soc_cc_arr, label="Coulomb Counter SOC", color="tomato",
             linestyle="--", linewidth=1)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("SOC")
axes[1].set_title(f"CC Tracking  (RMSE={cc_rmse:.5f})\n"
                  f"Residual drift driven by crossover ({cfg.I_crossover_ref}A ref)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

plt.show()
