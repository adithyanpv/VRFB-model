# experiments/test_B_coulombic_efficiency.py
"""
Coulombic Efficiency Check
==========================
Charges from SOC=0.05 → 0.95 then discharges back to SOC=0.05.
Measures η = Q_discharge / Q_charge using TRUE current (out["current"])
to isolate the physics efficiency, not sensor noise.

CRITICAL: start at soc_min=0.05 (not 0.10) so that charge and discharge
cover EQUAL SOC swing (both 0.90). If swings are unequal η is biased:
  η_bias = discharge_swing / charge_swing × η_coulombic
  e.g. start=0.10 → 0.90/0.85 × 0.98 = 1.038 → FAIL even if correct

CC behaviour during charge:
  - CC sees full 100A, true SOC rises at 98A-equivalent rate
  - CC overestimates SOC during charging (orange above blue in plot)
  - This is the correct signature of η_c = 0.98 being applied

PASS criteria:  0.96 ≤ η ≤ 0.995
If η ≈ 1.00  → coulombic_efficiency is NOT applied in ODE
If η > 1.00  → SOC swing asymmetry in test design (fix initial_soc)
If η < 0.95  → efficiency applied more than once
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.bms_controller import BMSController
from vrfb.sensor_model import SensorModel
from vrfb.coulomb_counter import CoulombCounter

os.makedirs("results", exist_ok=True)

cfg             = VRFBConfig()
cfg.initial_soc = 0.05   # FIX: must start at soc_min so charge and discharge
                          # swings are equal (both 0.90), otherwise η > 1
                          # purely from arithmetic (discharge covers more SOC range)
battery         = VRFB(cfg)
bms             = BMSController(cfg)
sensor          = SensorModel(cfg)
cc              = CoulombCounter(cfg)
cc.initialize(cfg.initial_soc)

dt    = cfg.dt_default
I_ch  = -100.0
I_dch =  100.0
Q_cmd = cfg.initial_flow
T_amb = 298.15

time_log     = []
soc_true_log = []
soc_cc_log   = []
current_log  = []
phase_log    = []    # 0=charge, 1=discharge

Q_charge_Ah    = 0.0
Q_discharge_Ah = 0.0
t = 0

print("Test B: Coulombic Efficiency")
print(f"  Config η = {cfg.coulombic_efficiency}  |  I={abs(I_ch)} A")
print("-" * 50)

# ── Phase 1: Charge ───────────────────────────────────────────────────────────
print("  Phase 1: Charging …")
while True:
    out    = battery.get_outputs()
    I_safe = bms.apply_protection(I_ch, out)
    if abs(I_safe) < 0.1:
        print(f"  Charge ended at SOC={out['soc_true']:.4f}  t={t:.0f}s")
        break

    battery.step(I_safe, Q_cmd, T_amb, dt)
    out      = battery.get_outputs()
    measured = sensor.measure(out)

    soc_cc = cc.update(
        measured_current=measured["current"],
        dt=dt,
        Q_nominal=out["capacity_nominal"]
    )

    # Use TRUE current for physics efficiency calculation
    Q_charge_Ah += abs(out["current"]) * dt / 3600.0

    time_log.append(t)
    soc_true_log.append(out["soc_true"])
    soc_cc_log.append(soc_cc)
    current_log.append(out["current"])
    phase_log.append(0)
    t += dt

# ── Phase 2: Discharge ────────────────────────────────────────────────────────
print("  Phase 2: Discharging …")
while True:
    out    = battery.get_outputs()
    I_safe = bms.apply_protection(I_dch, out)
    if abs(I_safe) < 0.1:
        print(f"  Discharge ended at SOC={out['soc_true']:.4f}  t={t:.0f}s")
        break

    battery.step(I_safe, Q_cmd, T_amb, dt)
    out      = battery.get_outputs()
    measured = sensor.measure(out)

    soc_cc = cc.update(
        measured_current=measured["current"],
        dt=dt,
        Q_nominal=out["capacity_nominal"]
    )

    Q_discharge_Ah += abs(out["current"]) * dt / 3600.0

    time_log.append(t)
    soc_true_log.append(out["soc_true"])
    soc_cc_log.append(soc_cc)
    current_log.append(out["current"])
    phase_log.append(1)
    t += dt

# ── Results ───────────────────────────────────────────────────────────────────
eta    = Q_discharge_Ah / Q_charge_Ah if Q_charge_Ah > 0 else 0.0
passed = 0.96 <= eta <= 0.995

soc_arr    = np.array(soc_true_log)
soc_cc_arr = np.array(soc_cc_log)
phase_arr  = np.array(phase_log)

cc_rmse  = np.sqrt(np.mean((soc_arr - soc_cc_arr) ** 2))
cc_drift = abs(soc_arr[-1] - soc_cc_arr[-1])

print(f"\n  Q_charge       = {Q_charge_Ah:.3f} Ah")
print(f"  Q_discharge    = {Q_discharge_Ah:.3f} Ah")
print(f"  η_coulombic    = {eta:.4f} ({eta*100:.2f}%)  "
      f"[{'PASS ✓' if passed else 'FAIL ✗'}]")
print(f"  Expected η     ≈ {cfg.coulombic_efficiency}")
if abs(eta - 1.0) < 0.005:
    print("  ⚠ η ≈ 1.00 — coulombic_efficiency likely NOT applied in ODE!")
print(f"\n  CC RMSE        = {cc_rmse:.5f}")
print(f"  CC final drift = {cc_drift:.5f}  (expected small — noise-driven)")

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(np.array(time_log)[phase_arr == 0], soc_arr[phase_arr == 0],
             color="steelblue", label="True SOC (charge)")
axes[0].plot(np.array(time_log)[phase_arr == 1], soc_arr[phase_arr == 1],
             color="steelblue", linestyle="--", label="True SOC (discharge)")
axes[0].plot(np.array(time_log)[phase_arr == 0], soc_cc_arr[phase_arr == 0],
             color="tomato", label="CC SOC (charge)")
axes[0].plot(np.array(time_log)[phase_arr == 1], soc_cc_arr[phase_arr == 1],
             color="tomato", linestyle="--", label="CC SOC (discharge)")
axes[0].set_ylabel("SOC")
axes[0].set_title(f"Test B — Coulombic Efficiency  η={eta:.4f}  |  CC RMSE={cc_rmse:.5f}")
axes[0].legend(fontsize=8)
axes[0].grid(True)

axes[1].plot(time_log, current_log, color="gray")
axes[1].set_ylabel("Actual current (A)")
axes[1].set_xlabel("Time (s)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("results/test_B_coulombic_efficiency.png", dpi=150)
plt.show()
print("Saved → results/test_B_coulombic_efficiency.png")