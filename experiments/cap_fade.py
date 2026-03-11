# experiments/test_C_capacity_fade.py
"""
Capacity Fade Over Many Cycles
================================
Runs N full charge/discharge cycles and records usable discharge capacity
per cycle. Each cycle is: CHARGE from soc_min → DISCHARGE to soc_min.

DESIGN DECISIONS (all intentional):
  - initial_soc = 0.05  →  first operation is a full charge, so every
    cycle has equal SOC swing (0.90). No partial-cycle artifact.
  - Fade reference = cycle 2, not cycle 1. Cycle 1 sometimes has a
    slightly different thermal/concentration starting point. Using
    cycle 2 as the stable baseline is more robust.
  - CC drift measured at SOC=0.50 during discharge (mid-cycle), NOT at
    the BMS cut-off boundary where both soc_true and soc_cc are clipped
    to soc_min and the difference is always forced to 0.

WHAT TO EXPECT:
  - Discharged Ah per cycle: very slowly declining (~0.004 Ah/cycle
    from k_capacity_fade=5e-8). Effect is subtle over 20 cycles but
    visible in the Q_nominal state which declines ~0.74 Ah/cycle.
  - The larger visible effect is crossover reducing usable capacity
    as half-cells go out of balance — this shows up as a gentle decline
    in actual Ah from cycle 3 onward.
  - CC drift mid-cycle grows because CC uses Q_nominal from the battery
    output, but the denominator lags the true electrochemical capacity.

PASS criteria:
  - Q_nominal at cycle 20 < Q_nominal at cycle 1  (degradation is real)
  - Q_nominal total drop < 5%                      (not catastrophically fast)
  - Discharged Ah shows a declining trend cycles 2→20
  - CC mid-cycle drift is non-zero
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
cfg.initial_soc = 0.05   # start at soc_min → first full charge gives equal swing every cycle
battery         = VRFB(cfg)
bms             = BMSController(cfg)
sensor          = SensorModel(cfg)
cc              = CoulombCounter(cfg)
cc.initialize(cfg.initial_soc)

dt       = cfg.dt_default
N_CYCLES = 20
I_dch    =  150.0
I_ch     = -150.0
Q_cmd    = cfg.initial_flow
T_amb    = 298.15

cycle_capacity_Ah  = []
cycle_Q_nominal    = []
cycle_cc_drift_mid = []    # measured at SOC≈0.50 during discharge
t = 0

print(f"Test C: Capacity Fade over {N_CYCLES} cycles")
print(f"  k_capacity_fade = {cfg.k_capacity_fade}")
print(f"  Theoretical Q   = {cfg.theoretical_capacity_Ah:.2f} Ah")
print(f"  Each cycle      = full charge (0.05→0.95) then full discharge (0.95→0.05)")
print("-" * 65)

for cycle in range(1, N_CYCLES + 1):

    # ── Phase 1: Charge ───────────────────────────────────────────────────────
    while True:
        out    = battery.get_outputs()
        I_safe = bms.apply_protection(I_ch, out)
        if abs(I_safe) < 0.1:
            break
        battery.step(I_safe, Q_cmd, T_amb, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        cc.update(measured_current=measured["current"],
                  dt=dt, Q_nominal=out["capacity_nominal"])
        t += dt

    # ── Phase 2: Discharge (record Ah + mid-cycle CC drift) ───────────────────
    Q_dch_Ah   = 0.0
    cc_mid     = None
    soc_mid    = None

    while True:
        out    = battery.get_outputs()
        I_safe = bms.apply_protection(I_dch, out)
        if abs(I_safe) < 0.1:
            break
        battery.step(I_safe, Q_cmd, T_amb, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        cc.update(measured_current=measured["current"],
                  dt=dt, Q_nominal=out["capacity_nominal"])
        Q_dch_Ah += abs(out["current"]) * dt / 3600.0
        t += dt

        # Capture CC drift when true SOC crosses 0.50 (mid-discharge)
        if cc_mid is None and out["soc_true"] <= 0.50:
            cc_mid  = abs(out["soc_true"] - cc.get_soc())
            soc_mid = out["soc_true"]

    out      = battery.get_outputs()
    Q_nom    = out["capacity_nominal"]
    cc_drift = cc_mid if cc_mid is not None else abs(out["soc_true"] - cc.get_soc())

    # Fade relative to cycle 2 (first stable full cycle) — avoids thermal artifact
    if cycle >= 3:
        ref      = cycle_capacity_Ah[1]   # cycle 2
        fade_pct = (ref - Q_dch_Ah) / ref * 100
    else:
        fade_pct = 0.0

    cycle_capacity_Ah.append(Q_dch_Ah)
    cycle_Q_nominal.append(Q_nom)
    cycle_cc_drift_mid.append(cc_drift)

    print(f"  Cycle {cycle:2d}:  Q_dch={Q_dch_Ah:.3f} Ah  "
          f"Q_nom={Q_nom:.3f} Ah  fade={fade_pct:.4f}%  "
          f"CC_drift_mid={cc_drift:.6f}")

# ── Validation ────────────────────────────────────────────────────────────────
cap_arr   = np.array(cycle_capacity_Ah)
qnom_arr  = np.array(cycle_Q_nominal)
drift_arr = np.array(cycle_cc_drift_mid)

# Q_nominal degradation (the ODE state — most reliable measure)
qnom_drop     = (qnom_arr[0] - qnom_arr[-1]) / qnom_arr[0] * 100
qnom_fade_ok  = qnom_arr[-1] < qnom_arr[0]
qnom_sane_ok  = qnom_drop < 5.0

# Discharged Ah trend (cycles 2-20)
stable_caps   = cap_arr[1:]   # skip cycle 1 (conditioning)
ah_declining  = stable_caps[-1] < stable_caps[0]
ah_drop       = (stable_caps[0] - stable_caps[-1]) / stable_caps[0] * 100

print(f"\n--- Q_nominal Degradation (ODE state) ---")
print(f"  Cycle 1  Q_nom : {qnom_arr[0]:.3f} Ah")
print(f"  Cycle 20 Q_nom : {qnom_arr[-1]:.3f} Ah")
print(f"  Total Q_nom drop : {qnom_drop:.4f}%  "
      f"[{'PASS ✓' if qnom_fade_ok else 'FAIL ✗  ← k_capacity_fade not working'}]")
print(f"  Drop < 5%        : [{'PASS ✓' if qnom_sane_ok else 'FAIL ✗  ← degradation too aggressive'}]")

print(f"\n--- Discharged Ah Trend (cycles 2→20) ---")
print(f"  Cycle 2  Q_dch : {stable_caps[0]:.3f} Ah")
print(f"  Cycle 20 Q_dch : {stable_caps[-1]:.3f} Ah")
print(f"  Ah drop        : {ah_drop:.4f}%  "
      f"[{'PASS ✓' if ah_declining else 'REVIEW — crossover may dominate'}]")

print(f"\n--- CC Mid-Cycle Drift ---")
print(f"  Mean drift : {drift_arr.mean():.6f}")
print(f"  Max drift  : {drift_arr.max():.6f}")
nonzero = drift_arr.mean() > 1e-5
print(f"  Non-zero   : [{'PASS ✓' if nonzero else 'FAIL ✗  ← CC not drifting'}]")

# ── Plot ──────────────────────────────────────────────────────────────────────
cycles = np.arange(1, N_CYCLES + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Capacity per cycle
axes[0].plot(cycles, cap_arr, marker="o", color="steelblue", linewidth=2)
axes[0].axvline(1.5, color="gray", linestyle=":", alpha=0.5)
axes[0].annotate("Cycle 1\n(conditioning)", xy=(1, cap_arr[0]),
                 xytext=(3, cap_arr[0] - 30), fontsize=8, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray"))
axes[0].set_xlabel("Cycle")
axes[0].set_ylabel("Discharge capacity (Ah)")
axes[0].set_title("Test C — Capacity per Cycle")
axes[0].grid(True)

# Q_nominal ODE state
axes[1].plot(cycles, qnom_arr, marker="s", color="tomato", linewidth=2)
axes[1].set_xlabel("Cycle")
axes[1].set_ylabel("Q_nominal (Ah)")
axes[1].set_title(f"Q_nominal Degradation\n(−{qnom_drop:.3f}% over {N_CYCLES} cycles)")
axes[1].grid(True)

# CC mid-cycle drift
axes[2].plot(cycles, drift_arr, marker="^", color="purple", linewidth=2)
axes[2].set_xlabel("Cycle")
axes[2].set_ylabel("|SOC_true − SOC_cc|  at SOC=0.50")
axes[2].set_title("CC Mid-Cycle Drift\n(measured at SOC=0.50 during discharge)")
axes[2].grid(True)

plt.suptitle(f"Test C — Capacity Fade  |  k_fade={cfg.k_capacity_fade}  |  {N_CYCLES} cycles",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("results/test_C_capacity_fade.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → results/test_C_capacity_fade.png")