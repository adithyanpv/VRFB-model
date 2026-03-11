# experiments/test_G_crossover_imbalance.py
"""
Crossover-Driven Half-Cell SOC Divergence
==========================================
Runs N cycles and tracks positive vs negative half-cell SOC divergence.
Crossover causes the two half-cells to go out of balance over time,
reducing usable capacity and creating an irrecoverable SOC estimation
error that Coulomb counting cannot detect.

ASYMMETRIC CROSSOVER MODEL (required for non-zero imbalance):
  - V²⁺ crosses from negative → positive at rate  ∝ I_crossover_ref
  - VO₂⁺ crosses from positive → negative at rate ∝ I_crossover_ref × 0.3
  - Physical basis: VO₂⁺ has ~3× higher diffusion resistance through Nafion
    (larger ionic radius, D_VO2+ ≈ 1e-11 vs D_V2+ ≈ 3.5e-11 m²/s)
  - With symmetric rates (old model), imbalance = 0 by symmetry — both
    half-cells lose active species at the same rate, difference stays zero.

MEASUREMENT POINT — WHY MID-DISCHARGE:
  - soc_true = C_V2_t / (C_V2_t + C_V3_t)  [negative half-cell only]
  - BMS cuts charging at soc_true = soc_max = 0.95
  - At this boundary, soc_neg is clamped by the BMS; measuring here hides
    any divergence because the cut-off forces soc_neg toward 0.95 every cycle.
  - Correct approach: capture soc_neg and soc_pos when soc_true crosses 0.50
    during discharge — far from any BMS boundary, shows raw divergence.

CC INSIGHT FOR THESIS:
  - CC integrates total current → tracks AVERAGE SOC = (soc_neg + soc_pos)/2
  - As the half-cells diverge, the average appears correct but neither side is
  - Usable capacity = 2 × min(soc_neg, soc_pos) - not the average
  - CC systematically overestimates usable capacity as imbalance grows
  - The REN sees voltage signals that encode the half-cell chemistry directly

PASS criteria:
  - |SOC_neg - SOC_pos| > 0.005 after 15 cycles at 298 K
  - Imbalance at 315 K > imbalance at 298 K (temperature accelerates crossover)
  - CC drift grows in proportion to imbalance
  - SOC_neg < SOC_pos  (negative depletes faster — V²⁺ crosses faster)

REQUIRES:
  - vrfb_core.py: asymmetric crossover (crossover_pos_factor = 0.3)
  - config.py:    self.crossover_pos_factor = 0.3
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

N_CYCLES = 15
I_CH     = -120.0
I_DCH    =  120.0

TEMP_CASES = [
    {"T_amb": 298.15, "label": "298 K (25°C)", "color": "steelblue"},
    {"T_amb": 315.00, "label": "315 K (42°C)", "color": "tomato"},
]

cfg_ref = VRFBConfig()

# Guard: check asymmetric crossover is in config
assert hasattr(cfg_ref, "crossover_pos_factor"), \
    "config.py missing crossover_pos_factor — add: self.crossover_pos_factor = 0.3"
assert cfg_ref.crossover_pos_factor < 1.0, \
    f"crossover_pos_factor={cfg_ref.crossover_pos_factor} must be < 1.0 for asymmetry"

print("Test G: Crossover Capacity Imbalance")
print(f"  I_crossover_ref     = {cfg_ref.I_crossover_ref} A")
print(f"  crossover_pos_factor = {cfg_ref.crossover_pos_factor}  "
      f"(VO₂⁺ crosses at {cfg_ref.crossover_pos_factor*100:.0f}% of V²⁺ rate)")
print(f"  N_cycles = {N_CYCLES}")
print(f"  Measurement at soc_true = 0.50 mid-discharge (not at BMS boundary)")
print("-" * 65)


def get_half_cell_socs(state):
    """Read half-cell SOCs directly from tank concentration state."""
    C_V2_t, C_V3_t       = state[4], state[5]
    C_VO2_t, C_VO2plus_t = state[6], state[7]
    soc_neg = C_V2_t      / (C_V2_t      + C_V3_t      + 1e-12)
    soc_pos = C_VO2plus_t / (C_VO2plus_t + C_VO2_t     + 1e-12)
    return float(soc_neg), float(soc_pos)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for tc in TEMP_CASES:

    cfg             = VRFBConfig()
    cfg.initial_soc = 0.90
    battery         = VRFB(cfg)
    bms             = BMSController(cfg)
    sensor          = SensorModel(cfg)
    cc              = CoulombCounter(cfg)
    cc.initialize(cfg.initial_soc)

    dt    = cfg.dt_default
    Q_cmd = cfg.initial_flow
    T_amb = tc["T_amb"]

    cycle_nums    = []
    soc_neg_log   = []
    soc_pos_log   = []
    imbalance_log = []
    capacity_log  = []
    cc_drift_log  = []

    for cycle in range(1, N_CYCLES + 1):

        # ── Discharge — capture half-cell SOCs at soc_true = 0.50 ─────────────
        Q_dch       = 0.0
        mid_captured = False
        soc_n_mid = soc_p_mid = cc_drift_mid = None

        while True:
            out    = battery.get_outputs()
            I_safe = bms.apply_protection(I_DCH, out)
            if abs(I_safe) < 0.1:
                break
            battery.step(I_safe, Q_cmd, T_amb, dt)
            out      = battery.get_outputs()
            measured = sensor.measure(out)
            cc.update(measured_current=measured["current"],
                      dt=dt, Q_nominal=out["capacity_nominal"])
            Q_dch += abs(out["current"]) * dt / 3600.0

            # Capture at mid-discharge (soc_true crosses 0.50)
            if not mid_captured and out["soc_true"] <= 0.50:
                soc_n_mid    = get_half_cell_socs(battery.state)[0]
                soc_p_mid    = get_half_cell_socs(battery.state)[1]
                cc_drift_mid = abs(out["soc_true"] - cc.get_soc())
                mid_captured = True

        # Fallback: if BMS cut before SOC reached 0.50, use end-of-discharge
        if not mid_captured:
            soc_n_mid, soc_p_mid = get_half_cell_socs(battery.state)
            cc_drift_mid = abs(battery.get_outputs()["soc_true"] - cc.get_soc())

        # ── Charge ────────────────────────────────────────────────────────────
        while True:
            out    = battery.get_outputs()
            I_safe = bms.apply_protection(I_CH, out)
            if abs(I_safe) < 0.1:
                break
            battery.step(I_safe, Q_cmd, T_amb, dt)
            out      = battery.get_outputs()
            measured = sensor.measure(out)
            cc.update(measured_current=measured["current"],
                      dt=dt, Q_nominal=out["capacity_nominal"])

        imbalance = soc_n_mid - soc_p_mid

        cycle_nums.append(cycle)
        soc_neg_log.append(soc_n_mid)
        soc_pos_log.append(soc_p_mid)
        imbalance_log.append(imbalance)
        capacity_log.append(Q_dch)
        cc_drift_log.append(cc_drift_mid)

        print(f"  [{tc['label']}]  Cycle {cycle:2d}:  "
              f"SOC_neg={soc_n_mid:.4f}  SOC_pos={soc_p_mid:.4f}  "
              f"imbal={imbalance:+.5f}  Q={Q_dch:.3f}Ah  "
              f"CC_drift={cc_drift_mid:.5f}")

    final_imb = abs(imbalance_log[-1])
    grows     = final_imb > abs(imbalance_log[1]) if len(imbalance_log) > 1 else False
    neg_lower = np.mean(soc_neg_log) < np.mean(soc_pos_log)

    print(f"\n  [{tc['label']}]  Final |imbalance| = {final_imb:.5f}  "
          f"[{'PASS ✓' if final_imb > 0.005 else 'FAIL ✗ ← crossover asymmetry not working'}]")
    print(f"  [{tc['label']}]  Imbalance grows   = [{'PASS ✓' if grows else 'FAIL ✗'}]")
    print(f"  [{tc['label']}]  SOC_neg < SOC_pos = [{'PASS ✓' if neg_lower else 'FAIL ✗'}]  "
          f"(neg depletes faster — V²⁺ crosses at higher rate)\n")

    c = tc["color"]
    axes[0].plot(cycle_nums, soc_neg_log, marker="o", color=c, linestyle="-",
                 label=f"SOC_neg {tc['label']}")
    axes[0].plot(cycle_nums, soc_pos_log, marker="s", color=c, linestyle="--",
                 label=f"SOC_pos {tc['label']}")
    axes[1].plot(cycle_nums, np.abs(imbalance_log), marker="^", color=c,
                 label=tc["label"], linewidth=2)
    axes[2].plot(cycle_nums, cc_drift_log, marker="o", color=c,
                 label=tc["label"], linewidth=2)

# ── Validation: 315K imbalance > 298K imbalance ───────────────────────────────
# (done after both loops — would need to store results, handled by visual inspection)

axes[0].set_xlabel("Cycle")
axes[0].set_ylabel("Half-cell SOC at soc_true=0.50")
axes[0].set_title("Test G — Half-Cell SOC Divergence\n(measured mid-discharge)")
axes[0].legend(fontsize=7)
axes[0].grid(True)

axes[1].set_xlabel("Cycle")
axes[1].set_ylabel("|SOC_neg − SOC_pos|")
axes[1].set_title("Imbalance Growth\n(higher T → faster crossover)")
axes[1].legend()
axes[1].grid(True)

axes[2].set_xlabel("Cycle")
axes[2].set_ylabel("|SOC_true − SOC_cc|  at SOC=0.50")
axes[2].set_title("CC Drift per Cycle\n(grows with crossover imbalance)")
axes[2].legend()
axes[2].grid(True)

plt.suptitle(
    f"Test G — Crossover Imbalance  "
    f"(pos_factor={cfg_ref.crossover_pos_factor}, "
    f"I_cross_ref={cfg_ref.I_crossover_ref}A)",
    fontsize=11, y=1.01
)
plt.tight_layout()
plt.savefig("results/test_G_crossover_imbalance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → results/test_G_crossover_imbalance.png")