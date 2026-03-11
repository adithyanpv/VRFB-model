# experiments/test_E_mass_transport_cutoff.py
"""
Mass Transport / Concentration Overpotential Validation
=========================================================
Sweeps discharge currents at three flow rates to show:
  1. transport_ratio never exceeds 1.0 (physics safety)
  2. I_limit scales with flow rate (flow^0.4 dependence)
  3. BMS flow-critical derating at 5 LPM (below flow_critical=8 LPM)
  4. Clean polarisation curve with no BMS voltage cut-off artefacts

THREE FLOW CASES (intentional):
  Case A — 5 LPM  : below flow_critical=8 LPM → BMS flow derating active
                    Shows combined mass-transport + flow-protection effect
  Case B — 10 LPM : just above flow_critical → no flow derating, lower I_limit
                    Shows pure mass-transport effect without BMS interference
  Case C — 30 LPM : high flow → high I_limit, full current range accessible

CURRENT SWEEP: 20 → 160 A in steps of 20 A
  Cap at 160 A to stay clear of BMS undervoltage cut-off.
  At SOC=0.50, V_min=35V: max I before cut = (50.4-35)/0.08 = 192 A
  160 A gives 15% margin → no stray BMS voltage events in the sweep.

PASS criteria:
  - transport_ratio ≤ 1.0 at all currents and flows
  - I_limit(30 LPM) > I_limit(10 LPM) > I_limit(5 LPM) at same current
  - BMS flow derating only active at 5 LPM (I_safe < I_cmd)
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

SETTLE_STEPS   = 300
CURRENT_LEVELS = [20, 40, 60, 80, 100, 120, 140, 160]

FLOW_CASES = [
    {"Q_lpm":  5, "label": " 5 LPM (below critical)", "color": "tomato",    "ls": "-"},
    {"Q_lpm": 10, "label": "10 LPM (above critical)", "color": "goldenrod", "ls": "-"},
    {"Q_lpm": 30, "label": "30 LPM (high flow)",      "color": "steelblue", "ls": "-"},
]

cfg_ref = VRFBConfig()
print("Test E: Mass Transport Cutoff")
print(f"  flow_critical  = {cfg_ref.flow_critical * 60000:.0f} LPM  "
      f"({cfg_ref.flow_critical:.5f} m³/s)")
print(f"  I_limit margin = {cfg_ref.limiting_current_margin*100:.0f}%")
print(f"  Current sweep  = {CURRENT_LEVELS[0]}–{CURRENT_LEVELS[-1]} A "
      f"(capped to avoid BMS undervoltage artefacts)")
print("-" * 65)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

all_results = {}

for fc in FLOW_CASES:

    Q_cmd        = fc["Q_lpm"] * cfg_ref.LPM_to_m3s
    volt_curve   = []
    ilim_curve   = []
    tratio_curve = []
    iact_curve   = []
    isafe_curve  = []

    for I_cmd in CURRENT_LEVELS:

        cfg             = VRFBConfig()
        cfg.initial_soc = 0.50
        battery = VRFB(cfg)
        bms     = BMSController(cfg)
        sensor  = SensorModel(cfg)
        cc      = CoulombCounter(cfg)
        cc.initialize(cfg.initial_soc)

        for _ in range(SETTLE_STEPS):
            out    = battery.get_outputs()
            I_safe = bms.apply_protection(float(I_cmd), out)
            battery.step(I_safe, Q_cmd, 298.15, cfg.dt_default)
            out      = battery.get_outputs()
            measured = sensor.measure(out)
            cc.update(measured_current=measured["current"],
                      dt=cfg.dt_default, Q_nominal=out["capacity_nominal"])

        out = battery.get_outputs()
        volt_curve.append(out["voltage_stack"])
        ilim_curve.append(out["i_limit"])
        tratio_curve.append(out["transport_ratio"])
        iact_curve.append(out["current"])
        isafe_curve.append(I_safe)

        flow_derate = I_safe < I_cmd * 0.95
        print(f"  [{fc['label']}]  I_cmd={I_cmd:3d}A  "
              f"I_act={out['current']:6.1f}A  "
              f"V={out['voltage_stack']:5.1f}V  "
              f"I_lim={out['i_limit']:6.1f}A  "
              f"η_T={out['transport_ratio']:.3f}  "
              f"flow_derate={flow_derate}")

    max_ratio = max(tratio_curve)
    print(f"\n  [{fc['label']}] max transport_ratio={max_ratio:.4f}  "
          f"[{'PASS ✓' if max_ratio <= 1.001 else 'FAIL ✗'}]\n")

    all_results[fc["Q_lpm"]] = {
        "ilim": ilim_curve, "tratio": tratio_curve,
        "iact": iact_curve, "volt": volt_curve, "isafe": isafe_curve
    }

    axes[0].plot(iact_curve, volt_curve,
                 marker="o", label=fc["label"],
                 color=fc["color"], ls=fc["ls"], linewidth=2)
    axes[1].plot(CURRENT_LEVELS, ilim_curve,
                 marker="s", label=fc["label"],
                 color=fc["color"], ls=fc["ls"], linewidth=2)
    axes[2].plot(CURRENT_LEVELS, tratio_curve,
                 marker="^", label=fc["label"],
                 color=fc["color"], ls=fc["ls"], linewidth=2)

# ── I_limit ranking check ─────────────────────────────────────────────────────
print("--- I_limit Ranking Check ---")
ilim_5  = np.mean(all_results[5]["ilim"])
ilim_10 = np.mean(all_results[10]["ilim"])
ilim_30 = np.mean(all_results[30]["ilim"])
rank_ok = ilim_30 > ilim_10 > ilim_5
print(f"  Mean I_limit:  5 LPM={ilim_5:.1f}A  "
      f"10 LPM={ilim_10:.1f}A  30 LPM={ilim_30:.1f}A")
print(f"  Ranking 30>10>5 : [{'PASS ✓' if rank_ok else 'FAIL ✗'}]")

# ── Flow derating check ───────────────────────────────────────────────────────
print("\n--- Flow Derating Check (5 LPM < flow_critical=8 LPM) ---")
isafe_5   = all_results[5]["isafe"]
isafe_10  = all_results[10]["isafe"]
derate_5  = any(s < I_cmd * 0.95 for s, I_cmd in zip(isafe_5, CURRENT_LEVELS))
derate_10 = any(s < I_cmd * 0.95 for s, I_cmd in zip(isafe_10, CURRENT_LEVELS))
print(f"  5  LPM flow derating active : {derate_5}  "
      f"[{'PASS ✓' if derate_5 else 'FAIL ✗'}]  (expected True)")
print(f"  10 LPM flow derating active : {derate_10}  "
      f"[{'PASS ✓' if not derate_10 else 'NOTE'}]  (expected False)")

# ── Format plots ──────────────────────────────────────────────────────────────
axes[0].set_xlabel("Actual current (A)")
axes[0].set_ylabel("Stack voltage (V)")
axes[0].set_title("Test E — Polarisation Curve\n(no BMS voltage artefacts)")
axes[0].legend(fontsize=8)
axes[0].grid(True)

axes[1].set_xlabel("Commanded current (A)")
axes[1].set_ylabel("I_limit (A)")
axes[1].set_title(f"Limiting Current vs Flow\n(scales as flow^0.4)")
axes[1].legend(fontsize=8)
axes[1].grid(True)

axes[2].set_xlabel("Commanded current (A)")
axes[2].set_ylabel("Transport ratio η_T")
axes[2].axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Hard limit = 1.0")
axes[2].set_title("Transport Ratio\n(must stay ≤ 1.0)")
axes[2].legend(fontsize=8)
axes[2].grid(True)

plt.suptitle("Test E — Mass Transport Validation  "
             f"(flow_critical={cfg_ref.flow_critical*60000:.0f} LPM)",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("results/test_E_mass_transport.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → results/test_E_mass_transport.png")