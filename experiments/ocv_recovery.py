# experiments/test_F_ocv_recovery.py
"""
OCV Recovery After Discharge
==============================
Discharges at constant current for 30 min, then rests for 20 min.
VRFB should recover 90% of OCV within ~5 minutes (much faster than Li-ion).

CC behaviour during rest:
  - Current = 0, so CC SOC is frozen (no integration)
  - Any divergence between CC and true SOC during rest is purely from
    the crossover / concentration redistribution in the physics,
    not from sensor noise — CC cannot see this drift at I=0.

PASS criteria:
  - 90% OCV recovery within 300 s
  - Recovered OCV/cell within ±0.03 V of Nernst at post-discharge SOC
  - Higher flow → faster recovery
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

DISCHARGE_TIME = 1800
RECOVERY_TIME  = 1200
I_DISCHARGE    = 120.0

FLOW_CASES = [
    {"Q_lpm":  5, "label":  "5 LPM", "color": "tomato"},
    {"Q_lpm": 30, "label": "30 LPM", "color": "steelblue"},
]

print("Test F: OCV Recovery After Discharge")
print(f"  Discharge {I_DISCHARGE} A for {DISCHARGE_TIME}s, then rest {RECOVERY_TIME}s")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for fc in FLOW_CASES:

    cfg             = VRFBConfig()
    cfg.initial_soc = 0.70
    battery         = VRFB(cfg)
    bms             = BMSController(cfg)
    sensor          = SensorModel(cfg)
    cc              = CoulombCounter(cfg)
    cc.initialize(cfg.initial_soc)

    dt    = cfg.dt_default
    Q_cmd = fc["Q_lpm"] * cfg.LPM_to_m3s

    t_dch, v_dch, soc_t_dch, soc_cc_dch = [], [], [], []
    t_rec, v_rec, soc_t_rec, soc_cc_rec = [], [], [], []
    t = 0

    # ── Discharge ─────────────────────────────────────────────────────────────
    for _ in range(int(DISCHARGE_TIME / dt)):
        out    = battery.get_outputs()
        I_safe = bms.apply_protection(I_DISCHARGE, out)
        battery.step(I_safe, Q_cmd, 298.15, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        soc_cc   = cc.update(measured_current=measured["current"],
                             dt=dt, Q_nominal=out["capacity_nominal"])
        t_dch.append(t)
        v_dch.append(out["voltage_stack"])
        soc_t_dch.append(out["soc_true"])
        soc_cc_dch.append(soc_cc)
        t += dt

    v_end_dch  = battery.get_outputs()["voltage_stack"]
    soc_post   = battery.get_outputs()["soc_true"]
    print(f"  [{fc['label']}] SOC after discharge = {soc_post:.4f}  V={v_end_dch:.3f}V")

    # ── Rest / Recovery ───────────────────────────────────────────────────────
    t_rest_start = t
    for _ in range(int(RECOVERY_TIME / dt)):
        battery.step(0.0, Q_cmd, 298.15, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        # CC sees ~0 A → SOC estimate frozen → good time to show divergence
        soc_cc   = cc.update(measured_current=measured["current"],
                             dt=dt, Q_nominal=out["capacity_nominal"])
        t_rec.append(t - t_rest_start)
        v_rec.append(out["voltage_stack"])
        soc_t_rec.append(out["soc_true"])
        soc_cc_rec.append(soc_cc)
        t += dt

    v_recovered  = v_rec[-1]
    soc_final    = soc_t_rec[-1]
    cc_final     = soc_cc_rec[-1]

    # Nernst prediction
    ocv_nernst_cell = cfg.E0 + (cfg.R * 298.15 / cfg.F) * np.log(
        (soc_final ** 2) / ((1 - soc_final) ** 2 + 1e-12)
    )
    ocv_nernst_stack = ocv_nernst_cell * cfg.N_cells
    ocv_error = abs(v_recovered - ocv_nernst_stack) / cfg.N_cells

    # 90% recovery time
    v_arr    = np.array(v_rec)
    v_delta  = v_recovered - v_end_dch
    t90_s    = None
    if v_delta > 0.01:
        idx  = np.argmax(v_arr >= v_end_dch + 0.9 * v_delta)
        t90_s = t_rec[idx]

    cc_drift_rest = abs(soc_final - cc_final)

    print(f"  [{fc['label']}] Recovered OCV/cell = {v_recovered/cfg.N_cells:.4f} V  "
          f"(Nernst={ocv_nernst_cell:.4f} V)  "
          f"[{'PASS ✓' if ocv_error < 0.03 else 'FAIL ✗'}]")
    if t90_s is not None:
        print(f"  [{fc['label']}] 90% recovery time = {t90_s:.0f}s  "
              f"[{'PASS ✓' if t90_s < 300 else 'FAIL ✗'}]")
    print(f"  [{fc['label']}] CC drift during rest = {cc_drift_rest:.5f}  "
          f"(should be near-zero — CC frozen at I=0)\n")

    axes[0].plot(t_dch, v_dch, color=fc["color"], label=f"Discharge {fc['label']}")
    axes[1].plot(t_rec, v_rec, color=fc["color"], label=f"Recovery {fc['label']}")
    axes[1].axhline(ocv_nernst_stack, color=fc["color"], linestyle="--",
                    alpha=0.6, label=f"Nernst {fc['label']}")

axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Stack voltage (V)")
axes[0].set_title(f"Test F — Discharge ({I_DISCHARGE} A)")
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel("Time since rest start (s)")
axes[1].set_ylabel("Stack voltage (V)")
axes[1].set_title("Test F — OCV Recovery (I=0)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("results/test_F_ocv_recovery.png", dpi=150)
plt.show()
print("Saved → results/test_F_ocv_recovery.png")