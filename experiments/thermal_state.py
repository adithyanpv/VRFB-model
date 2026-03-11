# experiments/test_D_thermal_steady_state.py
"""
Thermal Steady-State Validation
=================================
Validates the thermal model by comparing simulated steady-state stack
temperature against the analytical prediction:

  T_tank_ss  = T_amb  + P / h_tank_ambient
  T_stack_ss = T_tank + P / h_stack_tank
  where P = I² × R_stack

SYSTEM THERMAL PARAMETERS (corrected for 97 kWh / 5–10 kW class):
  h_stack_tank   = 200 W/K  (electrolyte pump, 20 LPM forced convection)
  h_tank_ambient = 150 W/K  (forced-air / water heat exchanger on 100 L tank)
  C_th_tank      = 227,500 J/K  (130 kg electrolyte × 3500 J/kg/K)
  τ_thermal      = C_th_tank / h_tank_ambient = 227500/150 ≈ 1517 s (25 min)

CURRENT SELECTION:
  Validation cases (analytical formula valid — T_ss well below T_derate):
    80 A  → T_stack_ss ≈ 304 K (31°C)  — 16 K margin to T_derate
   120 A  → T_stack_ss ≈ 312 K (39°C)  —  8 K margin to T_derate

  BMS derating demo (intentional — proves BMS thermal protection works):
   200 A  → T_stack_ss ≈ 335 K (62°C)  — exceeds T_max, BMS shuts down

  WHY NOT 150 A for validation:
    At 150 A, T_stack_ss ≈ 319 K — only 1 K below T_derate = 320 K.
    Any transient or noise could trigger derating, invalidating the
    analytical comparison. Use 120 A as the high-current validation case.

PASS criteria:
  - Simulated T_stack_ss within ±5 K of analytical (80 A and 120 A)
  - BMS derating fires at 200 A when T ≥ T_derate = 320 K
  - CC RMSE < 0.03 for both validation cases
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


def analytical_T_ss(I, cfg, T_amb=298.15):
    R_stack = (cfg.R_membrane_initial + cfg.R_contact) * cfg.N_cells
    P       = I ** 2 * R_stack
    T_t_ss  = T_amb + P / cfg.h_tank_ambient
    T_s_ss  = T_t_ss + P / cfg.h_stack_tank
    return T_s_ss, T_t_ss, P


# ── Validate config thermal parameters before running ──────────────────────
_cfg_check = VRFBConfig()
assert _cfg_check.h_stack_tank   == 200.0, \
    f"h_stack_tank={_cfg_check.h_stack_tank} — update config.py to 200.0 W/K"
assert _cfg_check.h_tank_ambient == 150.0, \
    f"h_tank_ambient={_cfg_check.h_tank_ambient} — update config.py to 150.0 W/K"

# τ = C_th_tank / h_tank_ambient ≈ 1517 s
# Run 8τ ≈ 12136 s — sufficient for thermal convergence (>99.97%) while
# keeping SOC well above the mass-transport derating threshold.
#
# WHY NOT 20τ:
#   At 120A for 20τ=30340s: ΔSOC = 120×30340/3600/2144 = 0.47
#   Starting SOC=0.70 → final SOC≈0.23 → I_limit drops to ~110A
#   → BMS derates current → less heat → T_ss falls 5.7K below analytical
#   That is correct BMS behaviour but invalidates the thermal SS comparison.
#
# WHY initial_soc=0.90:
#   At 120A for 8τ=12136s: ΔSOC = 120×12136/3600/2144 = 0.19
#   Starting SOC=0.90 → final SOC≈0.71 → I_limit≈329A → margin=230A >> 120A
#   → No BMS derating → analytical formula remains valid throughout
TAU   = _cfg_check.C_th_tank / _cfg_check.h_tank_ambient
STEPS = int(8 * TAU)    # ≈ 12136 steps (8τ is sufficient, avoids SOC depletion)

VALIDATION_CASES = [
    {"I": 80.0,  "label":  "80 A"},
    {"I": 120.0, "label": "120 A"},
]

print("Test D: Thermal Steady State")
print(f"  h_stack_tank   = {_cfg_check.h_stack_tank} W/K")
print(f"  h_tank_ambient = {_cfg_check.h_tank_ambient} W/K")
print(f"  C_th_tank      = {_cfg_check.C_th_tank:.0f} J/K")
print(f"  τ_thermal      = {TAU:.0f} s  ({TAU/60:.0f} min)")
print(f"  Simulation     = {STEPS} steps = {STEPS/TAU:.0f}τ")
print(f"  T_derate       = {_cfg_check.T_derate} K  |  T_max = {_cfg_check.T_max} K")
print("-" * 65)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for row, case in enumerate(VALIDATION_CASES):

    I_cmd = case["I"]
    T_amb = 298.15
    cfg   = VRFBConfig()

    cfg.initial_soc = 0.90   # start high — 8τ at 120A only depletes 0.19 SOC
                              # keeps I_limit >> I_cmd throughout → no transport derating
    battery = VRFB(cfg)
    bms     = BMSController(cfg)
    sensor  = SensorModel(cfg)
    cc      = CoulombCounter(cfg)
    cc.initialize(cfg.initial_soc)

    T_s_ana, T_t_ana, P_joule = analytical_T_ss(I_cmd, cfg, T_amb)

    t_log        = []
    Ts_log       = []
    Tt_log       = []
    soc_true_log = []
    soc_cc_log   = []

    for step in range(STEPS):
        t      = step * cfg.dt_default
        out    = battery.get_outputs()
        I_safe = bms.apply_protection(float(I_cmd), out)
        battery.step(I_safe, cfg.initial_flow, T_amb, cfg.dt_default)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        soc_cc   = cc.update(measured_current=measured["current"],
                             dt=cfg.dt_default,
                             Q_nominal=out["capacity_nominal"])

        t_log.append(t)
        Ts_log.append(out["temperature_stack"])
        Tt_log.append(out["temperature_tank"])
        soc_true_log.append(out["soc_true"])
        soc_cc_log.append(soc_cc)

        if step % 5000 == 0 and step > 0:
            print(f"  [{case['label']}] t={t:.0f}s ({t/TAU:.1f}τ)  "
                  f"T_stack={out['temperature_stack']:.2f}K  "
                  f"I_safe={I_safe:.1f}A  SOC={out['soc_true']:.4f}")

    T_s_sim = Ts_log[-1]
    T_t_sim = Tt_log[-1]
    err     = abs(T_s_sim - T_s_ana)
    cc_rmse = np.sqrt(np.mean(
        (np.array(soc_true_log) - np.array(soc_cc_log)) ** 2
    ))

    print(f"\n  [{case['label']}]  P_joule = {P_joule:.0f} W")
    print(f"    T_stack analytical = {T_s_ana:.2f} K  ({T_s_ana-273.15:.1f}°C)")
    print(f"    T_stack simulated  = {T_s_sim:.2f} K  ({T_s_sim-273.15:.1f}°C)")
    print(f"    T_tank  simulated  = {T_t_sim:.2f} K  (analytical={T_t_ana:.2f}K)")
    print(f"    Error              = {err:.2f} K  "
          f"[{'PASS ✓' if err < 5 else 'FAIL ✗'}]")
    print(f"    CC RMSE            = {cc_rmse:.5f}  "
          f"[{'PASS ✓' if cc_rmse < 0.03 else 'REVIEW'}]\n")

    t_arr  = np.array(t_log)
    Ts_arr = np.array(Ts_log)
    Tt_arr = np.array(Tt_log)

    # ── Temperature plot ──────────────────────────────────────────────────
    axes[row, 0].plot(t_arr, Ts_arr, color="tomato",    label="T_stack sim", linewidth=2)
    axes[row, 0].plot(t_arr, Tt_arr, color="steelblue", label="T_tank sim",  linewidth=2)
    axes[row, 0].axhline(T_s_ana, color="tomato",    linestyle="--",
                         label=f"Analytical T_stack={T_s_ana:.1f}K", alpha=0.7)
    axes[row, 0].axhline(T_t_ana, color="steelblue", linestyle="--",
                         label=f"Analytical T_tank={T_t_ana:.1f}K",  alpha=0.5)
    axes[row, 0].axhline(cfg.T_derate, color="orange", linestyle=":",
                         label=f"T_derate={cfg.T_derate:.0f}K", linewidth=1.2)
    axes[row, 0].axhline(cfg.T_max,    color="red",    linestyle=":",
                         label=f"T_max={cfg.T_max:.0f}K",       linewidth=1.2)
    axes[row, 0].set_xlabel("Time (s)")
    axes[row, 0].set_ylabel("Temperature (K)")
    axes[row, 0].set_title(f"Test D — Thermal SS [{case['label']}]\n"
                           f"Error = {err:.2f} K  "
                           f"{'PASS ✓' if err < 5 else 'FAIL ✗'}")
    axes[row, 0].legend(fontsize=7)
    axes[row, 0].grid(True)

    # ── CC tracking plot ──────────────────────────────────────────────────
    axes[row, 1].plot(t_arr, np.array(soc_true_log), label="True SOC",
                      color="steelblue", linewidth=1.5)
    axes[row, 1].plot(t_arr, np.array(soc_cc_log),   label="CC SOC",
                      color="tomato", linestyle="--", linewidth=1.5)
    axes[row, 1].set_xlabel("Time (s)")
    axes[row, 1].set_ylabel("SOC")
    axes[row, 1].set_title(f"CC Tracking  (RMSE={cc_rmse:.5f})")
    axes[row, 1].legend()
    axes[row, 1].grid(True)

# ── BMS Derating Demo: 200 A ──────────────────────────────────────────────────
print("  [200 A — BMS derating/shutdown demo] running…")
cfg3             = VRFBConfig()
cfg3.initial_soc = 0.90
bat3  = VRFB(cfg3)
bms3  = BMSController(cfg3)

t3_log, Ts3_log, I3_log = [], [], []
bms3_event_t = None

for step in range(STEPS):
    t      = step * cfg3.dt_default
    out    = bat3.get_outputs()
    I_safe = bms3.apply_protection(200.0, out)

    if bms3_event_t is None and I_safe < 190.0:
        bms3_event_t = t
        print(f"    BMS intervened at t={t:.0f}s  "
              f"T={out['temperature_stack']:.2f}K  I_safe={I_safe:.1f}A")

    bat3.step(I_safe, cfg3.initial_flow, 298.15, cfg3.dt_default)
    out = bat3.get_outputs()
    t3_log.append(t)
    Ts3_log.append(out["temperature_stack"])
    I3_log.append(I_safe)

T3_max = max(Ts3_log)
print(f"    Peak T_stack = {T3_max:.2f}K  |  Final = {Ts3_log[-1]:.2f}K")
print(f"    BMS regulated temperature below T_max = {cfg3.T_max}K  "
      f"[{'PASS ✓' if T3_max < cfg3.T_max + 1 else 'REVIEW'}]\n")

t3_arr  = np.array(t3_log)
Ts3_arr = np.array(Ts3_log)
I3_arr  = np.array(I3_log)

axes[0, 2].plot(t3_arr, Ts3_arr, color="tomato", linewidth=2, label="T_stack")
axes[0, 2].axhline(cfg3.T_derate, color="orange", linestyle=":",
                   label=f"T_derate={cfg3.T_derate}K", linewidth=1.2)
axes[0, 2].axhline(cfg3.T_max,    color="red",    linestyle=":",
                   label=f"T_max={cfg3.T_max}K",   linewidth=1.2)
if bms3_event_t:
    axes[0, 2].axvline(bms3_event_t, color="gray", linestyle="--", alpha=0.5,
                       label=f"BMS fires @ {bms3_event_t:.0f}s")
axes[0, 2].set_xlabel("Time (s)")
axes[0, 2].set_ylabel("Temperature (K)")
axes[0, 2].set_title("BMS Thermal Protection Demo [200 A]\n"
                      "(temperature clamped by derating)")
axes[0, 2].legend(fontsize=7)
axes[0, 2].grid(True)

axes[1, 2].plot(t3_arr, I3_arr, color="steelblue", linewidth=1.5, label="I_safe")
axes[1, 2].axhline(200, color="gray", linestyle="--", alpha=0.5, label="I_cmd=200A")
axes[1, 2].set_xlabel("Time (s)")
axes[1, 2].set_ylabel("Current after BMS (A)")
axes[1, 2].set_title("BMS Current Derating Response\n(current reduced as T rises)")
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True)

plt.suptitle(
    f"Test D — Thermal Model Validation  "
    f"(h_st={cfg.h_stack_tank} W/K, h_ta={cfg.h_tank_ambient} W/K)",
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig("results/test_D_thermal_steady_state.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → results/test_D_thermal_steady_state.png")