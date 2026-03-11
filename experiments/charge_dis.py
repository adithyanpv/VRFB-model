# experiments/test_charge_discharge_cycles.py
"""
Charge / Discharge Cycle Test
================================
Runs N complete charge-discharge cycles and tracks:
  - Voltage profile during each phase
  - True SOC vs CC SOC per cycle
  - CC drift accumulation over cycles
  - Capacity delivered per cycle (Ah)
  - Round-trip efficiency per cycle

This test directly supports the thesis argument:
  "CC error accumulates over repeated cycles due to coulombic
   inefficiency, crossover, and capacity fade. Each cycle adds
   a small systematic error that compounds — the REN corrects
   this by observing voltage directly."

WHAT TO EXPECT:
  - During charging: CC SOC > True SOC (CC integrates full current,
    true SOC rises at η_coulombic=0.98 rate)
  - During discharging: CC SOC tracks closely but with offset
  - Offset grows cycle by cycle → diverging lines
  - Round-trip efficiency ≈ 95–97% (coulombic × voltage efficiency)

PASS criteria:
  - Voltage rises during charge, falls during discharge (correct shape)
  - CC SOC diverges from true SOC across cycles
  - Round-trip efficiency between 93–99%
  - Capacity delivered per cycle slowly declining (capacity fade)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vrfb.config         import VRFBConfig
from vrfb.vrfb_core      import VRFB
from vrfb.bms_controller import BMSController
from vrfb.sensor_model   import SensorModel
from vrfb.coulomb_counter import CoulombCounter

os.makedirs("results", exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
N_CYCLES  = 5
I_CHARGE  = -120.0   # A  (negative = charging)
I_DISCH   =  120.0   # A  (positive = discharging)
T_AMB     = 298.15   # K
REST_TIME = 300       # s  rest between charge and discharge

cfg             = VRFBConfig()
cfg.initial_soc = 0.05   # start at soc_min for full swing every cycle
battery = VRFB(cfg)
bms     = BMSController(cfg)
sensor  = SensorModel(cfg)
cc      = CoulombCounter(cfg)
cc.initialize(cfg.initial_soc)

dt    = cfg.dt_default
Q_cmd = cfg.initial_flow

print(f"Charge/Discharge Cycle Test")
print(f"  N_cycles   = {N_CYCLES}")
print(f"  I_charge   = {I_CHARGE} A")
print(f"  I_discharge= {I_DISCH} A")
print(f"  SOC window = {cfg.soc_min} → {cfg.soc_max}")
print(f"  Rest time  = {REST_TIME} s between phases")
print("-" * 60)

# ── Logging ───────────────────────────────────────────────────────────────────
t_log        = []
soc_true_log = []
soc_cc_log   = []
v_log        = []
i_log        = []
temp_log     = []
phase_log    = []   # 0=charge, 1=rest, 2=discharge

# Per-cycle summary
cycle_Q_ch   = []   # Ah charged
cycle_Q_dch  = []   # Ah discharged
cycle_eta_rt = []   # round-trip efficiency
cycle_v_avg_ch  = []
cycle_v_avg_dch = []
cycle_cc_drift  = []   # |SOC_true - SOC_cc| at end of discharge

t = 0.0

for cycle in range(1, N_CYCLES + 1):

    print(f"  Cycle {cycle}/{N_CYCLES} — Charging...", end="", flush=True)

    # ── Phase 1: Charge ───────────────────────────────────────────────────────
    Q_ch_Ah = 0.0
    v_ch    = []

    while True:
        out    = battery.get_outputs()
        I_safe = bms.apply_protection(I_CHARGE, out)
        if abs(I_safe) < 0.1:
            break
        battery.step(I_safe, Q_cmd, T_AMB, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        soc_cc   = cc.update(measured_current=measured["current"],
                             dt=dt, Q_nominal=out["capacity_nominal"])
        Q_ch_Ah += abs(out["current"]) * dt / 3600.0
        v_ch.append(out["voltage_stack"])

        t_log.append(t);        t += dt
        soc_true_log.append(out["soc_true"])
        soc_cc_log.append(soc_cc)
        v_log.append(out["voltage_stack"])
        i_log.append(out["current"])
        temp_log.append(out["temperature_stack"])
        phase_log.append(0)

    print(f" done ({Q_ch_Ah:.1f} Ah)  Resting...", end="", flush=True)

    # ── Phase 2: Rest after charge ────────────────────────────────────────────
    rest_steps = int(REST_TIME / dt)
    for _ in range(rest_steps):
        battery.step(0.0, Q_cmd, T_AMB, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        soc_cc   = cc.update(measured_current=measured["current"],
                             dt=dt, Q_nominal=out["capacity_nominal"])
        t_log.append(t);        t += dt
        soc_true_log.append(out["soc_true"])
        soc_cc_log.append(soc_cc)
        v_log.append(out["voltage_stack"])
        i_log.append(0.0)
        temp_log.append(out["temperature_stack"])
        phase_log.append(1)

    print(f" Discharging...", end="", flush=True)

    # ── Phase 3: Discharge ────────────────────────────────────────────────────
    Q_dch_Ah = 0.0
    v_dch    = []

    while True:
        out    = battery.get_outputs()
        I_safe = bms.apply_protection(I_DISCH, out)
        if abs(I_safe) < 0.1:
            break
        battery.step(I_safe, Q_cmd, T_AMB, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        soc_cc   = cc.update(measured_current=measured["current"],
                             dt=dt, Q_nominal=out["capacity_nominal"])
        Q_dch_Ah += abs(out["current"]) * dt / 3600.0
        v_dch.append(out["voltage_stack"])

        t_log.append(t);        t += dt
        soc_true_log.append(out["soc_true"])
        soc_cc_log.append(soc_cc)
        v_log.append(out["voltage_stack"])
        i_log.append(out["current"])
        temp_log.append(out["temperature_stack"])
        phase_log.append(2)

    # ── Phase 4: Rest after discharge ─────────────────────────────────────────
    for _ in range(rest_steps):
        battery.step(0.0, Q_cmd, T_AMB, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)
        soc_cc   = cc.update(measured_current=measured["current"],
                             dt=dt, Q_nominal=out["capacity_nominal"])
        t_log.append(t);        t += dt
        soc_true_log.append(out["soc_true"])
        soc_cc_log.append(soc_cc)
        v_log.append(out["voltage_stack"])
        i_log.append(0.0)
        temp_log.append(out["temperature_stack"])
        phase_log.append(3)

    # ── Per-cycle metrics ─────────────────────────────────────────────────────
    eta_rt  = Q_dch_Ah / Q_ch_Ah if Q_ch_Ah > 0 else 0
    drift   = abs(out["soc_true"] - cc.get_soc())
    v_avg_c = np.mean(v_ch)   if v_ch  else 0
    v_avg_d = np.mean(v_dch)  if v_dch else 0

    cycle_Q_ch.append(Q_ch_Ah)
    cycle_Q_dch.append(Q_dch_Ah)
    cycle_eta_rt.append(eta_rt)
    cycle_v_avg_ch.append(v_avg_c)
    cycle_v_avg_dch.append(v_avg_d)
    cycle_cc_drift.append(drift)

    print(f" done ({Q_dch_Ah:.1f} Ah)")
    print(f"    Q_ch={Q_ch_Ah:.2f} Ah  Q_dch={Q_dch_Ah:.2f} Ah  "
          f"η_RT={eta_rt*100:.2f}%  CC_drift={drift:.5f}")


# ── Arrays ────────────────────────────────────────────────────────────────────
t_arr        = np.array(t_log)
soc_true_arr = np.array(soc_true_log)
soc_cc_arr   = np.array(soc_cc_log)
v_arr        = np.array(v_log)
i_arr        = np.array(i_log)
temp_arr     = np.array(temp_log)
phase_arr    = np.array(phase_log)
cycles_x     = np.arange(1, N_CYCLES + 1)

rmse = np.sqrt(np.mean((soc_true_arr - soc_cc_arr)**2))
mae  = np.mean(np.abs(soc_true_arr - soc_cc_arr))

print(f"\n=== SUMMARY ===")
print(f"  Overall CC RMSE = {rmse:.5f}  MAE = {mae:.5f}")
print(f"  Mean round-trip η = {np.mean(cycle_eta_rt)*100:.2f}%")
print(f"  Capacity cycle 1 → {N_CYCLES}: "
      f"{cycle_Q_dch[0]:.2f} → {cycle_Q_dch[-1]:.2f} Ah  "
      f"(fade = {(cycle_Q_dch[0]-cycle_Q_dch[-1])/cycle_Q_dch[0]*100:.3f}%)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax_soc  = fig.add_subplot(gs[0, :2])   # wide — SOC over time
ax_v    = fig.add_subplot(gs[1, :2])   # wide — voltage over time
ax_i    = fig.add_subplot(gs[2, :2])   # wide — current + temp

ax_q    = fig.add_subplot(gs[0, 2])    # capacity per cycle
ax_eta  = fig.add_subplot(gs[1, 2])    # round-trip efficiency
ax_drift= fig.add_subplot(gs[2, 2])    # CC drift per cycle

# Shade phases
phase_colors = {0: "#d4edff", 1: "#f0f0f0", 2: "#ffe4d4", 3: "#f0f0f0"}
phase_labels = {0: "Charge", 2: "Discharge"}
labeled = set()
for ax in [ax_soc, ax_v, ax_i]:
    prev_p = phase_arr[0]
    seg_start = t_arr[0]
    for idx in range(1, len(phase_arr)):
        if phase_arr[idx] != prev_p or idx == len(phase_arr) - 1:
            lbl = phase_labels.get(prev_p, None)
            ax.axvspan(seg_start, t_arr[idx],
                       color=phase_colors[prev_p], alpha=0.6,
                       label=lbl if (lbl and lbl not in labeled) else "")
            if lbl:
                labeled.add(lbl)
            seg_start = t_arr[idx]
            prev_p = phase_arr[idx]

# SOC
ax_soc.plot(t_arr, soc_true_arr, color="steelblue", lw=2,   label="True SOC")
ax_soc.plot(t_arr, soc_cc_arr,   color="tomato",    lw=1.5,
            ls="--", label="CC SOC")
ax_soc.set_ylabel("SOC")
ax_soc.set_title(f"SOC — True vs Coulomb Counter  (RMSE={rmse:.4f})")
ax_soc.legend(fontsize=8); ax_soc.grid(True)

# Voltage
ax_v.plot(t_arr, v_arr, color="darkorange", lw=1.5)
ax_v.axhline(cfg.V_max_stack, color="red",    ls=":", lw=1,
             label=f"V_max={cfg.V_max_stack}V")
ax_v.axhline(cfg.V_min_stack, color="navy",   ls=":", lw=1,
             label=f"V_min={cfg.V_min_stack}V")
ax_v.set_ylabel("Stack Voltage (V)")
ax_v.set_title("Stack Voltage — rises during charge, falls during discharge")
ax_v.legend(fontsize=8); ax_v.grid(True)

# Current
ax_i2 = ax_i.twinx()
ax_i.plot(t_arr, i_arr,    color="steelblue", lw=1.5, label="Current (A)")
ax_i2.plot(t_arr, temp_arr, color="tomato",   lw=1,   ls="--",
           label="Temperature (K)", alpha=0.7)
ax_i.set_ylabel("Current (A)");  ax_i.set_xlabel("Time (s)")
ax_i2.set_ylabel("Temperature (K)", color="tomato")
ax_i.set_title("Current & Temperature")
ax_i.legend(loc="upper left", fontsize=8)
ax_i2.legend(loc="upper right", fontsize=8)
ax_i.grid(True)

# Capacity per cycle
ax_q.bar(cycles_x - 0.2, cycle_Q_ch,  0.35, label="Charged",    color="steelblue")
ax_q.bar(cycles_x + 0.2, cycle_Q_dch, 0.35, label="Discharged", color="tomato")
ax_q.set_xlabel("Cycle"); ax_q.set_ylabel("Capacity (Ah)")
ax_q.set_title("Capacity per Cycle")
ax_q.legend(fontsize=8); ax_q.grid(True, axis="y")
ax_q.set_xticks(cycles_x)

# Round-trip efficiency
ax_eta.plot(cycles_x, np.array(cycle_eta_rt) * 100,
            marker="o", color="green", lw=2, ms=8)
ax_eta.set_ylim(90, 102)
ax_eta.axhline(100, color="gray", ls="--", lw=1)
ax_eta.set_xlabel("Cycle"); ax_eta.set_ylabel("η_RT (%)")
ax_eta.set_title("Round-Trip Efficiency")
ax_eta.grid(True); ax_eta.set_xticks(cycles_x)

# CC drift per cycle
ax_drift.plot(cycles_x, np.array(cycle_cc_drift) * 100,
              marker="^", color="purple", lw=2, ms=8)
ax_drift.set_xlabel("Cycle"); ax_drift.set_ylabel("CC Drift (%SOC)")
ax_drift.set_title("CC Drift at End of Each Cycle\n(grows = CC cannot self-correct)")
ax_drift.grid(True); ax_drift.set_xticks(cycles_x)

plt.suptitle(
    f"Charge / Discharge Cycle Test  |  {N_CYCLES} cycles  |  "
    f"I={abs(I_CHARGE)}A  |  T={T_AMB}K\n"
    f"CC RMSE={rmse:.4f}  Mean η_RT={np.mean(cycle_eta_rt)*100:.2f}%",
    fontsize=11
)
plt.savefig("results/test_charge_discharge_cycles.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → results/test_charge_discharge_cycles.png")