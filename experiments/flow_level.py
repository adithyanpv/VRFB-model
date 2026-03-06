# experiments/test_flow_level_nonlinearity.py

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.coulomb_counter import CoulombCounter
from vrfb.sensor_model import SensorModel
from vrfb.bms_controller import BMSController
from vrfb.utils import compute_metrics

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# INITIALIZATION
# ============================================================

cfg = VRFBConfig()
battery = VRFB(cfg)
bms = BMSController(cfg)
cc = CoulombCounter(cfg)
sensor = SensorModel(cfg)

cc.initialize(cfg.initial_soc)

dt = cfg.dt_default
cycles_target = 5
rest_time = 300  # shorter rest for faster test


# ------------------------------------------------------------
# FLOW LEVELS PER CYCLE (LPM → m³/s)
# ------------------------------------------------------------

flow_LPM_levels = [10, 20, 40, 15, 30]

flow_levels = [f * cfg.LPM_to_m3s for f in flow_LPM_levels]


# ============================================================
# LOGGING
# ============================================================

time_log = []
soc_true = []
soc_est = []
voltage = []
current = []
flow_used = []

t = 0
completed_cycles = 0


# ============================================================
# STATE MACHINE
# ============================================================

STATE_CHARGE = 0
STATE_REST_AFTER_CHARGE = 1
STATE_DISCHARGE = 2
STATE_REST_AFTER_DISCHARGE = 3

state = STATE_CHARGE
rest_counter = 0


while completed_cycles < cycles_target:

    out = battery.get_outputs()
    current_flow = flow_levels[completed_cycles]

    # ========================================================
    # CHARGE
    # ========================================================
    if state == STATE_CHARGE:

        I_cmd = -150
        I_safe = bms.apply_protection(I_cmd, out)

        battery.step(I_safe, current_flow, 298.15, dt)

        if out["soc_true"] >= cfg.soc_max - 1e-4:
            state = STATE_REST_AFTER_CHARGE
            rest_counter = 0
            print(f"Cycle {completed_cycles+1} (Flow {flow_LPM_levels[completed_cycles]} LPM): Charge Complete")

    # ========================================================
    # REST AFTER CHARGE
    # ========================================================
    elif state == STATE_REST_AFTER_CHARGE:

        battery.step(0, current_flow, 298.15, dt)
        rest_counter += dt

        if rest_counter >= rest_time:
            state = STATE_DISCHARGE

    # ========================================================
    # DISCHARGE
    # ========================================================
    elif state == STATE_DISCHARGE:

        I_cmd = 150
        I_safe = bms.apply_protection(I_cmd, out)

        battery.step(I_safe, current_flow, 298.15, dt)

        if out["soc_true"] <= cfg.soc_min + 1e-4:
            state = STATE_REST_AFTER_DISCHARGE
            rest_counter = 0
            print(f"Cycle {completed_cycles+1}: Discharge Complete")

    # ========================================================
    # REST AFTER DISCHARGE
    # ========================================================
    elif state == STATE_REST_AFTER_DISCHARGE:

        battery.step(0, current_flow, 298.15, dt)
        rest_counter += dt

        if rest_counter >= rest_time:
            completed_cycles += 1
            state = STATE_CHARGE
            print(f"Completed Cycle {completed_cycles}\n")

    # ========================================================
    # MEASURE & LOG
    # ========================================================

    out = battery.get_outputs()
    measured = sensor.measure(out)

    soc_cc = cc.update(
        measured_current=measured["current"],
        dt=dt,
        Q_nominal=out["capacity_nominal"]
    )

    time_log.append(t)
    soc_true.append(out["soc_true"])
    soc_est.append(soc_cc)
    voltage.append(out["voltage_stack"])
    current.append(out["current"])
    flow_used.append(current_flow)

    t += dt


# ============================================================
# PLOT RESULTS
# ============================================================

time_log = np.array(time_log)
soc_true = np.array(soc_true)
voltage = np.array(voltage)
flow_used = np.array(flow_used)


# 1️⃣ SOC
plt.figure()
plt.plot(time_log, soc_true)
plt.title("SOC vs Time (Variable Flow)")
plt.xlabel("Time (s)")
plt.ylabel("SOC")
plt.grid()

# 2️⃣ Voltage
plt.figure()
plt.plot(time_log, voltage)
plt.title("Voltage vs Time (Variable Flow)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid()

# 3️⃣ Voltage vs SOC (NONLINEARITY VIEW)
plt.figure()
plt.scatter(soc_true, voltage, c=flow_used, s=2)
plt.colorbar(label="Flow (m³/s)")
plt.title("Voltage vs SOC (Color = Flow Level)")
plt.xlabel("SOC")
plt.ylabel("Voltage (V)")
plt.grid()

plt.show()