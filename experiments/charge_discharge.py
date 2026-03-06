# experiments/test_charge_discharge_cycle.py

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.coulomb_counter import CoulombCounter
from vrfb.sensor_model import SensorModel
from vrfb.bms_controller import BMSController
from vrfb.utils import compute_metrics
import numpy as np
import matplotlib.pyplot as plt


cfg = VRFBConfig()
battery = VRFB(cfg)
bms = BMSController(cfg)
cc = CoulombCounter(cfg)
sensor = SensorModel(cfg)

cc.initialize(cfg.initial_soc)

dt = cfg.dt_default
cycles_target = 5
rest_time = 600  # seconds

time_log = []
soc_true = []
soc_est = []
voltage = []
current = []
temperature = []

t = 0
completed_cycles = 0

# ============================================================
# STATE MACHINE DEFINITIONS
# ============================================================

STATE_CHARGE = 0
STATE_REST_AFTER_CHARGE = 1
STATE_DISCHARGE = 2
STATE_REST_AFTER_DISCHARGE = 3

state = STATE_CHARGE
rest_counter = 0


while completed_cycles < cycles_target:

    out = battery.get_outputs()

    # ========================================================
    # CHARGE STATE
    # ========================================================
    if state == STATE_CHARGE:

        I_cmd = -150
        I_safe = bms.apply_protection(I_cmd, out)

        battery.step(I_safe, cfg.initial_flow, 298.15, dt)

        # Transition when SOC reaches upper limit
        if out["soc_true"] >= cfg.soc_max - 1e-4:
            state = STATE_REST_AFTER_CHARGE
            rest_counter = 0
            print(f"Cycle {completed_cycles+1}: Charge complete")

    # ========================================================
    # REST AFTER CHARGE
    # ========================================================
    elif state == STATE_REST_AFTER_CHARGE:

        battery.step(0, cfg.initial_flow, 298.15, dt)
        rest_counter += dt

        if rest_counter >= rest_time:
            state = STATE_DISCHARGE

    # ========================================================
    # DISCHARGE STATE
    # ========================================================
    elif state == STATE_DISCHARGE:

        I_cmd = 150
        I_safe = bms.apply_protection(I_cmd, out)

        battery.step(I_safe, cfg.initial_flow, 298.15, dt)

        # Transition when SOC reaches lower limit
        if out["soc_true"] <= cfg.soc_min + 1e-4:
            state = STATE_REST_AFTER_DISCHARGE
            rest_counter = 0
            print(f"Cycle {completed_cycles+1}: Discharge complete")

    # ========================================================
    # REST AFTER DISCHARGE
    # ========================================================
    elif state == STATE_REST_AFTER_DISCHARGE:

        battery.step(0, cfg.initial_flow, 298.15, dt)
        rest_counter += dt

        if rest_counter >= rest_time:
            completed_cycles += 1
            state = STATE_CHARGE
            print(f"Completed Cycle {completed_cycles}")

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
    temperature.append(out["temperature_stack"])

    t += dt


# ============================================================
# METRICS
# ============================================================

soc_true = np.array(soc_true)
soc_est = np.array(soc_est)

rmse, mae, max_e, final_e = compute_metrics(soc_true, soc_est)

print("\n=== PROFESSIONAL BMS CYCLING RESULTS ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Max Error: {max_e:.6f}")
print(f"Final Drift: {final_e:.6f}")


# ============================================================
# PLOTS
# ============================================================

time_log = np.array(time_log)

plt.figure()
plt.plot(time_log, soc_true, label="True SOC")
plt.plot(time_log, soc_est, label="Estimated SOC")
plt.xlabel("Time (s)")
plt.ylabel("SOC")
plt.title("SOC Comparison - State Machine Controlled Cycling")
plt.legend(loc="upper right")
plt.grid()

plt.figure()
plt.plot(time_log, voltage)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Stack Voltage")
plt.grid()

plt.figure()
plt.plot(time_log, current)
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("Current Profile")
plt.grid()

plt.show()