# experiments/test_6_flow_sweep_dynamic.py

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.coulomb_counter import CoulombCounter
from vrfb.sensor_model import SensorModel
from vrfb.bms_controller import BMSController
from vrfb.utils import compute_metrics, save_plots, save_metrics

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Initialize
# ------------------------------------------------------------
cfg = VRFBConfig()
battery = VRFB(cfg)
bms = BMSController(cfg)
cc = CoulombCounter(cfg)
sensor = SensorModel(cfg)

cc.initialize(cfg.initial_soc)

dt = cfg.dt_default
sim_time = 10000
steps = int(sim_time / dt)

time_log = []
true_soc = []
est_soc = []
voltage = []
current = []
temperature = []
flow = []
i_limit_log = []
pump_power = []

# ------------------------------------------------------------
# Simulation Loop
# ------------------------------------------------------------
for step in range(steps):

    t = step * dt

    # -------------------------------
    # Dynamic Step Load (same as Test 5)
    # -------------------------------
    I_cmd = 250

    # -------------------------------
    # Flow Sweep (key part of test)
    # -------------------------------
    if t < 2500:
        Q_cmd = 2.0 * cfg.LPM_to_m3s
    elif t < 5000:
        Q_cmd = 10.0 * cfg.LPM_to_m3s
    elif t < 7500:
        Q_cmd = 30.0 * cfg.LPM_to_m3s
    else:
        Q_cmd = 60.0 * cfg.LPM_to_m3s

    T = 298.15

    # --------------------------------------------------------
    # BMS Protection
    # --------------------------------------------------------
    prev_out = battery.get_outputs()
    I_safe = bms.apply_protection(I_cmd, prev_out)

    # --------------------------------------------------------
    # Run Physics
    # --------------------------------------------------------
    battery.step(I_safe, Q_cmd, T, dt)
    out = battery.get_outputs()

    # --------------------------------------------------------
    # Sensor + Estimator
    # --------------------------------------------------------
    measured = sensor.measure(out)

    soc_cc = cc.update(
        measured_current=measured["current"],
        dt=dt,
        Q_nominal=out["capacity_nominal"]
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    time_log.append(t)
    true_soc.append(out["soc_true"])
    est_soc.append(soc_cc)
    voltage.append(out["voltage_stack"])
    current.append(out["current"])
    temperature.append(out["temperature_stack"])
    flow.append(out["flow_rate"])
    i_limit_log.append(out["i_limit"])
    pump_power.append(out["pump_power"])


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
true_soc = np.array(true_soc)
est_soc = np.array(est_soc)

rmse, mae, max_e, final_e = compute_metrics(true_soc, est_soc)

print("\n=== TEST 6: FLOW SWEEP UNDER DYNAMIC LOAD ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Max Error: {max_e:.6f}")
print(f"Final Drift: {final_e:.6f}")

# ------------------------------------------------------------
# Save Results
# ------------------------------------------------------------
folder = "results/test_6_flow_sweep_dynamic"

save_plots(folder,
           time_log, true_soc, est_soc,
           voltage, current, temperature, flow)

save_metrics(folder, rmse, mae, max_e, final_e)
plt.figure()
plt.plot(time_log, current, label="Actual Current")
plt.plot(time_log, i_limit_log, label="Limiting Current")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.legend()
plt.grid()
plt.show()