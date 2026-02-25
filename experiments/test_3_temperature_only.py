# experiments/test_3_temperature_only.py

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.coulomb_counter import CoulombCounter
from vrfb.sensor_model import SensorModel
from vrfb.bms_controller import BMSController
from vrfb.utils import compute_metrics, save_plots, save_metrics

import numpy as np


# ------------------------------------------------------------
# Initialize
# ------------------------------------------------------------
cfg = VRFBConfig()
battery = VRFB(cfg)
bms = BMSController(cfg)
cc = CoulombCounter(cfg)
sensor = SensorModel(cfg)
bms = BMSController(cfg)

cc.initialize(cfg.initial_soc)

dt = cfg.dt_default
sim_time = 36000  # 10 hours
steps = int(sim_time / dt)

time_log = []
true_soc = []
est_soc = []
voltage = []
current = []
temperature = []
flow = []

# ------------------------------------------------------------
# Simulation Loop
# ------------------------------------------------------------
for step in range(steps):

    t = step * dt

    # Mild discharge for first hour
    if t < 3600:
        I_cmd = 120
    else:
        I_cmd = 0   # Rest period to observe crossover

    # Elevated temperature entire simulation
    T = 320.0  # High temperature
    Q_cmd = cfg.initial_flow

    prev_out = battery.get_outputs()

    I_safe = bms.apply_protection(I_cmd,prev_out)
    battery.step(I_safe, Q_cmd, T, dt)

   
    out = battery.get_outputs()

    measured = sensor.measure(out)

    soc_cc = cc.update(
        measured_current=measured["current"],
        dt=dt,
        Q_nominal=out["capacity_nominal"]
    )

    # Log
    time_log.append(t)
    true_soc.append(out["soc_true"])
    est_soc.append(soc_cc)
    voltage.append(out["voltage_stack"])
    current.append(out["current"])
    temperature.append(out["temperature_stack"])
    flow.append(out["flow_rate"])


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
true_soc = np.array(true_soc)
est_soc = np.array(est_soc)

rmse, mae, max_e, final_e = compute_metrics(true_soc, est_soc)

print("\n=== TEST 3: TEMPERATURE ONLY ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Max Error: {max_e:.6f}")
print(f"Final Drift: {final_e:.6f}")

# ------------------------------------------------------------
# Save Results
# ------------------------------------------------------------
folder = "results/test_3_temperature_only"

save_plots(folder,
           time_log, true_soc, est_soc,
           voltage, current, temperature, flow)

save_metrics(folder, rmse, mae, max_e, final_e)