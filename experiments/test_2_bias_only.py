# experiments/test_2_bias_only.py

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.coulomb_counter import CoulombCounter
from vrfb.sensor_model import SensorModel
from vrfb.utils import compute_metrics, save_plots, save_metrics
from vrfb.bms_controller import BMSController
import numpy as np


# ------------------------------------------------------------
# Initialize
# ------------------------------------------------------------
cfg = VRFBConfig()
battery = VRFB(cfg)
cc = CoulombCounter(cfg)
sensor = SensorModel(cfg)
bms = BMSController(cfg)

cc.initialize(cfg.initial_soc)

# 🔴 Add current bias (ONLY disturbance in this test)
sensor.set_current_bias(0.5)  # 0.5 A bias

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

    # Constant discharge for simplicity
    I_cmd = 120
    T = 298.15
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

print("\n=== TEST 2: BIAS ONLY ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Max Error: {max_e:.6f}")
print(f"Final Drift: {final_e:.6f}")

# ------------------------------------------------------------
# Save Results
# ------------------------------------------------------------
folder = "results/test_2_bias_only"

save_plots(folder,
           time_log, true_soc, est_soc,
           voltage, current, temperature, flow)

save_metrics(folder, rmse, mae, max_e, final_e)