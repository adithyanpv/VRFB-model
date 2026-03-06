# experiments/test_5_step_load.py

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

cc.initialize(cfg.initial_soc)

dt = cfg.dt_default
sim_time = 8000   # ~2.2 hours
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

    # --------------------------------------------------------
    # STEP LOAD PROFILE
    # --------------------------------------------------------
    if t < 1000:
        I_cmd = 0
    elif t < 2500:
        I_cmd = 150
    elif t < 4000:
        I_cmd = 50
    elif t < 5500:
        I_cmd = 180
    elif t < 7000:
        I_cmd = -120   # Charging step
    else:
        I_cmd = 0

    T = 298.15
    Q_cmd = cfg.initial_flow

    # --------------------------------------------------------
    # 1️⃣ Get previous outputs
    # --------------------------------------------------------
    prev_out = battery.get_outputs()

    # --------------------------------------------------------
    # 2️⃣ Apply BMS protection
    # --------------------------------------------------------
    I_safe = bms.apply_protection(I_cmd, prev_out)

    # --------------------------------------------------------
    # 3️⃣ Run physics (with converter ramp inside)
    # --------------------------------------------------------
    battery.step(I_safe, Q_cmd, T, dt)

    # --------------------------------------------------------
    # 4️⃣ Get outputs
    # --------------------------------------------------------
    out = battery.get_outputs()

    # --------------------------------------------------------
    # 5️⃣ Sensor + Estimator
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
    current.append(out["current"])  # actual ramped current
    temperature.append(out["temperature_stack"])
    flow.append(out["flow_rate"])


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
true_soc = np.array(true_soc)
est_soc = np.array(est_soc)

rmse, mae, max_e, final_e = compute_metrics(true_soc, est_soc)

print("\n=== TEST 5: STEP LOAD TEST ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Max Error: {max_e:.6f}")
print(f"Final Drift: {final_e:.6f}")

# ------------------------------------------------------------
# Save Results
# ------------------------------------------------------------
folder = "results/test_5_step_load"

save_plots(folder,
           time_log, true_soc, est_soc,
           voltage, current, temperature, flow)

save_metrics(folder, rmse, mae, max_e, final_e)