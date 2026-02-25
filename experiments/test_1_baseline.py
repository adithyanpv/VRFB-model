from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.coulomb_counter import CoulombCounter
from vrfb.sensor_model import SensorModel
from vrfb.utils import compute_metrics, save_plots,save_metrics
from vrfb.bms_controller import BMSController
import numpy as np


cfg = VRFBConfig()
battery = VRFB(cfg)
cc = CoulombCounter(cfg)
sensor = SensorModel(cfg)
bms = BMSController(cfg)

cc.initialize(cfg.initial_soc)

dt = cfg.dt_default
sim_time = 3600
steps = int(sim_time / dt)

time_log = []
true_soc = []
est_soc = []
voltage = []
current = []
temperature = []
flow = []

for step in range(steps):

    t = step * dt
    I_cmd = 120 if 500 < t < 1500 else 0
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

    time_log.append(t)
    true_soc.append(out["soc_true"])
    est_soc.append(soc_cc)
    voltage.append(out["voltage_stack"])
    current.append(out["current"])
    temperature.append(out["temperature_stack"])
    flow.append(out["flow_rate"])

true_soc = np.array(true_soc)
est_soc = np.array(est_soc)

rmse, mae, max_e, final_e = compute_metrics(true_soc, est_soc)

print("=== BASELINE TEST ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Max Error: {max_e:.6f}")
print(f"Final Drift: {final_e:.6f}")

folder = "results/test_1_baseline"

save_plots(folder,
           time_log, true_soc, est_soc,
           voltage, current, temperature, flow)

save_metrics(folder, rmse, mae, max_e, final_e)