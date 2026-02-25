# experiment_runner_stress.py

import numpy as np
import matplotlib.pyplot as plt

from config import VRFBConfig
from vrfb_core import VRFB
from coulomb_counter import CoulombCounter
from sensor_model import SensorModel


cfg = VRFBConfig()
battery = VRFB(cfg)
cc = CoulombCounter(cfg)
sensor = SensorModel(cfg)

cc.initialize(cfg.initial_soc)

# Add current bias
sensor.set_current_bias(0.5)  # 0.5 A bias

dt = cfg.dt_default
simulation_time = 36000  # 10 hours
steps = int(simulation_time / dt)

time_log = []
true_soc_log = []
cc_soc_log = []
error_log = []

voltage_log = []
current_log = []
temperature_log = []
flow_log = []

for step in range(steps):

    t = step * dt

    # -------------------------------
    # Load Profile
    # -------------------------------
    if t < 3600:                     # 0–1 hr
        I = 120
    elif t < 10800:                  # 1–3 hr
        I = 150
    elif t < 18000:                  # 3–5 hr
        I = 180
    elif t < 25200:                  # 5–7 hr
        I = -140
    else:                            # 7–10 hr
        I = 0

    # -------------------------------
    # Temperature Profile
    # -------------------------------
    if 3600 < t < 10800:
        T = 320.0   # Elevated temperature
    elif t > 25200:
        T = 315.0
    else:
        T = 298.15

    # -------------------------------
    # Flow Starvation
    # -------------------------------
    if 10800 < t < 18000:
        flow = cfg.flow_min   # Starved
    else:
        flow = cfg.initial_flow

    # Step battery
    battery.step(I, flow, T, dt)
    outputs = battery.get_outputs()

    measured = sensor.measure(outputs)

    soc_cc = cc.update(
        measured_current=measured["current"],
        dt=dt,
        Q_nominal=outputs["capacity_nominal"]
    )

    # Log data
    time_log.append(t)
    true_soc_log.append(outputs["soc_true"])
    cc_soc_log.append(soc_cc)
    error_log.append(outputs["soc_true"] - soc_cc)

    voltage_log.append(outputs["voltage_stack"])
    current_log.append(outputs["current"])
    temperature_log.append(outputs["temperature_stack"])
    flow_log.append(outputs["flow_rate"])


# Convert arrays
true_soc_log = np.array(true_soc_log)
cc_soc_log = np.array(cc_soc_log)
error_log = np.array(error_log)

# -------------------------------
# Metrics
# -------------------------------
rmse = np.sqrt(np.mean(error_log**2))
mae = np.mean(np.abs(error_log))
max_error = np.max(np.abs(error_log))
final_error = error_log[-1]

print("\n=== STRESS TEST RESULTS ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Max Absolute Error: {max_error:.6f}")
print(f"Final Drift Error: {final_error:.6f}")


# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(14, 10))

plt.subplot(3, 2, 1)
plt.plot(time_log, true_soc_log, label="True SOC")
plt.plot(time_log, cc_soc_log, label="Coulomb Counter SOC")
plt.title("SOC Comparison")
plt.legend()
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(time_log, error_log)
plt.title("SOC Error")
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(time_log, voltage_log)
plt.title("Voltage")
plt.grid()

plt.subplot(3, 2, 4)
plt.plot(time_log, current_log)
plt.title("Current")
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(time_log, temperature_log)
plt.title("Temperature")
plt.grid()

plt.subplot(3, 2, 6)
plt.plot(time_log, flow_log)
plt.title("Flow Rate")
plt.grid()

plt.tight_layout()
plt.show()