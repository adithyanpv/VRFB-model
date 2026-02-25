# experiment_runner.py

import numpy as np
import matplotlib.pyplot as plt

from config import VRFBConfig
from vrfb_core import VRFB
from coulomb_counter import CoulombCounter
from sensor_model import SensorModel


# ------------------------------------------------------------
# Initialize System
# ------------------------------------------------------------
cfg = VRFBConfig()
battery = VRFB(cfg)
cc = CoulombCounter(cfg)
sensor = SensorModel(cfg)

cc.initialize(cfg.initial_soc)

dt = cfg.dt_default
simulation_time = 3600  # 1 hour
steps = int(simulation_time / dt)

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
time_log = []
true_soc_log = []
cc_soc_log = []
error_log = []

voltage_log = []
current_log = []
temperature_log = []
flow_log = []

# ------------------------------------------------------------
# Simulation Loop
# ------------------------------------------------------------
for step in range(steps):

    t = step * dt

    # Example load profile
    if 500 < t < 1500:
        I = 150
    elif 2000 < t < 2500:
        I = -120
    else:
        I = 0

    battery.step(I, cfg.initial_flow, 298.15, dt)
    outputs = battery.get_outputs()

    measured = sensor.measure(outputs)

    soc_cc = cc.update(
        measured_current=measured["current"],
        dt=dt,
        Q_nominal=outputs["capacity_nominal"]
    )

    # Log values
    time_log.append(t)
    true_soc_log.append(outputs["soc_true"])
    cc_soc_log.append(soc_cc)
    error_log.append(outputs["soc_true"] - soc_cc)

    voltage_log.append(outputs["voltage_stack"])
    current_log.append(outputs["current"])
    temperature_log.append(outputs["temperature_stack"])
    flow_log.append(outputs["flow_rate"])


# ------------------------------------------------------------
# Convert to numpy arrays
# ------------------------------------------------------------
true_soc_log = np.array(true_soc_log)
cc_soc_log = np.array(cc_soc_log)
error_log = np.array(error_log)

# ------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------
rmse = np.sqrt(np.mean(error_log**2))
mae = np.mean(np.abs(error_log))
max_error = np.max(np.abs(error_log))
final_error = error_log[-1]

print("\n=== SOC Estimation Performance ===")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Max Absolute Error: {max_error:.6f}")
print(f"Final Drift Error: {final_error:.6f}")


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------

plt.figure(figsize=(14, 10))

# SOC comparison
plt.subplot(3, 2, 1)
plt.plot(time_log, true_soc_log, label="True SOC")
plt.plot(time_log, cc_soc_log, label="Coulomb Counter SOC")
plt.title("SOC Comparison")
plt.xlabel("Time (s)")
plt.ylabel("SOC")
plt.legend()
plt.grid()

# SOC Error
plt.subplot(3, 2, 2)
plt.plot(time_log, error_log)
plt.title("SOC Estimation Error")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.grid()

# Voltage
plt.subplot(3, 2, 3)
plt.plot(time_log, voltage_log)
plt.title("Stack Voltage")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid()

# Current
plt.subplot(3, 2, 4)
plt.plot(time_log, current_log)
plt.title("Current Profile")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.grid()

# Temperature
plt.subplot(3, 2, 5)
plt.plot(time_log, temperature_log)
plt.title("Stack Temperature")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.grid()

# Flow
plt.subplot(3, 2, 6)
plt.plot(time_log, flow_log)
plt.title("Flow Rate")
plt.xlabel("Time (s)")
plt.ylabel("Flow (m³/s)")
plt.grid()

plt.tight_layout()
plt.show()