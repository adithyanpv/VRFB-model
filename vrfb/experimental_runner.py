# experiment_runner.py

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

dt = cfg.dt_default
simulation_time = 3600  # 1 hour
steps = int(simulation_time / dt)

true_soc_log = []
cc_soc_log = []
time_log = []

for step in range(steps):

    t = step * dt

    # Example test profile (pulse current)
    if 500 < t < 1500:
        I = 350
    elif 2000 < t < 2500:
        I = -320
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

    true_soc_log.append(outputs["soc_true"])
    cc_soc_log.append(soc_cc)
    time_log.append(t)


plt.plot(time_log, true_soc_log, label="True SOC")
plt.plot(time_log, cc_soc_log, label="Coulomb Counter SOC")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("SOC")
plt.title("SOC Comparison")
plt.show()