import numpy as np
import matplotlib.pyplot as plt

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB

cfg = VRFBConfig()

currents_cmd = [0, 50, 100, 150, 200, 250]

voltages = []
currents_actual = []
powers = []

for I in currents_cmd:

    battery = VRFB(cfg)
    dt = cfg.dt_default

    for _ in range(1500):
        battery.step(I, cfg.initial_flow, 298.15, dt)

    out = battery.get_outputs()

    V = out["voltage_stack"]
    I_actual = out["current"]

    P = V * I_actual

    voltages.append(V)
    currents_actual.append(I_actual)
    powers.append(P)

    print(f"Requested: {I} A | Actual: {I_actual:.2f} A | Voltage: {V:.2f} V | Power: {P/1000:.2f} kW")


# Plot Power Curve
plt.figure()
plt.plot(currents_actual, powers)
plt.xlabel("Current (A)")
plt.ylabel("Power (W)")
plt.title("VRFB Power Curve")
plt.grid()

plt.show()