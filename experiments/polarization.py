import numpy as np
import matplotlib.pyplot as plt

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.bms_controller import BMSController

cfg = VRFBConfig()
bms = BMSController(cfg)

currents = [0, 50, 100, 150, 200, 250]

voltages = []

for I_cmd in currents:

    battery = VRFB(cfg)
    dt = cfg.dt_default

    for _ in range(1500):

        # Get current battery state
        outputs = battery.get_outputs()

        # Apply BMS protection
        I_safe = bms.apply_protection(I_cmd, outputs)

        # Run physics with safe current
        battery.step(I_safe, cfg.initial_flow, 298.15, dt)

    out = battery.get_outputs()

    voltages.append(out["voltage_stack"])

    print(f"Requested current: {I_cmd}")
    print(f"Actual current: {out['current']}")
    print(f"Voltage: {out['voltage_stack']:.2f} V\n")


plt.plot(currents, voltages, marker="o")

plt.xlabel("Commanded Current (A)")
plt.ylabel("Stack Voltage (V)")
plt.title("VRFB Polarization Curve (With BMS)")

plt.grid()

plt.show()