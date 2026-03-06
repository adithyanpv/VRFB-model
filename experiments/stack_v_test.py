from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.bms_controller import BMSController

cfg = VRFBConfig()
battery = VRFB(cfg)
bms = BMSController(cfg)

dt = cfg.dt_default
sim_time = 2000

I_cmd = 120  # constant discharge current

for step in range(int(sim_time/dt)):

    out = battery.get_outputs()

    I_safe = bms.apply_protection(I_cmd, out)

    battery.step(I_safe, cfg.initial_flow, 298.15, dt)

    if step % 100 == 0:
        print("Voltage:", out["voltage_stack"])
out = battery.get_outputs()

print("Voltage:", out["voltage_stack"])
print("Current:", out["current"])