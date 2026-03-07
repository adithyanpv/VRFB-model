import torch
import numpy as np

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.sensor_model import SensorModel
from vrfb.coulomb_counter import CoulombCounter
from vrfb.utils import compute_metrics

from ren.ren_model import REN
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load REN model
# --------------------------------------------------

model = REN()
model.load_state_dict(torch.load("ren/ren_soc_model.pth"))
model.eval()

z = torch.zeros(1,64)


# --------------------------------------------------
# Initialize system
# --------------------------------------------------

cfg = VRFBConfig()

battery = VRFB(cfg)
sensor = SensorModel(cfg)
cc = CoulombCounter(cfg)

cc.initialize(cfg.initial_soc)

dt = cfg.dt_default
sim_time = 20000
steps = int(sim_time / dt)


# --------------------------------------------------
# Logs
# --------------------------------------------------

true_soc = []
cc_soc = []
ren_soc = []

prev_v = 0
prev_i = 0


# --------------------------------------------------
# Simulation loop
# --------------------------------------------------

for step in range(steps):

    t = step * dt

    # dynamic load
    I_cmd = np.random.uniform(-150,150)

    Q_cmd = np.random.uniform(cfg.flow_min, cfg.flow_max)

    battery.step(I_cmd, Q_cmd, 298.15, dt)

    out = battery.get_outputs()
    measured = sensor.measure(out)

    V = measured["voltage"]
    I = measured["current"]
    T = measured["temperature"]

    flow = out["flow_rate"]
    I_limit = out["i_limit"]
    tr = out["transport_ratio"]

    dVdt = (V - prev_v)/dt
    dIdt = (I - prev_i)/dt

    prev_v = V
    prev_i = I


    # ----------------------------
    # Coulomb counter
    # ----------------------------

    soc_cc = cc.update(
        measured_current=I,
        dt=dt,
        Q_nominal=out["capacity_nominal"]
    )


    # ----------------------------
    # REN estimator
    # ----------------------------

    x = torch.tensor([[V,I,T,flow,I_limit,tr,dVdt,dIdt]],dtype=torch.float32)

    with torch.no_grad():
        pred,z = model(x,z)

    soc_ren = pred.item()


    # ----------------------------
    # Logging
    # ----------------------------

    true_soc.append(out["soc_true"])
    cc_soc.append(soc_cc)
    ren_soc.append(soc_ren)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

true_soc = np.array(true_soc)
cc_soc = np.array(cc_soc)
ren_soc = np.array(ren_soc)

rmse_cc, mae_cc, max_cc, drift_cc = compute_metrics(true_soc,cc_soc)
rmse_ren, mae_ren, max_ren, drift_ren = compute_metrics(true_soc,ren_soc)


print("\nCoulomb Counter")
print("RMSE:",rmse_cc)
print("MAE:",mae_cc)
print("Max error:",max_cc)
print("Drift:",drift_cc)


print("\nREN Estimator")
print("RMSE:",rmse_ren)
print("MAE:",mae_ren)
print("Max error:",max_ren)
print("Drift:",drift_ren)
time = np.arange(len(true_soc))

plt.figure(figsize=(10,6))

plt.plot(time, true_soc, label="True SOC", linewidth=2)
plt.plot(time, cc_soc, label="Coulomb Counter SOC")
plt.plot(time, ren_soc, label="REN SOC")

plt.xlabel("Time Step")
plt.ylabel("SOC")
plt.title("SOC Estimator Comparison")
plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(10,6))

plt.plot(time, true_soc - cc_soc, label="CC Error")
plt.plot(time, true_soc - ren_soc, label="REN Error")

plt.xlabel("Time Step")
plt.ylabel("SOC Error")
plt.title("SOC Estimation Error")
plt.legend()
plt.grid()

plt.show()