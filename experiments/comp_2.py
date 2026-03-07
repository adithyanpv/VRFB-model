import torch
import numpy as np
import joblib  # 🔴 IMPORT JOBLIB TO LOAD YOUR SCALER
import matplotlib.pyplot as plt

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.sensor_model import SensorModel
from vrfb.coulomb_counter import CoulombCounter
from vrfb.utils import compute_metrics
from ren.ren_model import REN

# --------------------------------------------------
# Load REN model and Scaler
# --------------------------------------------------
model = REN()
model.load_state_dict(torch.load("ren/ren_soc_model.pth"))
model.eval()

# 🔴 LOAD THE SCALER
scaler = joblib.load("ren/scaler.pkl")

z = torch.zeros(1, 64)

# --------------------------------------------------
# Initialize system
# --------------------------------------------------
cfg = VRFBConfig()
battery = VRFB(cfg)
sensor = SensorModel(cfg)
cc = CoulombCounter(cfg)

cc.initialize(cfg.initial_soc)

# 🔴 ADD BIAS TO MAKE THE COULOMB COUNTER DRIFT
sensor.set_current_bias(0.8)

dt = cfg.dt_default
sim_time = 20000
steps = int(sim_time / dt)

# --------------------------------------------------
# Logs
# --------------------------------------------------
true_soc = []
cc_soc = []
ren_soc = []
prev_v = 46.0  # Initialize with a realistic voltage to prevent dVdt spikes
prev_i = 0.0

# --------------------------------------------------
# Simulation loop
# --------------------------------------------------
current_load = 100.0  # Start with a 100A discharge
time_to_switch = 2000 # Hold it for 2000 seconds

for step in range(steps):
    t = step * dt

    # 🔴 REALISTIC LOAD PROFILE
    if time_to_switch <= 0:
        current_load = np.random.uniform(-120, 150)
        time_to_switch = np.random.randint(1000, 4000)
    time_to_switch -= dt
    
    I_cmd = current_load
    Q_cmd = cfg.initial_flow # Keep flow constant for basic validation first

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
    # 🔴 CRITICAL: SCALE THE DATA BEFORE PASSING TO PYTORCH
    raw_inputs = np.array([[V, I, T, flow, I_limit, tr, dVdt, dIdt]])
    scaled_inputs = scaler.transform(raw_inputs)
    
    x = torch.tensor(scaled_inputs, dtype=torch.float32)

    with torch.no_grad():
        pred, z = model(x, z)

    soc_ren = pred.item()

    # ----------------------------
    # Logging
    # ----------------------------
    true_soc.append(out["soc_true"])
    cc_soc.append(soc_cc)
    ren_soc.append(soc_ren)

# --------------------------------------------------
# Metrics & Plotting
# --------------------------------------------------
true_soc = np.array(true_soc)
cc_soc = np.array(cc_soc)
ren_soc = np.array(ren_soc)

rmse_cc, mae_cc, max_cc, drift_cc = compute_metrics(true_soc,cc_soc)
rmse_ren, mae_ren, max_ren, drift_ren = compute_metrics(true_soc,ren_soc)

print("\nCoulomb Counter")
print(f"RMSE: {rmse_cc:.6f}")

print("\nREN Estimator")
print(f"RMSE: {rmse_ren:.6f}")

time = np.arange(len(true_soc))

plt.figure(figsize=(10,6))
plt.plot(time, true_soc, label="True SOC", linewidth=2)
plt.plot(time, cc_soc, label="Coulomb Counter SOC")
plt.plot(time, ren_soc, label="REN SOC", linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("SOC")
plt.title("SOC Estimator Comparison")
plt.legend()
plt.grid()
plt.show()