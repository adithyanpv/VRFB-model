import os
import numpy as np
import pandas as pd

from vrfb.config import VRFBConfig
from vrfb.vrfb_core import VRFB
from vrfb.sensor_model import SensorModel
from vrfb.bms_controller import BMSController


# ============================================================
# CONFIGURATION
# ============================================================

cfg = VRFBConfig()

battery = VRFB(cfg)
sensor = SensorModel(cfg)


dt = cfg.dt_default

simulation_time = 300000   # seconds (~83 hours)
steps = int(simulation_time / dt)

os.makedirs("datasets", exist_ok=True)


# ============================================================
# DATA STORAGE
# ============================================================

data = []

prev_v = 0.0
prev_i = 0.0


# ============================================================
# LOAD PROFILE GENERATOR
# ============================================================

def load_profile(t):

    # random but physically realistic loads
    if 0 <= t < 50000:
        return np.random.uniform(50, 150)

    elif 50000 <= t < 100000:
        return np.random.uniform(-120, -40)

    elif 100000 <= t < 150000:
        return np.random.uniform(-150, 150)

    elif 150000 <= t < 200000:
        return 0

    elif 200000 <= t < 250000:
        return np.random.uniform(80, 200)

    else:
        return np.random.uniform(-100, 100)


# ============================================================
# FLOW PROFILE
# ============================================================

def flow_profile(t):

    if 0 <= t < 80000:
        return cfg.initial_flow

    elif 80000 <= t < 140000:
        return cfg.flow_min

    elif 140000 <= t < 220000:
        return (cfg.flow_min + cfg.flow_max) / 2

    else:
        return cfg.flow_max


# ============================================================
# SIMULATION LOOP
# ============================================================

for step in range(steps):

    t = step * dt

    I_cmd = load_profile(t)
    Q_cmd = flow_profile(t)

    # ambient temperature
    T_amb = np.random.uniform(295, 305)

    # BMS protection
    prev_out = battery.get_outputs()
    

    # simulate
    battery.step(I_cmd, Q_cmd, T_amb, dt)

    out = battery.get_outputs()
    measured = sensor.measure(out)

    V = measured["voltage"]
    I = measured["current"]
    T = measured["temperature"]

    flow = out["flow_rate"]
    I_limit = out["i_limit"]
    transport_ratio = out["transport_ratio"]

    soc_true = out["soc_true"]

    # derivatives
    dVdt = (V - prev_v) / dt
    dIdt = (I - prev_i) / dt

    prev_v = V
    prev_i = I

    data.append([
        t,
        V,
        I,
        T,
        flow,
        I_limit,
        transport_ratio,
        dVdt,
        dIdt,
        soc_true
    ])


# ============================================================
# SAVE DATASET
# ============================================================

columns = [
    "time",
    "voltage",
    "current",
    "temperature",
    "flow",
    "I_limit",
    "transport_ratio",
    "dVdt",
    "dIdt",
    "SOC_true"
]

df = pd.DataFrame(data, columns=columns)

dataset_path = "datasets/vrfb_dataset.csv"

df.to_csv(dataset_path, index=False)

print("\nDataset generated successfully.")
print("Samples:", len(df))
print("Saved to:", dataset_path)