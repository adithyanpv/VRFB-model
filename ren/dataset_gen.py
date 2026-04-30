
"""
VRFB Dataset Generator — v6  (Direct SOC Estimation)
=====================================================

CHANGES FROM v5
---------------
1. `target` column REMOVED from CSV.
   TARGET is now SOC_true (direct SOC), not SOC_true − soc_cc.
   train_ren.py reads SOC_true directly — no derived target needed.

2. `voltage` column REMOVED from CSV.
   v_ocv_approx = V_terminal − I×R_nom replaces it.
   voltage and v_ocv_approx are linearly dependent given current →
   collinearity caused overfitting in v5.

3. `soc_cc` KEPT in CSV (not a model feature — CC baseline metric only).
   train_ren.py uses soc_cc solely to compute the CC RMSE baseline.
   It is NOT passed to the model.

4. `soc_init_decay` NOT in CSV — computed dynamically in load_episodes():
     soc_init_decay[t] = SOC_true[t=0] × exp(−t / 1800s)
   At inference: computed from Nernst inversion of v_ocv_approx[0].

5. current_bias restored to ±5.0 A (was reduced to ±1.5 A in v5).
   At ±1.5 A, drift was too small — model had no incentive to learn
   from Nernst signal. At ±5 A, meaningful drift accumulates and the
   voltage-based correction matters.

CSV COLUMNS SAVED:
  episode_id, time,
  v_ocv_approx, current, temperature_stack, temperature_tank, flow_rate,
  soc_cc,    ← CC baseline metric only, NOT a model feature
  SOC_true   ← direct target for train_ren.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from vrfb.config          import VRFBConfig
from vrfb.vrfb_core       import VRFB
from vrfb.bms_controller  import BMSController, BMSMode
from vrfb.sensor_model    import SensorModel
from vrfb.coulomb_counter import CoulombCounter

os.makedirs("datasets", exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

N_EPISODES    = 200
EPISODE_STEPS = 36000     # 10 hours at dt=1s
TRAIN_FRAC    = 0.80      # 96 train / 24 test
SEED          = 42
rng           = np.random.default_rng(SEED)
_R_STACK_NOMINAL = (0.0015 + 0.0005) * 40   # 0.08 Ω
# ── Columns that train_ren.py needs from the CSV ─────────────────────────────
# soc_init_decay is computed dynamically in load_episodes() — NOT saved here.
# soc_cc is saved for CC baseline computation in train_ren.py, NOT as model input.
FEATURE_COLS = [
    "v_ocv_approx",      # Nernst OCV: V_terminal − I×R_nom  (no Ohmic jump)
    "current",           # sensor current [A]  (includes bias + noise)
    "temperature_stack",
    "temperature_tank",
    "flow_rate",
]
# No TARGET_COL in CSV — train_ren.py reads SOC_true directly as target.

# All columns saved to CSV
ROW_COLS = ["episode_id", "time"] + FEATURE_COLS + ["soc_cc", "SOC_true"]
N_COLS   = len(ROW_COLS)

# SOC bands for stratification
SOC_BANDS = [
    (0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45),
    (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95),
]
BAND_WEIGHTS = [3, 1, 1, 1, 1, 1, 1, 1, 3]
assert len(BAND_WEIGHTS) == len(SOC_BANDS)

_total_w  = sum(BAND_WEIGHTS)
_per_band = [round(N_EPISODES * w / _total_w) for w in BAND_WEIGHTS]
_per_band[4] += N_EPISODES - sum(_per_band)

episode_bands: list[tuple] = []
for band, n in zip(SOC_BANDS, _per_band):
    episode_bands.extend([band] * n)

_rng_bands = np.random.default_rng(SEED + 2)
_rng_bands.shuffle(episode_bands)

print("VRFB Dataset Generator  v6  (Direct SOC Estimation)")
print(f"  Episodes      : {N_EPISODES}  ({int(N_EPISODES*TRAIN_FRAC)} train / "
      f"{N_EPISODES - int(N_EPISODES*TRAIN_FRAC)} test, shuffled split)")
print(f"  Steps/episode : {EPISODE_STEPS:,}  ({EPISODE_STEPS/3600:.1f} hours)")
print(f"  Total rows    : ~{N_EPISODES * EPISODE_STEPS:,}")
print(f"  CSV columns   : {ROW_COLS}")
print(f"  Model features: {FEATURE_COLS} + soc_init_decay (computed in load_episodes)")
print(f"  Target        : SOC_true  (direct SOC — read by train_ren.py)")
print(f"  soc_cc in CSV : YES (CC baseline metric only, NOT a model input)")
print(f"  current_bias  : ±5.0 A  (restored from ±1.5 A)")
print("-" * 60)


# =============================================================================
# PROFILE GENERATORS
# =============================================================================

def make_current_profile(profile_type: str, steps: int,
                         init_soc: float, rng) -> np.ndarray:
    if profile_type == "discharge":
        base = rng.uniform(60, 140)
        I = np.full(steps, base)
        for _ in range(rng.integers(3, 7)):
            pos = rng.integers(500, steps - 500)
            I[pos:] = rng.uniform(40, 160)
        return I

    elif profile_type == "charge":
        base = rng.uniform(-140, -60)
        I = np.full(steps, base)
        for _ in range(rng.integers(3, 7)):
            pos = rng.integers(500, steps - 500)
            I[pos:] = rng.uniform(-160, -40)
        return I

    elif profile_type == "mixed_cycles":
        I    = np.zeros(steps)
        pos  = 0
        c_mag = rng.uniform(80, 130)
        d_mag = rng.uniform(80, 130)
        sign  = -1 if init_soc < 0.5 else 1
        while pos < steps:
            dur            = rng.integers(2000, 5000)
            I[pos:pos+dur] = sign * (c_mag if sign < 0 else d_mag)
            sign          *= -1
            pos           += dur
        return I

    elif profile_type == "step":
        I   = np.zeros(steps)
        pos = 0
        while pos < steps:
            dur = rng.integers(300, 2000)
            mag = rng.uniform(20, 150) * rng.choice([-1, 1])
            I[pos:pos+dur] = mag
            pos += dur
        return I

    elif profile_type == "variable_rate":
        t    = np.linspace(0, 1, steps)
        base = rng.uniform(60, 120)
        sign = 1 if init_soc > 0.5 else -1
        return sign * base * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 3) * t))

    elif profile_type == "standby_then_active":
        I        = np.zeros(steps)
        standby  = rng.integers(1800, 5400)
        mag      = rng.uniform(60, 140) * (1 if init_soc > 0.5 else -1)
        I[standby:] = mag
        return I
    elif profile_type == "current_reversal":
        # Profile specifically designed to teach Ohmic drop decoupling.
        # Multiple rapid direction changes at random magnitudes.
        # Forces the model to see V_terminal spike while SOC is smooth.
        I    = np.zeros(steps)
        pos  = 0
        while pos < steps:
            dur  = rng.integers(1200, 4000)   # 20-67 min per half
            mag  = rng.uniform(60, 180)
            sign = rng.choice([-1, 1])
            I[pos:pos+dur] = sign * mag
            pos += dur
        return I

    else:
        return np.zeros(steps)


CURRENT_PROFILES = ["discharge", "charge", "mixed_cycles", "step",
                    "variable_rate", "standby_then_active", "current_reversal"]

def make_flow_profile(profile_type: str, steps: int, cfg: VRFBConfig) -> np.ndarray:
    lo, hi = cfg.flow_min, cfg.flow_max
    if profile_type == "low":
        return np.full(steps, rng.uniform(lo, lo + 0.3*(hi-lo)))
    elif profile_type == "high":
        return np.full(steps, rng.uniform(lo + 0.6*(hi-lo), hi))
    elif profile_type == "very_high":
        return np.full(steps, hi)
    elif profile_type == "multi_step":
        Q   = np.zeros(steps)
        pos = 0
        while pos < steps:
            dur = rng.integers(2000, 6000)
            Q[pos:pos+dur] = rng.uniform(lo, hi)
            pos += dur
        return Q
    elif profile_type == "slow_sine":
        t = np.linspace(0, 1, steps)
        return lo + (hi - lo) * (0.5 + 0.5 * np.sin(2 * np.pi * t))
    else:
        return np.full(steps, rng.uniform(lo, hi))


FLOW_PROFILES = ["low", "high", "very_high", "multi_step", "slow_sine", "step"]


def make_temperature_profile(steps: int, T_base: float, rng) -> np.ndarray:
    increments      = rng.uniform(-0.01, 0.01, size=steps)
    increments[0]   = 0.0
    T = T_base + np.cumsum(increments)
    return np.clip(T, 290.0, 330.0).astype(np.float32)


# =============================================================================
# EPISODE GENERATION
# =============================================================================

all_chunks: list[np.ndarray] = []

for ep in tqdm(range(N_EPISODES), desc="Generating episodes"):
    band        = episode_bands[ep]
    init_soc    = float(rng.uniform(*band))

    if init_soc < 0.25:
        current_pool = ["charge", "mixed_cycles", "step",
                        "variable_rate", "standby_then_active"]
    elif init_soc > 0.75:
        current_pool = ["discharge", "mixed_cycles", "step",
                        "variable_rate", "standby_then_active"]
    else:
        current_pool = CURRENT_PROFILES

    current_type = rng.choice(current_pool)
    flow_type    = rng.choice(FLOW_PROFILES)

    if rng.random() < 0.20:
        T_base = rng.uniform(308.0, 322.0)   # hot regime (20% of episodes)
    else:
        T_base = rng.uniform(296.0, 303.0)

    # ── CC initial offset ─────────────────────────────────────────────────
    r = rng.random()
    if r < 0.20:
        # Hard case: large initial offset
        sign           = rng.choice([-1, 1])
        cc_init_offset = sign * rng.uniform(0.15, 0.25)
    elif r < 0.45:
        # NEW: CC starts accurate (matches cycle_test deployment scenario)
        # 25% of episodes — model must learn NOT to over-correct when CC is good
        cc_init_offset = rng.uniform(-0.02, 0.02)
    else:
        # Moderate initial offset
        cc_init_offset = rng.uniform(-0.10, 0.10)
    soc_cc_init = float(np.clip(init_soc + cc_init_offset, 0.06, 0.94))

    # ── Current sensor DC bias ────────────────────────────────────────────
    # ±5.0 A restored (was ±1.5 A in v5 — too small, model had no drift
    # signal and never learned to use Nernst voltage).
    current_bias = rng.uniform(-5.0, 5.0)

    # ── Physics objects ───────────────────────────────────────────────────
    cfg             = VRFBConfig()
    cfg.initial_soc = init_soc
    battery = VRFB(cfg)
    bms     = BMSController(cfg)
    sensor  = SensorModel(cfg)
    cc      = CoulombCounter(cfg)
    cc.initialize(soc_cc_init)
    sensor.set_current_bias(current_bias)
    

    # ── Half-cell imbalance injection ─────────────────────────────────────
    _r = rng.random()
    if _r < 0.60:
        imb_half = rng.uniform(0.0, 0.05)
    elif _r < 0.85:
        imb_half = rng.uniform(0.05, 0.12)
    else:
        imb_half = rng.uniform(0.12, 0.22)
    imb_sign  = rng.choice([-1.0, 1.0])
    imb_half *= imb_sign
    soc_neg_0 = float(np.clip(init_soc + imb_half, 0.05, 0.95))
    soc_pos_0 = float(np.clip(init_soc - imb_half, 0.05, 0.95))
    C = cfg.C_total
    battery.state[0] = soc_neg_0 * C;          battery.state[1] = (1-soc_neg_0)*C
    battery.state[2] = (1-soc_pos_0) * C;      battery.state[3] = soc_pos_0 * C
    battery.state[4] = soc_neg_0 * C;          battery.state[5] = (1-soc_neg_0)*C
    battery.state[6] = (1-soc_pos_0) * C;      battery.state[7] = soc_pos_0 * C

    dt = cfg.dt_default

    # ── Profiles ─────────────────────────────────────────────────────────
    I_profile = make_current_profile(current_type, EPISODE_STEPS, init_soc, rng)
    Q_profile = make_flow_profile(flow_type, EPISODE_STEPS, cfg)
    T_profile = make_temperature_profile(EPISODE_STEPS, T_base, rng)

    # ── Flow warmup (200 steps, I=0) ──────────────────────────────────────
    Q_first = float(Q_profile[0])
    for _ in range(200):
        battery.step(0.0, Q_first, T_profile[0], dt)
    cc.initialize(soc_cc_init)   # reset CC after warmup

    # ── Pre-allocate row buffer ───────────────────────────────────────────
    ep_data = np.empty((EPISODE_STEPS, N_COLS), dtype=np.float32)
    IDX     = {name: i for i, name in enumerate(ROW_COLS)}

    for step in range(EPISODE_STEPS):
        t     = step * dt
        I_cmd = float(I_profile[step])
        Q_cmd = float(Q_profile[step])
        T_amb = float(T_profile[step])

        out = battery.get_outputs()

        bms.update_mode(I_cmd, out, dt)
        flow_override = bms.get_flow_override()
        Q_eff         = flow_override if flow_override is not None else Q_cmd
        I_safe        = bms.apply_protection(I_cmd, out)

        battery.step(I_safe, Q_eff, T_amb, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)

        # CC update — measured["current"] already has sensor bias applied by
        # SensorModel (sensor.set_current_bias was called above).
        # Do NOT add current_bias again here — that was the old double-bias bug.
        soc_cc = float(cc.update(
            measured_current = measured["current"],
            dt               = dt,
            Q_nominal        = out["capacity_nominal"],
        ))

        soc_true = float(out["soc_true"])
        # soc_cc saved for CC baseline metric in train_ren.py — NOT a model feature
        # raw_target (SOC_true - soc_cc) is NO LONGER saved — TARGET is SOC_true directly

        # Ohmic-corrected OCV approximation (hardware-computable)
        v_ocv = measured["voltage"] + measured["current"] * _R_STACK_NOMINAL

        ep_data[step, IDX["episode_id"]]        = ep
        ep_data[step, IDX["time"]]              = t
        ep_data[step, IDX["v_ocv_approx"]]      = float(v_ocv)
        ep_data[step, IDX["current"]]           = measured["current"]
        ep_data[step, IDX["temperature_stack"]] = measured["temperature"]
        ep_data[step, IDX["temperature_tank"]]  = measured["temperature_tank"]
        ep_data[step, IDX["flow_rate"]]         = measured["flow_rate"]
        ep_data[step, IDX["soc_cc"]]            = soc_cc
        ep_data[step, IDX["SOC_true"]]          = soc_true

    all_chunks.append(ep_data)


# =============================================================================
# BUILD DATAFRAME AND SPLIT
# =============================================================================

df = pd.DataFrame(
    np.concatenate(all_chunks, axis=0),
    columns=ROW_COLS,
)
df["episode_id"] = df["episode_id"].astype(int)
df["time"]       = df["time"].astype(np.float32)

# Shuffled train/test split
all_ep_ids   = np.arange(N_EPISODES)
rng_split    = np.random.default_rng(SEED + 99)
shuffled_ids = rng_split.permutation(all_ep_ids)
n_train      = int(N_EPISODES * TRAIN_FRAC)
train_ids    = set(shuffled_ids[:n_train].tolist())
test_ids     = set(shuffled_ids[n_train:].tolist())

df_train = df[df["episode_id"].isin(train_ids)].reset_index(drop=True)
df_test  = df[df["episode_id"].isin(test_ids)].reset_index(drop=True)

df_train.to_csv("datasets/vrfb_train.csv", index=False)
df_test.to_csv( "datasets/vrfb_test.csv",  index=False)

print(f"\nSaved datasets/vrfb_train.csv  ({len(df_train):,} rows, "
      f"{df_train['episode_id'].nunique()} episodes)")
print(f"Saved datasets/vrfb_test.csv   ({len(df_test):,} rows, "
      f"{df_test['episode_id'].nunique()} episodes)")
print(f"\nCSV columns  : {ROW_COLS}")
print(f"Model inputs : {FEATURE_COLS} + soc_init_decay (dynamic, from SOC_true[0])")
print(f"Target       : SOC_true  (read directly by train_ren.py)")
print(f"CC baseline  : soc_cc column (metric only, not model input)")
print("Done.")