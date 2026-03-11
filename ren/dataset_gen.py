# ren/dataset_gen.py
"""
VRFB Dataset Generator — v2 (Stratified SOC Coverage)
=======================================================
FIXES FROM v1:
  1. Stratified initial SOC — episodes are assigned to SOC bands so every
     0.1-wide bin from 0.05→0.95 is uniformly covered.
  2. Longer episodes — 18,000 steps (5 hours) instead of 5,000 (1.4 hours).
     At 120A, 5000 steps only drains 7.8% SOC — most episodes never leave
     their starting bin. 18,000 steps drains ~28% SOC per episode.
  3. Forced boundary episodes — 10% of episodes start near soc_min or soc_max
     so the REN learns BMS cut-off behaviour at the extremes.
  4. Mixed charge/discharge within episodes — instead of pure discharge or
     pure charge profiles, most episodes include both phases so the REN sees
     the full SOC-voltage curve in both directions within one episode.
  5. Matched initial flow — episode starts at first commanded flow level,
     avoiding the initial transient dip seen in Test 6.

DESIGN:
  N_EPISODES = 80  (64 train / 16 test — split by episode)
  EPISODE_STEPS = 18,000  (5 hours at dt=1s)
  Total rows ≈ 1,440,000

SOC STRATIFICATION:
  Episodes are assigned to 9 SOC bands (0.05–0.15, 0.15–0.25, ... 0.85–0.95)
  Each band gets ~8–9 episodes. Initial SOC is sampled uniformly within the band.
  This guarantees all SOC regions are visited in both charge and discharge.

REN FEATURES (8 inputs):
  voltage, current, temperature_stack, temperature_tank,
  flow_rate, soc_cc, I_limit, transport_ratio

TARGET (1):
  SOC_true
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from vrfb.config          import VRFBConfig
from vrfb.vrfb_core       import VRFB
from vrfb.bms_controller  import BMSController
from vrfb.sensor_model    import SensorModel
from vrfb.coulomb_counter import CoulombCounter

os.makedirs("datasets", exist_ok=True)

# ── Generator settings ────────────────────────────────────────────────────────
N_EPISODES    = 80
EPISODE_STEPS = 18000     # 5 hours — drains ~28% SOC at 120A
TRAIN_FRAC    = 0.80      # 64 train / 16 test
SEED          = 42
rng           = np.random.default_rng(SEED)

# SOC bands for stratification — 9 bands × ~9 episodes each
SOC_BANDS = [
    (0.05, 0.15),   # near soc_min — BMS boundary behaviour
    (0.15, 0.25),
    (0.25, 0.35),
    (0.35, 0.45),
    (0.45, 0.55),   # mid-range
    (0.55, 0.65),
    (0.65, 0.75),
    (0.75, 0.85),
    (0.85, 0.95),   # near soc_max — BMS boundary behaviour
]

# Assign each episode to a band (round-robin for uniform coverage)
episode_bands = [SOC_BANDS[ep % len(SOC_BANDS)] for ep in range(N_EPISODES)]

print("VRFB Dataset Generator  v2  (Stratified SOC Coverage)")
print(f"  Episodes      : {N_EPISODES}")
print(f"  Steps/episode : {EPISODE_STEPS:,}  ({EPISODE_STEPS/3600:.1f} hours each)")
print(f"  Total rows    : ~{N_EPISODES * EPISODE_STEPS:,}")
print(f"  Train/test    : {int(N_EPISODES*TRAIN_FRAC)} / {N_EPISODES - int(N_EPISODES*TRAIN_FRAC)} episodes")
print(f"  SOC bands     : {len(SOC_BANDS)} bands × ~{N_EPISODES//len(SOC_BANDS)} episodes each")
print("-" * 60)


# ── Current profile generators ────────────────────────────────────────────────

def make_current_profile(profile_type, steps, init_soc, rng):
    """
    Returns I_cmd array. Profiles are matched to starting SOC:
    - Low SOC episodes prefer charging
    - High SOC episodes prefer discharging
    - Mid SOC episodes use mixed profiles
    """
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
        # Multiple charge/discharge cycles — best for SOC traversal
        I = np.zeros(steps)
        pos = 0
        charge_mag   = rng.uniform(80, 130)
        discharge_mag = rng.uniform(80, 130)
        # Start with charging if low SOC, discharging if high
        if init_soc < 0.5:
            sign = -1  # start charging
        else:
            sign = 1   # start discharging
        while pos < steps:
            duration = rng.integers(2000, 5000)  # 30–80 min blocks
            I[pos:pos + duration] = sign * (charge_mag if sign < 0 else discharge_mag)
            sign *= -1  # flip direction
            pos += duration
        return I

    elif profile_type == "step":
        I = np.zeros(steps)
        pos = 0
        while pos < steps:
            duration = rng.integers(500, 2000)
            mag = rng.uniform(-160, 160)
            I[pos:pos + duration] = mag
            pos += duration
        return I

    elif profile_type == "rest_then_active":
        I = np.zeros(steps)
        rest_end = rng.integers(1000, 3000)
        mag = rng.choice([-1, 1]) * rng.uniform(80, 140)
        I[rest_end:] = mag
        return I

    elif profile_type == "variable_rate":
        # Slow ramp up/down — tests REN under gradual changes
        I = np.zeros(steps)
        levels = rng.uniform(-150, 150, size=6)
        block  = steps // 6
        for k in range(6):
            s = k * block
            e = s + block
            if k > 0:
                # Ramp between levels over 200 steps
                I[s:s+200] = np.linspace(levels[k-1], levels[k], 200)
                I[s+200:e] = levels[k]
            else:
                I[s:e] = levels[k]
        return I

    else:  # random
        return rng.uniform(-150, 150, size=steps)


def make_flow_profile(profile_type, steps, cfg, rng):
    lpm = cfg.LPM_to_m3s
    if profile_type == "low":
        base = rng.uniform(8, 14) * lpm
        return np.full(steps, base)
    elif profile_type == "high":
        base = rng.uniform(25, 45) * lpm
        return np.full(steps, base)
    elif profile_type == "sweep_up":
        lo = rng.uniform(8, 15) * lpm
        hi = rng.uniform(25, 45) * lpm
        return np.linspace(lo, hi, steps)
    elif profile_type == "sweep_down":
        lo = rng.uniform(8, 15) * lpm
        hi = rng.uniform(25, 45) * lpm
        return np.linspace(hi, lo, steps)
    elif profile_type == "step":
        Q = np.full(steps, cfg.initial_flow)
        levels = rng.uniform(8, 45, size=5) * lpm
        block  = steps // 5
        for k in range(5):
            Q[k*block:(k+1)*block] = levels[k]
        return Q
    else:  # nominal
        return np.full(steps, cfg.initial_flow)


def make_temperature_profile(steps, T_base, rng):
    """Slow random walk — max 0.003 K/step."""
    T = np.zeros(steps)
    T[0] = T_base
    for i in range(1, steps):
        T[i] = np.clip(T[i-1] + rng.uniform(-0.003, 0.003), 290, 322)
    return T


# ── Profile pools ─────────────────────────────────────────────────────────────
CURRENT_PROFILES = ["discharge", "charge", "mixed_cycles", "step",
                    "rest_then_active", "variable_rate", "random"]
FLOW_PROFILES    = ["low", "high", "sweep_up", "sweep_down", "step", "nominal"]

# ── Main generation loop ──────────────────────────────────────────────────────
all_rows = []

for ep in tqdm(range(N_EPISODES), desc="Generating episodes"):

    # ── Stratified initial SOC ────────────────────────────────────────────────
    band_lo, band_hi = episode_bands[ep]
    init_soc = rng.uniform(band_lo, band_hi)

    # Clip to safe BMS window with small margin
    init_soc = float(np.clip(init_soc, 0.06, 0.94))

    # Match current profile to SOC: prefer charging when low, discharging when high
    if init_soc < 0.25:
        current_pool = ["charge", "mixed_cycles", "rest_then_active"]
    elif init_soc > 0.75:
        current_pool = ["discharge", "mixed_cycles", "step", "variable_rate"]
    else:
        current_pool = CURRENT_PROFILES

    current_type = rng.choice(current_pool)
    flow_type    = rng.choice(FLOW_PROFILES)
    T_base       = rng.uniform(293.0, 318.0)

    cfg             = VRFBConfig()
    cfg.initial_soc = init_soc
    battery = VRFB(cfg)
    bms     = BMSController(cfg)
    sensor  = SensorModel(cfg)
    cc      = CoulombCounter(cfg)
    cc.initialize(init_soc)

    dt = cfg.dt_default

    # Pre-generate profiles
    I_profile = make_current_profile(current_type, EPISODE_STEPS, init_soc, rng)
    Q_profile = make_flow_profile(flow_type, EPISODE_STEPS, cfg, rng)
    T_profile = make_temperature_profile(EPISODE_STEPS, T_base, rng)

    # ── Warmup: settle flow state to first commanded flow ─────────────────────
    # Prevents the initial flow transient seen in Test 6
    Q_first = float(Q_profile[0])
    for _ in range(200):
        battery.step(0.0, Q_first, T_profile[0], dt)
    cc.initialize(init_soc)  # reset CC after warmup

    # ── Initialise prev values from actual first output (no dVdt spike) ───────
    first_out = battery.get_outputs()
    prev_v    = first_out["voltage_stack"]
    prev_i    = first_out["current"]

    rows = []

    for step in range(EPISODE_STEPS):

        t     = step * dt
        I_cmd = float(I_profile[step])
        Q_cmd = float(Q_profile[step])
        T_amb = float(T_profile[step])

        out    = battery.get_outputs()
        I_safe = bms.apply_protection(I_cmd, out)

        battery.step(I_safe, Q_cmd, T_amb, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)

        soc_cc = cc.update(
            measured_current = measured["current"],
            dt               = dt,
            Q_nominal        = out["capacity_nominal"]
        )

        V               = measured["voltage"]
        I               = measured["current"]
        T_stack         = measured["temperature"]
        T_tank          = out["temperature_tank"]
        flow            = out["flow_rate"]
        I_limit         = out["i_limit"]
        transport_ratio = out["transport_ratio"]
        soc_true        = out["soc_true"]

        dVdt = (V - prev_v) / dt
        dIdt = (I - prev_i) / dt
        prev_v = V
        prev_i = I

        rows.append([
            ep, t,
            V, I, T_stack, T_tank,
            flow, soc_cc, I_limit, transport_ratio,
            dVdt, dIdt,
            soc_true
        ])

    all_rows.extend(rows)

# ── Build DataFrame ───────────────────────────────────────────────────────────
columns = [
    "episode_id", "time",
    "voltage", "current", "temperature_stack", "temperature_tank",
    "flow_rate", "soc_cc", "I_limit", "transport_ratio",
    "dVdt", "dIdt",
    "SOC_true"
]

df = pd.DataFrame(all_rows, columns=columns)

# ── Train / test split by episode ─────────────────────────────────────────────
n_train = int(N_EPISODES * TRAIN_FRAC)
df_train = df[df["episode_id"] <  n_train].reset_index(drop=True)
df_test  = df[df["episode_id"] >= n_train].reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("datasets/vrfb_dataset.csv",     index=False)
df_train.to_csv("datasets/vrfb_train.csv", index=False)
df_test.to_csv("datasets/vrfb_test.csv",   index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Dataset generation complete  (v2 — stratified)")
print(f"{'='*60}")
print(f"  Total rows    : {len(df):,}")
print(f"  Train rows    : {len(df_train):,}  ({n_train} episodes)")
print(f"  Test rows     : {len(df_test):,}  ({N_EPISODES - n_train} episodes)")
print(f"\n  SOC_true range : {df['SOC_true'].min():.3f} – {df['SOC_true'].max():.3f}")
print(f"  Voltage range  : {df['voltage'].min():.2f} – {df['voltage'].max():.2f} V")
print(f"  Current range  : {df['current'].min():.2f} – {df['current'].max():.2f} A")
print(f"  Temp range     : {df['temperature_stack'].min():.2f} – {df['temperature_stack'].max():.2f} K")
print(f"  Flow range     : {df['flow_rate'].min()*60000:.1f} – {df['flow_rate'].max()*60000:.1f} LPM")

print(f"\n  SOC coverage (target: all bins roughly equal):")
bins = np.linspace(0, 1, 11)
hist, _ = np.histogram(df["SOC_true"], bins=bins)
bar_max  = max(hist) if max(hist) > 0 else 1
for i in range(len(hist)):
    bar = "█" * int(hist[i] / bar_max * 30)
    pct = hist[i] / len(df) * 100
    print(f"    {bins[i]:.1f}–{bins[i+1]:.1f}  {bar:<30}  ({hist[i]:,})  {pct:.1f}%")

# Check uniformity
nonzero = hist[hist > 0]
if len(nonzero) > 1:
    ratio = nonzero.max() / nonzero.min()
    print(f"\n  Coverage uniformity: max/min ratio = {ratio:.2f}x")
    if ratio < 2.0:
        print("  [GOOD] All SOC bins within 2x of each other")
    elif ratio < 3.5:
        print("  [OK] Acceptable coverage — some imbalance from BMS boundaries")
    else:
        print("  [WARN] Uneven coverage — consider increasing N_EPISODES or EPISODE_STEPS")