# ren/dataset_gen.py
"""
VRFB Dataset Generator — v4
============================

IMPROVEMENTS OVER v3
---------------------

1. Removed dVdt / dIdt columns
   These were computed and saved in every previous version but never appear
   in FEATURE_COLS. They inflated the CSV by ~14% (2 float64 columns out of
   14 total) and added 1.44M derivative calculations per run. Removed.

2. Vectorised temperature profile
   The old make_temperature_profile() ran a Python scalar loop with 18,000
   iterations per episode (1.44M iterations total). Replaced with a single
   numpy AR(1) cumulative-sum call — same random-walk statistics, ~50x faster.

3. Pre-allocated numpy row buffer
   The old code did 18,000 list.append() calls per episode, triggering
   repeated Python list reallocation. Each episode now pre-allocates a
   (EPISODE_STEPS, N_COLS) float32 array and fills it by index assignment.

4. Randomised train/test split
   The old split (episodes 0–63 train, 64–79 test) was deterministic by
   index. Because episodes are generated in round-robin band order, the
   test set was not representative of all SOC bands. Fixed: episode indices
   are shuffled before splitting, with a fixed seed for reproducibility.

5. Harder CC init offset distribution
   Old: uniform ±0.10. Real power-on SOC uncertainty ranges from small
   (BMS saved state) to large (battery off for days, self-discharge unknown).
   New: uniform ±0.15 with a 20% chance of drawing from ±[0.15, 0.25],
   creating occasional hard cases where the CC starts ~20% off. REN needs
   to see these to learn robust correction.

6. Standby profile added
   The BMS mode machine has a 30-min drain delay (STANDBY_DRAIN_DELAY=1800s).
   No v3 profile generated a standby period longer than a few hundred steps,
   so the drain flow rate (flow_min) never appeared in training data.
   New "standby_then_active" profile: I=0 for 1800–5400 steps (30–90 min),
   then transitions to charge or discharge. REN now sees the drain transient.

7. Increased N_EPISODES to 120 (96 train / 24 test)
   With 9 SOC bands, 80 episodes gave ~8-9 per band. The stateful BPTT
   trainer processes whole episodes, so a thin test set (16 episodes) gives
   noisy val metrics. 120 episodes gives ~13 per band and 24 test episodes.

DESIGN:
  N_EPISODES    = 120  (96 train / 24 test — shuffled split)
  EPISODE_STEPS = 18,000  (5 hours at dt=1s)
  Total rows    ≈ 2,160,000

REN FEATURES (9 inputs):
  voltage, current, temperature_stack, temperature_tank,
  flow_rate, soc_cc, I_limit, transport_ratio, soc_imbalance

TARGET (1):
  SOC_true  (negative-side bulk tank ratio — REN training label)
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

N_EPISODES    = 120
EPISODE_STEPS = 36000     # 10 hours at dt=1s  (covers live demo duration)
TRAIN_FRAC    = 0.80      # 96 train / 24 test
SEED          = 42
rng           = np.random.default_rng(SEED)

# Columns saved to CSV  (dVdt and dIdt removed — not used as REN features)
# Only physically measurable signals — no derived or competing estimator outputs.
# These are the 5 sensors standard in any deployed VRFB system.
FEATURE_COLS = [
    "voltage",            # stack terminal voltage [V]   — Nernst equation encodes SOC
    "current",            # stack current [A]            — sign: + discharge, - charge
    "temperature_stack",  # stack thermocouple [K]       — affects OCV via RT/nF term
    "temperature_tank",   # tank thermocouple [K]        — thermal lag, slow dynamics
    "flow_rate",          # electrolyte flow [m3/s]      — mass transport conditions
]
TARGET_COL = "SOC_true"

# Column order in the row buffer
# Extra columns saved for hybrid model training.
# NOT in FEATURE_COLS (standalone uses only 5 physical sensors).
# I_limit_approx and transport_ratio_approx are derived from
# measurable Q and I — no extra hardware required.
EXTRA_COLS = ["soc_cc", "I_limit_approx", "transport_ratio_approx","elapsed_time_norm"]
ROW_COLS = ["episode_id", "time"] + FEATURE_COLS + EXTRA_COLS + [TARGET_COL]

# I_limit approximation constants (from config.py)
_IL_CONST = 1 * 96485.0 * 2e-5 * 0.15 * 1600.0 * 0.5  # 231.6 A at Q_ref
_Q_REF    = 20.0 / 60000.0  # 20 LPM in m³/s

def approx_i_limit(Q_m3s: float) -> float:
    """I_limit from flow rate alone (fixed mid-SOC concentration)."""
    return _IL_CONST * (max(Q_m3s, 1e-6) / _Q_REF) ** 0.4

def approx_transport_ratio(I_A: float, Q_m3s: float) -> float:
    """Fraction of limiting current being drawn."""
    return abs(I_A) / max(approx_i_limit(Q_m3s), 1.0)
N_COLS   = len(ROW_COLS)

# SOC bands for stratification — 9 bands × ~13 episodes each
SOC_BANDS = [
    (0.05, 0.15),
    (0.15, 0.25),
    (0.25, 0.35),
    (0.35, 0.45),
    (0.45, 0.55),
    (0.55, 0.65),
    (0.65, 0.75),
    (0.75, 0.85),
    (0.85, 0.95),
]

# Weighted band allocation — 3x weight on extreme bands (0.05–0.15, 0.85–0.95)
# to compensate for the BMS cutting current near the limits, which naturally
# causes episodes to spend less time in the extreme SOC bins.
BAND_WEIGHTS = [3, 1, 1, 1, 1, 1, 1, 1, 3]
assert len(BAND_WEIGHTS) == len(SOC_BANDS)

_total_w    = sum(BAND_WEIGHTS)
_per_band   = [round(N_EPISODES * w / _total_w) for w in BAND_WEIGHTS]
_per_band[4] += N_EPISODES - sum(_per_band)    # absorb rounding error into mid band

episode_bands: list[tuple] = []
for band, n in zip(SOC_BANDS, _per_band):
    episode_bands.extend([band] * n)

# Shuffle so band order doesn't align with episode index
_rng_bands = np.random.default_rng(SEED + 2)
_rng_bands.shuffle(episode_bands)   # in-place shuffle

print("VRFB Dataset Generator  v4")
print(f"  Episodes      : {N_EPISODES}  ({int(N_EPISODES*TRAIN_FRAC)} train / "
      f"{N_EPISODES - int(N_EPISODES*TRAIN_FRAC)} test, shuffled split)")
print(f"  Steps/episode : {EPISODE_STEPS:,}  ({EPISODE_STEPS/3600:.1f} hours)")
print(f"  Total rows    : ~{N_EPISODES * EPISODE_STEPS:,}")
print(f"  Features      : {len(FEATURE_COLS)}  {FEATURE_COLS}")
print(f"  SOC bands     : {len(SOC_BANDS)} × ~{N_EPISODES//len(SOC_BANDS)} eps each")
print("-" * 60)


# =============================================================================
# PROFILE GENERATORS
# =============================================================================

def make_current_profile(profile_type: str, steps: int,
                         init_soc: float, rng) -> np.ndarray:
    """
    Returns an I_cmd array shaped (steps,).
    Profiles are matched to the episode's starting SOC.
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
        I    = np.zeros(steps)
        pos  = 0
        c_mag = rng.uniform(80, 130)
        d_mag = rng.uniform(80, 130)
        sign  = -1 if init_soc < 0.5 else 1
        while pos < steps:
            dur           = rng.integers(2000, 5000)
            I[pos:pos+dur] = sign * (c_mag if sign < 0 else d_mag)
            sign          *= -1
            pos           += dur
        return I

    elif profile_type == "step":
        I   = np.zeros(steps)
        pos = 0
        while pos < steps:
            dur           = rng.integers(500, 2000)
            I[pos:pos+dur] = rng.uniform(-160, 160)
            pos           += dur
        return I

    elif profile_type == "rest_then_active":
        I        = np.zeros(steps)
        rest_end = rng.integers(1000, 3000)
        mag      = rng.choice([-1, 1]) * rng.uniform(80, 140)
        I[rest_end:] = mag
        return I

    elif profile_type == "standby_then_active":
        # Long rest (>30 min) followed by charge or discharge.
        # Exposes the BMS drain-flow behaviour (STANDBY_DRAIN_DELAY=1800s)
        # so the REN learns to handle the flow-rate transient.
        I        = np.zeros(steps)
        # Standby lasts 1800–5400 s (30–90 min), ensuring drain activates
        rest_end = rng.integers(1800, min(5400, steps - 1000))
        mag      = rng.choice([-1, 1]) * rng.uniform(80, 140)
        I[rest_end:] = mag
        return I

    elif profile_type == "variable_rate":
        I      = np.zeros(steps)
        levels = rng.uniform(-150, 150, size=6)
        block  = steps // 6
        for k in range(6):
            s = k * block
            e = s + block
            if k > 0:
                I[s:s+200]  = np.linspace(levels[k-1], levels[k], 200)
                I[s+200:e]  = levels[k]
            else:
                I[s:e] = levels[k]
        return I

    else:   # random
        return rng.uniform(-150, 150, size=steps)


def make_flow_profile(profile_type: str, steps: int, cfg) -> np.ndarray:
    lpm = cfg.LPM_to_m3s
    if profile_type == "low":
        return np.full(steps, rng.uniform(8, 14) * lpm)
    elif profile_type == "high":
        return np.full(steps, rng.uniform(25, 45) * lpm)
    elif profile_type == "sweep_up":
        return np.linspace(rng.uniform(8, 15)*lpm, rng.uniform(25, 45)*lpm, steps)
    elif profile_type == "sweep_down":
        return np.linspace(rng.uniform(25, 45)*lpm, rng.uniform(8, 15)*lpm, steps)
    elif profile_type == "step":
        Q      = np.full(steps, cfg.initial_flow)
        levels = rng.uniform(8, 45, size=5) * lpm
        block  = steps // 5
        for k in range(5):
            Q[k*block:(k+1)*block] = levels[k]
        return Q
    else:   # nominal
        return np.full(steps, cfg.initial_flow)


def make_temperature_profile(steps: int, T_base: float, rng) -> np.ndarray:
    """
    Vectorised AR(1) temperature random walk, max rate 0.003 K/step.

    Implementation note: instead of a Python loop that clips each step
    individually, we generate all increments at once, compute the
    cumulative sum, then clip the entire trajectory to [290, 322] K.
    This gives the same statistical properties as the step-wise version
    while being ~50x faster.
    """
    increments = rng.uniform(-0.003, 0.003, size=steps)
    increments[0] = 0.0                        # first step stays at T_base
    T = T_base + np.cumsum(increments)
    T = np.clip(T, 290.0, 322.0)
    return T.astype(np.float32)


# =============================================================================
# PROFILE POOLS
# =============================================================================

CURRENT_PROFILES = [
    "discharge", "charge", "mixed_cycles", "step",
    "rest_then_active", "standby_then_active", "variable_rate", "random",
]
FLOW_PROFILES = ["low", "high", "sweep_up", "sweep_down", "step", "nominal"]


# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================

all_chunks: list[np.ndarray] = []   # collect per-episode arrays

for ep in tqdm(range(N_EPISODES), desc="Generating episodes"):

    # ── Stratified initial SOC ────────────────────────────────────────────
    band_lo, band_hi = episode_bands[ep]
    init_soc = float(np.clip(rng.uniform(band_lo, band_hi), 0.06, 0.94))

    # Match current profile to starting SOC
    if init_soc < 0.25:
        current_pool = ["charge", "mixed_cycles", "rest_then_active",
                        "standby_then_active"]
    elif init_soc > 0.75:
        current_pool = ["discharge", "mixed_cycles", "step",
                        "variable_rate", "standby_then_active"]
    else:
        current_pool = CURRENT_PROFILES

    current_type = rng.choice(current_pool)
    flow_type    = rng.choice(FLOW_PROFILES)
    if rng.random() < 0.20:
        T_base = rng.uniform(308.0, 322.0)   # hot operating regime
    else:
        T_base = rng.uniform(296.0, 303.0)   # normal start temperature

    # ── CC initial SOC error (Option A) ──────────────────────────────────
    # Base: uniform ±0.15.  20% chance of a harder outlier in ±[0.15, 0.25].
    if rng.random() < 0.20:
        sign           = rng.choice([-1, 1])
        cc_init_offset = sign * rng.uniform(0.15, 0.25)
    else:
        cc_init_offset = rng.uniform(-0.15, 0.15)
    soc_cc_init = float(np.clip(init_soc + cc_init_offset, 0.06, 0.94))

    # ── Current sensor DC bias (Option B) ────────────────────────────────
    current_bias = rng.uniform(-1.5, 1.5)

    # ── Instantiate per-episode objects ──────────────────────────────────
    cfg             = VRFBConfig()
    cfg.initial_soc = init_soc
    battery = VRFB(cfg)
    bms     = BMSController(cfg)
    sensor  = SensorModel(cfg)
    cc      = CoulombCounter(cfg)
    cc.initialize(soc_cc_init)

    # ── Inject pre-existing half-cell imbalance ───────────────────────────
    # Real deployed batteries accumulate half-cell imbalance over hundreds
    # of cycles due to asymmetric crossover.  A 5-hour episode starting
    # from perfectly balanced concentrations only produces ~0.001 imbalance
    # (verified from crossover rate constants).  We inject a random offset
    # at the start of each episode to simulate batteries at various stages
    # of their operational lifetime, giving soc_imbalance a genuine signal
    # range for the REN to train on.
    #
    # Distribution: 60% small ±0.05, 25% medium ±0.12, 15% large ±0.22
    # The sign is drawn independently, so half the episodes have neg > pos
    # and half have pos > neg — both directions appear in training data.
    _r = rng.random()
    if _r < 0.60:
        imb_half = rng.uniform(0.0, 0.05)
    elif _r < 0.85:
        imb_half = rng.uniform(0.05, 0.12)
    else:
        imb_half = rng.uniform(0.12, 0.22)
    imb_sign = rng.choice([-1.0, 1.0])
    imb_half *= imb_sign   # signed offset applied to neg side (+), pos side (-)

    # Resulting per-half-cell initial SOCs (symmetric around init_soc)
    soc_neg_0 = float(np.clip(init_soc + imb_half, 0.05, 0.95))
    soc_pos_0 = float(np.clip(init_soc - imb_half, 0.05, 0.95))

    # Directly modify the battery state vector.
    # State vector layout (from vrfb_core._initialize_state docstring):
    #   [0] C_V2_s   [1] C_V3_s   [2] C_VO2_s  [3] C_VO2plus_s  (stack)
    #   [4] C_V2_t   [5] C_V3_t   [6] C_VO2_t  [7] C_VO2plus_t  (tank)
    C = cfg.C_total
    # Stack (will equilibrate quickly during flow warmup)
    battery.state[0] = soc_neg_0 * C         # C_V2_s
    battery.state[1] = (1.0 - soc_neg_0) * C # C_V3_s
    battery.state[2] = (1.0 - soc_pos_0) * C # C_VO2_s
    battery.state[3] = soc_pos_0 * C         # C_VO2plus_s
    # Tank
    battery.state[4] = soc_neg_0 * C
    battery.state[5] = (1.0 - soc_neg_0) * C
    battery.state[6] = (1.0 - soc_pos_0) * C
    battery.state[7] = soc_pos_0 * C

    dt = cfg.dt_default

    # ── Pre-generate profiles ─────────────────────────────────────────────
    I_profile = make_current_profile(current_type, EPISODE_STEPS, init_soc, rng)
    Q_profile = make_flow_profile(flow_type, EPISODE_STEPS, cfg)
    T_profile = make_temperature_profile(EPISODE_STEPS, T_base, rng)

    # ── Flow warmup: settle to first commanded flow (200 steps, I=0) ─────
    Q_first = float(Q_profile[0])
    for _ in range(200):
        battery.step(0.0, Q_first, T_profile[0], dt)
    cc.initialize(soc_cc_init)   # reset CC after warmup (Option A)

    # ── Pre-allocate numpy row buffer ─────────────────────────────────────
    # Avoids 18,000 list.append() calls and repeated Python list reallocation.
    ep_data = np.empty((EPISODE_STEPS, N_COLS), dtype=np.float32)

    # Column index map for fast assignment
    IDX = {name: i for i, name in enumerate(ROW_COLS)}

    for step in range(EPISODE_STEPS):

        t     = step * dt
        I_cmd = float(I_profile[step])
        Q_cmd = float(Q_profile[step])
        T_amb = float(T_profile[step])

        out = battery.get_outputs()

        # ── Mode state machine ────────────────────────────────────────────
        bms.update_mode(I_cmd, out, dt)
        flow_override = bms.get_flow_override()
        Q_eff         = flow_override if flow_override is not None else Q_cmd
        I_safe        = bms.apply_protection(I_cmd, out)

        battery.step(I_safe, Q_eff, T_amb, dt)
        out      = battery.get_outputs()
        measured = sensor.measure(out)

        soc_cc = cc.update(
            measured_current = measured["current"],
            dt               = dt,
            Q_nominal        = out["capacity_nominal"],
        )

        # ── Fill row buffer ───────────────────────────────────────────────
        ep_data[step, IDX["episode_id"]]       = ep
        ep_data[step, IDX["time"]]             = t
        ep_data[step, IDX["voltage"]]           = measured["voltage"]
        ep_data[step, IDX["current"]]           = measured["current"]
        ep_data[step, IDX["temperature_stack"]] = measured["temperature"]
        ep_data[step, IDX["temperature_tank"]]  = measured["temperature_tank"]
        ep_data[step, IDX["flow_rate"]]         = measured["flow_rate"]
        ep_data[step, IDX["soc_cc"]]                     = float(soc_cc)
        ep_data[step, IDX["elapsed_time_norm"]] = step / EPISODE_STEPS  # 0→1 over episode
        _il   = approx_i_limit(measured["flow_rate"])
        _tr   = approx_transport_ratio(measured["current"], measured["flow_rate"])
        ep_data[step, IDX["I_limit_approx"]]             = _il
        ep_data[step, IDX["transport_ratio_approx"]]     = _tr
        ep_data[step, IDX["SOC_true"]]                   = out["soc_true"]

    all_chunks.append(ep_data)

# =============================================================================
# BUILD DATAFRAME
# =============================================================================

df = pd.DataFrame(
    np.concatenate(all_chunks, axis=0),
    columns=ROW_COLS,
)
# episode_id must be integer for groupby
df["episode_id"] = df["episode_id"].astype(int)

# =============================================================================
# RANDOMISED TRAIN / TEST SPLIT
# =============================================================================
# Shuffle episode indices before splitting so every SOC band is represented
# proportionally in both train and test sets.

all_ep_ids = df["episode_id"].unique()
rng_split  = np.random.default_rng(SEED + 1)   # separate seed for reproducibility
shuffled   = rng_split.permutation(all_ep_ids)

n_train      = int(N_EPISODES * TRAIN_FRAC)
train_ep_ids = set(shuffled[:n_train].tolist())
test_ep_ids  = set(shuffled[n_train:].tolist())

df_train = df[df["episode_id"].isin(train_ep_ids)].reset_index(drop=True)
df_test  = df[df["episode_id"].isin(test_ep_ids)].reset_index(drop=True)

# =============================================================================
# SAVE
# =============================================================================

df.to_csv("datasets/vrfb_dataset.csv",     index=False)
df_train.to_csv("datasets/vrfb_train.csv", index=False)
df_test.to_csv("datasets/vrfb_test.csv",   index=False)

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"Dataset generation complete  (v4)")
print(f"{'='*60}")
print(f"  Total rows    : {len(df):,}")
print(f"  Train rows    : {len(df_train):,}  ({len(train_ep_ids)} episodes)")
print(f"  Test rows     : {len(df_test):,}  ({len(test_ep_ids)} episodes)")
print(f"\n  SOC_true range : {df['SOC_true'].min():.3f} – {df['SOC_true'].max():.3f}")
print(f"  Voltage range  : {df['voltage'].min():.2f} – {df['voltage'].max():.2f} V")
print(f"  Current range  : {df['current'].min():.2f} – {df['current'].max():.2f} A")
print(f"  Temp range     : {df['temperature_stack'].min():.2f} – "
      f"{df['temperature_stack'].max():.2f} K")
print(f"  Flow range     : {df['flow_rate'].min()*60000:.1f} – "
      f"{df['flow_rate'].max()*60000:.1f} LPM")
# soc_imbalance is not a model feature (needs reference electrodes)

# ── CC baseline ───────────────────────────────────────────────────────────────
cc_errors = np.abs(df["SOC_true"].values - df["soc_cc"].values)
cc_rmse   = np.sqrt(np.mean(cc_errors**2))
cc_mae    = cc_errors.mean()
cc_max    = cc_errors.max()
print(f"\n  CC baseline errors (what REN must beat):")
print(f"    RMSE      : {cc_rmse:.4f}  ({cc_rmse*100:.2f}% SOC)")
print(f"    MAE       : {cc_mae:.4f}  ({cc_mae*100:.2f}% SOC)")
print(f"    Max error : {cc_max:.4f}  ({cc_max*100:.2f}% SOC)")
if cc_rmse < 0.02:
    print(f"  [WARN] CC RMSE too low — consider widening cc_init_offset further")
elif cc_rmse > 0.03:
    print(f"  [GOOD] CC RMSE gives REN a clear target to beat")

# ── SOC coverage histogram ────────────────────────────────────────────────────
print(f"\n  SOC coverage (target: all bins roughly equal):")
bins = np.linspace(0, 1, 11)
hist, _ = np.histogram(df["SOC_true"], bins=bins)
bar_max  = max(hist) if max(hist) > 0 else 1
for i in range(len(hist)):
    bar = "█" * int(hist[i] / bar_max * 30)
    pct = hist[i] / len(df) * 100
    print(f"    {bins[i]:.1f}–{bins[i+1]:.1f}  {bar:<30}  ({hist[i]:,})  {pct:.1f}%")

nonzero = hist[hist > 0]
if len(nonzero) > 1:
    ratio = nonzero.max() / nonzero.min()
    print(f"\n  Coverage uniformity: max/min ratio = {ratio:.2f}x")
    if ratio < 2.0:
        print("  [GOOD] All SOC bins within 2x of each other")
    elif ratio < 3.5:
        print("  [OK] Acceptable — some imbalance expected near BMS boundaries")
    else:
        print("  [WARN] Uneven coverage — consider increasing N_EPISODES")

# ── Profile distribution ──────────────────────────────────────────────────────
print(f"\n  Standby-then-active coverage:")
ep_ids_with_standby = []
# We can infer standby presence from long zero-current blocks
for ep_id, grp in df.groupby("episode_id"):
    zero_blocks = (grp["current"].abs() < 1.0).sum()
    if zero_blocks > 1800:
        ep_ids_with_standby.append(ep_id)
print(f"    Episodes with >1800 zero-current steps: {len(ep_ids_with_standby)} "
      f"/ {N_EPISODES}  ({100*len(ep_ids_with_standby)/N_EPISODES:.0f}%)")