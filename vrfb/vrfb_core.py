# vrfb_core.py

import numpy as np
from scipy.integrate import solve_ivp


class VRFB:

    def __init__(self, config):
        self.cfg   = config
        self.state = self._initialize_state()
        self.time  = 0.0

    # ============================================================
    # INITIAL STATE
    # ============================================================
    def _initialize_state(self):
        """
        State vector (14 elements):
          [0]  C_V2_s      — V²⁺  concentration in stack   (mol/m³)
          [1]  C_V3_s      — V³⁺  concentration in stack   (mol/m³)
          [2]  C_VO2_s     — VO²⁺ concentration in stack   (mol/m³)
          [3]  C_VO2plus_s — VO₂⁺ concentration in stack   (mol/m³)
          [4]  C_V2_t      — V²⁺  concentration in tank    (mol/m³)
          [5]  C_V3_t      — V³⁺  concentration in tank    (mol/m³)
          [6]  C_VO2_t     — VO²⁺ concentration in tank    (mol/m³)
          [7]  C_VO2plus_t — VO₂⁺ concentration in tank    (mol/m³)
          [8]  T_s         — Stack temperature              (K)
          [9]  T_t         — Tank  temperature              (K)
          [10] Q           — Volumetric flow rate           (m³/s)
          [11] R_m         — Membrane resistance per cell   (Ω)
          [12] Q_nom       — Nominal capacity               (Ah)
          [13] I_actual    — Actual current through stack   (A)
        """
        soc   = self.cfg.initial_soc
        C_tot = self.cfg.C_total

        # Half-cell concentrations from SOC
        # Negative side: V²⁺ increases with SOC, V³⁺ decreases
        C_V2_t      = soc       * C_tot
        C_V3_t      = (1 - soc) * C_tot
        # Positive side: VO₂⁺ increases with SOC, VO²⁺ decreases
        C_VO2_t     = (1 - soc) * C_tot
        C_VO2plus_t = soc       * C_tot

        # Stack starts at same composition as tank (no electrolyte gradient yet)
        return np.array([
            C_V2_t, C_V3_t, C_VO2_t, C_VO2plus_t,   # Stack concentrations [0–3]
            C_V2_t, C_V3_t, C_VO2_t, C_VO2plus_t,   # Tank  concentrations [4–7]
            self.cfg.initial_temperature,             # T_stack  [8]
            self.cfg.initial_temperature,             # T_tank   [9]
            self.cfg.initial_flow,                    # Flow     [10]
            self.cfg.R_membrane_initial,              # R_mem    [11]
            self.cfg.initial_Q_nominal,               # Q_nom    [12]
            0.0,                                      # I_actual [13]  (starts at rest)
        ], dtype=float)

    # ============================================================
    # ODE RIGHT-HAND SIDE
    # ============================================================
    def _derivatives(self, _, x, I, Q_cmd, T_amb):

        cfg = self.cfg

        (
            C_V2_s, C_V3_s, C_VO2_s, C_VO2plus_s,
            C_V2_t, C_V3_t, C_VO2_t, C_VO2plus_t,
            T_s, T_t,
            Q, R_m, Q_nom, I_actual
        ) = x

        # ----------------------------------------------------------------
        # COULOMBIC EFFICIENCY
        # Applied only during charging (I_actual < 0):
        # only eta_c fraction of charge current causes useful SOC change.
        # During discharge (I_actual ≥ 0), efficiency = 1.0 (loss modelled
        # separately via ohmic / concentration overpotentials in voltage).
        # ----------------------------------------------------------------
        eta_c = cfg.coulombic_efficiency if I_actual < 0 else 1.0

        # Effective electrochemical current for concentration change
        I_eff = eta_c * I_actual

        # ----------------------------------------------------------------
        # STACK SPECIES DYNAMICS  (mol/m³/s)
        # Convective exchange with tank  +  electrochemical reaction term
        #
        # Discharge (I > 0):  V²⁺ → V³⁺  (negative side)
        #                     VO²⁺ → VO₂⁺  (positive side)
        # Charge    (I < 0):  reverse
        # ----------------------------------------------------------------
        dC_V2_s      = (Q / cfg.V_stack) * (C_V2_t      - C_V2_s)      - I_eff / (cfg.n * cfg.F * cfg.V_stack)
        dC_V3_s      = (Q / cfg.V_stack) * (C_V3_t      - C_V3_s)      + I_eff / (cfg.n * cfg.F * cfg.V_stack)
        dC_VO2_s     = (Q / cfg.V_stack) * (C_VO2_t     - C_VO2_s)     + I_eff / (cfg.n * cfg.F * cfg.V_stack)
        dC_VO2plus_s = (Q / cfg.V_stack) * (C_VO2plus_t - C_VO2plus_s) - I_eff / (cfg.n * cfg.F * cfg.V_stack)

        # ----------------------------------------------------------------
        # TANK SPECIES DYNAMICS  (mol/m³/s)
        # Tank receives the electrolyte that left the stack.
        # ----------------------------------------------------------------
        dC_V2_t      = (Q / cfg.V_tank) * (C_V2_s      - C_V2_t)
        dC_V3_t      = (Q / cfg.V_tank) * (C_V3_s      - C_V3_t)
        dC_VO2_t     = (Q / cfg.V_tank) * (C_VO2_s     - C_VO2_t)
        dC_VO2plus_t = (Q / cfg.V_tank) * (C_VO2plus_s - C_VO2plus_t)

        # ----------------------------------------------------------------
        # MEMBRANE CROSSOVER  (temperature + SOC dependent)
        #
        # V²⁺ from the negative side diffuses across the membrane into the
        # positive side and is immediately oxidised by VO₂⁺:
        #   V²⁺  +  VO₂⁺  →  V³⁺  +  VO²⁺   (spontaneous)
        #
        # Net effect on tank concentrations:
        #   Negative side: V²⁺ ↓,  V³⁺ ↑
        #   Positive side: VO₂⁺ ↓, VO²⁺ ↑    ← FIX: was missing
        # ----------------------------------------------------------------
        # FIX: Asymmetric crossover — each half-cell driven by its own SOC
        # Physical basis: V²⁺ and VO₂⁺ have different diffusion coefficients
        # through Nafion membrane (V²⁺ ~3x faster than VO₂⁺ due to ionic radius).
        # Using the same rate for both sides made imbalance = 0 by symmetry.
        #
        # crossover_pos_factor = 0.3  (VO₂⁺ diffusion ~30% of V²⁺ rate)
        # This creates growing half-cell SOC divergence over repeated cycles.

        soc_neg = C_V2_t      / (C_V2_t      + C_V3_t      + 1e-12)
        soc_pos = C_VO2plus_t / (C_VO2plus_t + C_VO2_t      + 1e-12)

        T_factor = np.exp(cfg.crossover_beta * (T_s - cfg.T_ref))

        I_cross_neg = (
            cfg.I_crossover_ref * T_factor
            * soc_neg * (1.0 - soc_neg)
        )
        I_cross_pos = (
            cfg.I_crossover_ref * cfg.crossover_pos_factor * T_factor
            * soc_pos * (1.0 - soc_pos)
        )

        crossover_rate_neg = I_cross_neg / (cfg.n * cfg.F * cfg.V_tank)
        crossover_rate_pos = I_cross_pos / (cfg.n * cfg.F * cfg.V_tank)

        # Negative side: V²⁺ consumed by crossover
        dC_V2_t -= crossover_rate_neg
        dC_V3_t += crossover_rate_neg

        # Positive side: VO₂⁺ consumed by crossover (slower — larger ion)
        dC_VO2plus_t -= crossover_rate_pos
        dC_VO2_t     += crossover_rate_pos

        # ----------------------------------------------------------------
        # THERMAL MODEL
        # Q_joule = I²·R_stack (Joule heating in stack)
        # Stack → Tank → Ambient cascade
        # ----------------------------------------------------------------
        R_stack         = (R_m + cfg.R_contact) * cfg.N_cells
        heat_generation = I_actual ** 2 * R_stack

        dT_s = (heat_generation       - cfg.h_stack_tank  * (T_s - T_t)) / cfg.C_th_stack
        dT_t = (cfg.h_stack_tank * (T_s - T_t) - cfg.h_tank_ambient * (T_t - T_amb)) / cfg.C_th_tank

        # ----------------------------------------------------------------
        # PUMP DYNAMICS  (first-order lag to commanded flow)
        # ----------------------------------------------------------------
        dQ = (Q_cmd - Q) / cfg.tau_pump

        # ----------------------------------------------------------------
        # DEGRADATION (slow timescale)
        # ----------------------------------------------------------------
        dR_m   =  cfg.k_membrane_aging * abs(I_actual)   # Ω/s
        dQ_nom = -cfg.k_capacity_fade  * abs(I_actual)   # Ah/s

        # ----------------------------------------------------------------
        # CONVERTER CURRENT DYNAMICS  (first-order lag to commanded current)
        # ----------------------------------------------------------------
        dI_actual = (I - I_actual) / cfg.tau_converter

        return [
            dC_V2_s, dC_V3_s, dC_VO2_s, dC_VO2plus_s,
            dC_V2_t, dC_V3_t, dC_VO2_t, dC_VO2plus_t,
            dT_s, dT_t,
            dQ, dR_m, dQ_nom,
            dI_actual
        ]

    # ============================================================
    # STEP  —  advance simulation by dt seconds
    # ============================================================
    def step(self, I, Q_cmd, T_amb, dt):

        cfg = self.cfg

        # Hard-clip inputs to physical limits before ODE
        I     = np.clip(I,     cfg.I_min,    cfg.I_max)
        Q_cmd = np.clip(Q_cmd, cfg.flow_min, cfg.flow_max)

        sol = solve_ivp(
            fun     = lambda t, x: self._derivatives(t, x, I, Q_cmd, T_amb),
            t_span  = (0.0, dt),
            y0      = self.state,
            method  = "BDF",        # stiff solver — correct for this system
            rtol    = 1e-6,
            atol    = 1e-9,
        )

        self.state = sol.y[:, -1]

        # Prevent negative concentrations (numerical floor)
        self.state[:8] = np.clip(self.state[:8], 1e-12, None)

        self.time += dt

    # ============================================================
    # OUTPUT PORTS
    # ============================================================
    def get_outputs(self):

        cfg = self.cfg

        (
            C_V2_s, C_V3_s, C_VO2_s, C_VO2plus_s,
            C_V2_t, C_V3_t, _,       _,
            T_s,    T_t,
            Q,      R_m,    Q_nom,    I_actual
        ) = self.state

        # ── True SOC  (bulk tank, negative side) ─────────────────────────
        soc = np.clip(
            C_V2_t / (C_V2_t + C_V3_t + 1e-12),
            0.0, 1.0
        )

        # ── Open-Circuit Voltage (Nernst, per cell) ───────────────────────
        # Full VRFB Nernst: E = E0 + (RT/F)·ln([VO₂⁺][V²⁺] / [VO²⁺][V³⁺])
        # [H⁺]² term is absorbed into the calibrated E0 = 1.26 V
        eps = 1e-12
        num = max(C_VO2plus_s * C_V2_s,  eps)
        den = max(C_VO2_s     * C_V3_s,  eps)
        E   = cfg.E0 + (cfg.R * T_s / cfg.F) * np.log(num / den)

        # ── Ohmic Loss (stack level) ──────────────────────────────────────
        R_stack = (R_m + cfg.R_contact) * cfg.N_cells
        V_ohmic = R_stack * I_actual

        # ── Active Concentration for Mass Transport ───────────────────────
        # Limiting species depends on direction:
        #   Discharge (I ≥ 0): V²⁺ (neg) and VO₂⁺ (pos) are consumed
        #   Charge    (I < 0): V³⁺ (neg) and VO²⁺ (pos) are consumed
        if I_actual >= 0:
            C_active = min(C_V2_s, C_VO2plus_s)
        else:
            C_active = min(C_V3_s, C_VO2_s)

        # ── Flow-Dependent Mass Transfer Coefficient ──────────────────────
        Q_ref = cfg.initial_flow
        k_m   = cfg.k_mass_transfer_coeff * (Q / Q_ref) ** 0.4
        A     = cfg.electrode_area

        # ── Limiting Current ──────────────────────────────────────────────
        I_limit = cfg.n * cfg.F * k_m * A * C_active

        # ── Concentration Overpotential (per cell) ────────────────────────
        ratio = abs(I_actual) / (I_limit + 1e-9)
        ratio = np.clip(ratio, 0.0, 0.999)
        # V_conc = RT/nF · ln(1 − |I|/I_lim)  — always negative (loss)
        V_conc = (cfg.R * T_s / (cfg.n * cfg.F)) * np.log(1.0 - ratio)

        # ── Cell Voltage ──────────────────────────────────────────────────
        # Discharge: V_cell = E + V_conc      (V_conc < 0 → reduces output)
        # Charge:    V_cell = E − V_conc      (V_conc < 0 → −V_conc > 0 → raises demand)
        if I_actual >= 0:
            V_cell = E + V_conc
        else:
            V_cell = E - V_conc

        # Stack voltage: Nernst and concentration terms are per-cell,
        # ohmic loss is already stack-level
        voltage = (cfg.N_cells * V_cell) - V_ohmic

        # ── Transport Ratio (diagnostic) ─────────────────────────────────
        transport_ratio = abs(I_actual) / (I_limit + 1e-9)

        # ── Pump Power ────────────────────────────────────────────────────
        pump_power = cfg.pump_power_coeff * Q ** 3

        return {
            "time"               : self.time,
            "soc_true"           : soc,
            "voltage_stack"      : voltage,
            "temperature_stack"  : T_s,
            "temperature_tank"   : T_t,
            "flow_rate"          : Q,
            "membrane_resistance": R_m,
            "capacity_nominal"   : Q_nom,
            "pump_power"         : pump_power,
            "transport_ratio"    : transport_ratio,
            "i_limit"            : I_limit,
            "current"            : I_actual,
        }