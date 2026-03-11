# config.py

class VRFBConfig:
    def __init__(self):

        # ============================================================
        # 1. UNIVERSAL CONSTANTS
        # ============================================================
        self.F = 96485.0              # Faraday constant (C/mol)
        self.R = 8.314                # Gas constant (J/mol·K)
        self.n = 1                    # Electrons transferred per vanadium ion
        self.alpha = 0.5              # Charge transfer coefficient (Butler-Volmer)

        # ============================================================
        # 2. STACK ARCHITECTURE (Lab-Scale 5–10 kW)
        # ============================================================
        self.N_cells = 40
        self.electrode_area = 0.15    # m² per cell (1500 cm²)

        # Electrolyte volumes
        self.V_stack = 0.003          # m³ (3 litres in stack channels)
        self.V_tank  = 0.05           # m³ (50 litres per half-tank)

        # ============================================================
        # 3. ELECTROLYTE PROPERTIES
        # ============================================================
        # FIX: E0 reduced from 1.40 V to 1.26 V (standard cell potential
        # V²⁺/V³⁺ vs VO²⁺/VO₂⁺).  The full Nernst equation for VRFB is:
        #   E = E0 + (RT/F)·ln([VO₂⁺][V²⁺][H⁺]² / [VO²⁺][V³⁺])
        # Because we do not explicitly track [H⁺], using E0 = 1.26 V
        # keeps the mid-SOC OCV accurate without the H⁺ correction term.
        # Using 1.40 V without the H⁺ term causes systematic OCV over-
        # prediction of ~0.1–0.15 V across the SOC range.
        self.E0 = 1.26                # V  (corrected from 1.40)

        self.C_total = 1600.0         # mol/m³  (≈1.6 M vanadium)

        # Diffusion / mass transfer
        self.diffusivity           = 3e-10   # m²/s
        self.k_mass_transfer_coeff = 2e-5    # m/s  (baseline at reference flow)

        # Exchange current density
        self.i0 = 0.5                 # A/m²

        # ============================================================
        # 4. ELECTRICAL PARAMETERS
        # ============================================================
        self.R_membrane_initial = 0.0015   # Ω per cell (Nafion typical)
        self.R_contact          = 0.0005   # Ω per cell (contacts + busbars)

        # ============================================================
        # 5. THERMAL PARAMETERS
        # ============================================================
        self.C_th_stack = 8000.0      # J/K  — stack thermal mass (correct)

        # FIX: C_th_tank corrected from 25,000 J/K to 227,500 J/K.
        # 50 L electrolyte: density ≈ 1300 kg/m³ → mass = 65 kg
        # Specific heat ≈ 3500 J/(kg·K) → C_th = 65 × 3500 = 227,500 J/K
        # Previous value caused tank to heat 9× too fast.
        self.C_th_tank = 227500.0     # J/K  (corrected from 25,000)

        self.h_stack_tank    = 60.0   # W/K  (convective coupling, stack → tank)
        self.h_tank_ambient  = 25.0   # W/K  (tank → ambient)

        self.initial_temperature = 298.15   # K  (25 °C)

        # ============================================================
        # 6. PUMP & FLOW PARAMETERS
        # ============================================================
        self.tau_pump = 2.0           # s  (pump first-order inertia)

        # Conversion factor (used everywhere — defined once)
        self.LPM_to_m3s = 1.0 / 60000.0   # 1 LPM = 1.6667e-5 m³/s

        self.initial_flow = 20.0 * self.LPM_to_m3s   # 20 LPM in m³/s

        self.flow_min = 5.0  * self.LPM_to_m3s        # 5  LPM
        self.flow_max = 60.0 * self.LPM_to_m3s        # 60 LPM

        # Pump hydraulic power coefficient  P_pump = k · Q³
        self.pump_power_coeff = 8e8   # W/(m³/s)³  (tuned for realistic ~50–200 W)

        # ============================================================
        # 7. DEGRADATION PARAMETERS
        # ============================================================
        # FIX: k_membrane_aging reduced from 2e-10 to 2e-12 Ω/(s·A).
        # Original caused resistance to quadruple over an 83-hour run,
        # which is years-worth of degradation in a single simulation.
        # At 2e-12: 100 A × 300,000 s → +0.06 mΩ/cell added (realistic).
        self.k_membrane_aging = 2e-12   # Ω/(s·A)  (corrected from 2e-10)

        # FIX: k_capacity_fade increased from 5e-11 to 5e-8 Ah/(s·A).
        # Original caused <0.001% capacity loss — completely invisible.
        # At 5e-8: 100 A × 300,000 s → ~1.5 Ah lost from ~2146 Ah (0.07%)
        # Visible but not catastrophic over one simulation run.
        self.k_capacity_fade  = 5e-8    # Ah/(s·A)  (corrected from 5e-11)

        # ============================================================
        # 8. INITIAL SOC & CAPACITY
        # ============================================================
        self.initial_soc = 0.50

        # Theoretical capacity: Q = n·F·C_total·V_tank / 3600  [Ah]
        self.theoretical_capacity_Ah = (
            self.n * self.F * self.C_total * self.V_tank / 3600.0
        )
        self.initial_Q_nominal = self.theoretical_capacity_Ah

        # Operating current limits
        self.I_max =  200.0    # A  (discharge)
        self.I_min = -200.0    # A  (charge)

        # SOC operating window
        self.soc_min = 0.05
        self.soc_max = 0.95

        # Coulombic efficiency (applied during charging in ODE)
        self.coulombic_efficiency = 0.98

        # ============================================================
        # 9. SENSOR NOISE (standard deviations)
        # ============================================================
        self.voltage_noise_std     = 0.02   # V
        self.current_noise_std     = 0.20   # A
        self.temperature_noise_std = 0.30   # K

        # ============================================================
        # 10. MEMBRANE CROSSOVER PARAMETERS
        # ============================================================
        # FIX: I_crossover_ref increased from 0.05 A to 1.0 A.
        # Published lab-scale VRFB data (1500 cm² stacks) shows crossover
        # equivalent currents of 1–5 A. At 0.05 A the capacity imbalance
        # effect was invisible over an 83-hour simulation.
        self.I_crossover_ref  = 1.0    # A  (corrected from 0.05)
        self.crossover_beta   = 0.03   # 1/K  (temperature sensitivity)
        # FIX: Asymmetric crossover factor for positive half-cell.
        # VO₂⁺ has ~3× higher diffusion resistance through Nafion than V²⁺
        # (larger ionic radius, lower diffusion coefficient ~1e-11 vs 3.5e-11 m²/s).
        # Without this, both half-cells change at the same rate → imbalance = 0 always.
        self.crossover_pos_factor = 0.3  # dimensionless (VO₂⁺ / V²⁺ diffusion ratio)
        self.T_ref            = 298.15 # K   (reference temperature)

        # ============================================================
        # 11. SIMULATION SETTINGS
        # ============================================================
        self.dt_default = 1.0          # s

        # ============================================================
        # 12. BMS PROTECTION LIMITS
        # ============================================================
        # Voltage window for 40-cell stack
        self.V_min_stack = 35.0        # V  (0.875 V/cell)
        self.V_max_stack = 70.0        # V  (1.750 V/cell)

        # Thermal thresholds
        self.T_max    = 330.0          # K  (~57 °C hard shutdown)
        self.T_derate = 320.0          # K  (~47 °C start derating)

        # Limiting-current safety margin (fraction of I_limit allowed)
        self.limiting_current_margin = 0.7

        # Minimum safe flow (below this → protective current derating)
        self.flow_critical = 8.0 * self.LPM_to_m3s   # 8 LPM

        # ============================================================
        # 13. CONVERTER DYNAMICS
        # ============================================================
        self.tau_converter = 0.2       # s  (first-order current control lag)