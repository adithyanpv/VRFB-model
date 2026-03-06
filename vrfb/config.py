# config.py
#import numpy as np


class VRFBConfig:
    def __init__(self):

        # ============================================================
        # 1. UNIVERSAL CONSTANTS
        # ============================================================
        self.F = 96485.0              # Faraday constant (C/mol)
        self.R = 8.314                # Gas constant (J/mol-K)
        self.n = 1                    # Electrons transferred
        self.alpha = 0.5              # Charge transfer coefficient

        # ============================================================
        # 2. STACK ARCHITECTURE (Lab-Scale 5–10 kW)
        # ============================================================
        self.N_cells = 40
        self.electrode_area = 0.15    # m² per cell (1500 cm² realistic lab stack)

        # Electrolyte volumes
        self.V_stack = 0.003         # m³ (3 liters in stack channels)
        self.V_tank = 0.05           # m³ (50 liters per tank)

        # ============================================================
        # 3. ELECTROLYTE PROPERTIES
        # ============================================================
        self.C_total = 1600.0        # mol/m³ (≈1.6 M vanadium)
        self.E0 = 1.40               # Standard cell potential (V)

        # Diffusion / mass transfer
        self.diffusivity = 3e-10     # m²/s (typical vanadium ion diffusivity)
        self.k_mass_transfer_coeff = 2e-5  # baseline mass transfer coeff (m/s)

        # Exchange current density (realistic order)
        self.i0 = 0.5                # A/m² (moderate kinetics)

        # ============================================================
        # 4. ELECTRICAL PARAMETERS
        # ============================================================
        self.R_membrane_initial = 0.0015   # Ohm per cell
        self.R_contact = 0.0005            # Ohm per cell (contacts + busbars)

        # ============================================================
        # 5. THERMAL PARAMETERS
        # ============================================================
        self.C_th_stack = 8000.0     # J/K (stack thermal capacitance)
        self.C_th_tank = 25000.0     # J/K (tank thermal capacitance)

        self.h_stack_tank = 60.0     # W/K (heat transfer stack → tank)
        self.h_tank_ambient = 25.0   # W/K (tank → ambient cooling)

        self.initial_temperature = 298.15  # K

        # ============================================================
        # 6. PUMP & FLOW PARAMETERS
        # ============================================================
        self.tau_pump = 2.0          # s (pump inertia time constant)

        # Typical operating flow 10–60 LPM
        # Convert LPM → m³/s: 1 LPM = 1.6667e-5 m³/s
        self.initial_flow = 20.0 * 1.6667e-5   # 20 LPM

        # Pump power ~ k * Q³
        self.pump_power_coeff = 8e8  # tuned for realistic pump power (W)

        # ============================================================
        # 7. DEGRADATION PARAMETERS (Slow Timescale)
        # ============================================================
        self.k_membrane_aging = 2e-10   # Ohm/s per Amp magnitude
        self.k_capacity_fade = 5e-11    # Ah/s per Amp magnitude

        # ============================================================
        # 8. INITIAL SOC & CAPACITY
        # ============================================================
        self.initial_soc = 0.50

        # Theoretical capacity from chemistry
        # Q = n F C V / 3600  (Ah)
        self.theoretical_capacity_Ah = (
            self.n * self.F * self.C_total * self.V_tank / 3600.0
        )

        self.initial_Q_nominal = self.theoretical_capacity_Ah

        # Operating current limits
        self.I_max = 200.0     # A
        self.I_min = -200.0    # A

        #SOC min and max
        self.soc_min = 0.05
        self.soc_max = 0.95

        #Flow conversion
        self.LPM_to_m3s = 1.0 / 60000.0
        self.initial_flow = 20.0 * self.LPM_to_m3s

        #Flow limits
        self.flow_min = 5.0 * self.LPM_to_m3s
        self.flow_max = 60.0 * self.LPM_to_m3s

        self.coulombic_efficiency = 0.98

        #Sensor noise
        self.voltage_noise_std = 0.02
        self.current_noise_std = 0.2
        self.temperature_noise_std = 0.3
        # 10. MEMBRANE CROSSOVER PARAMETERS
        # ============================================================

        self.I_crossover_ref = 0.05      # A at reference temperature
        self.crossover_beta = 0.03       # Temperature sensitivity (1/K)
        self.T_ref = 298.15              # Reference temperature (K)

        # ============================================================
        # 9. SIMULATION SETTINGS
        # ============================================================
        self.dt_default = 1.0        # seconds

        # ============================================================
        # 11. BMS PROTECTION LIMITS
        # ============================================================

        # Voltage limits (40-cell stack)
        self.V_min_stack = 35.0     # V  (≈0.9 V per cell)
        self.V_max_stack = 70.0     # V  (≈1.75 V per cell)

        # Temperature limit
        self.T_max = 330.0          # K (≈57°C shutdown)
        self.T_derate = 320.0       # K start derating

        # Current derating margin
        self.limiting_current_margin = 0.7

        # Flow protection threshold
        self.flow_critical = 8.0 * self.LPM_to_m3s  # below this is dangerous
        # 12. CONVERTER DYNAMICS
# ============================================================

        self.tau_converter = 0.2   # seconds (current control response time)