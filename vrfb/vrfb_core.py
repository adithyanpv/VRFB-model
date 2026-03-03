# vrfb_core.py

import numpy as np
from scipy.integrate import solve_ivp


class VRFB:

    def __init__(self, config):
        self.cfg = config
        self.state = self._initialize_state()
        self.time = 0.0
        

    # ============================================================
    # INITIAL STATE
    # ============================================================
    def _initialize_state(self):

        soc = self.cfg.initial_soc
        C_tot = self.cfg.C_total

        # Tank concentrations from SOC
        C_V2_t = soc * C_tot
        C_V3_t = (1 - soc) * C_tot
        C_VO2_t = (1 - soc) * C_tot
        C_VO2plus_t = soc * C_tot

        # Stack initially equal to tank
        return np.array([
            C_V2_t, C_V3_t, C_VO2_t, C_VO2plus_t,   # Stack
            C_V2_t, C_V3_t, C_VO2_t, C_VO2plus_t,   # Tank
            self.cfg.initial_temperature,            # T_stack
            self.cfg.initial_temperature,            # T_tank
            self.cfg.initial_flow,                   # Flow
            self.cfg.R_membrane_initial,             # R_membrane
            self.cfg.initial_Q_nominal,
            0.0               # Q_nominal
        ], dtype=float)

    # ============================================================
    # ODE SYSTEM
    # ============================================================
    def _derivatives(self, _, x, I, Q_cmd, T_amb):

        cfg = self.cfg

        (
            C_V2_s, C_V3_s, C_VO2_s, C_VO2plus_s,
            C_V2_t, C_V3_t, C_VO2_t, C_VO2plus_t,
            T_s, T_t,
            Q, R_m, Q_nom,I_actual
        ) = x

        # ---------------------------
        # Stack species dynamics
        # ---------------------------
        dC_V2_s = (Q/cfg.V_stack)*(C_V2_t - C_V2_s) - I_actual/(cfg.n*cfg.F*cfg.V_stack)
        dC_V3_s = (Q/cfg.V_stack)*(C_V3_t - C_V3_s) + I_actual/(cfg.n*cfg.F*cfg.V_stack)

        dC_VO2_s = (Q/cfg.V_stack)*(C_VO2_t - C_VO2_s) + I_actual/(cfg.n*cfg.F*cfg.V_stack)
        dC_VO2plus_s = (Q/cfg.V_stack)*(C_VO2plus_t - C_VO2plus_s) - I_actual/(cfg.n*cfg.F*cfg.V_stack)

        # ---------------------------
        # Tank dynamics
        # ---------------------------
        dC_V2_t = (Q/cfg.V_tank)*(C_V2_s - C_V2_t)
        dC_V3_t = (Q/cfg.V_tank)*(C_V3_s - C_V3_t)

        dC_VO2_t = (Q/cfg.V_tank)*(C_VO2_s - C_VO2_t)
        dC_VO2plus_t = (Q/cfg.V_tank)*(C_VO2plus_s - C_VO2plus_t)

        # ---------------------------
        # MEMBRANE CROSSOVER (Temperature + SOC Dependent)
        # ---------------------------

        # Tank SOC (negative side)
        soc = C_V2_t / (C_V2_t + C_V3_t + 1e-12)

        # Temperature-dependent crossover current
        I_cross = (
            cfg.I_crossover_ref
            * np.exp(cfg.crossover_beta * (T_s - cfg.T_ref))
            * soc * (1 - soc)
        )

        # Convert crossover current to concentration rate
        crossover_rate = I_cross / (cfg.n * cfg.F * cfg.V_tank)

        # Internal redox short-circuit effect
        dC_V2_t -= crossover_rate
        dC_V3_t += crossover_rate

        # ---------------------------
        # Thermal model
        # ---------------------------
        R_stack = (R_m + cfg.R_contact) * cfg.N_cells
        heat_generation = I_actual**2 * R_stack

        dT_s = (heat_generation - cfg.h_stack_tank*(T_s - T_t)) / cfg.C_th_stack
        dT_t = (cfg.h_stack_tank*(T_s - T_t)
                - cfg.h_tank_ambient*(T_t - T_amb)) / cfg.C_th_tank

        # ---------------------------
        # Pump dynamics
        # ---------------------------
        dQ = (Q_cmd - Q) / cfg.tau_pump

        # ---------------------------
        # Aging
        # ---------------------------
        dR_m = cfg.k_membrane_aging * abs(I_actual)
        dQ_nom = -cfg.k_capacity_fade * abs(I_actual)
        dI_actual = (I - I_actual) / cfg.tau_converter

        return [
            dC_V2_s, dC_V3_s, dC_VO2_s, dC_VO2plus_s,
            dC_V2_t, dC_V3_t, dC_VO2_t, dC_VO2plus_t,
            dT_s, dT_t,
            dQ, dR_m, dQ_nom,
            dI_actual
        ]
    

    # ============================================================
    # STEP SIMULATION
    # ============================================================
    def step(self, I, Q_cmd, T_amb, dt):

        cfg = self.cfg

        # 1️⃣ Enforce operating limits
        I = np.clip(I, cfg.I_min, cfg.I_max)
        Q_cmd = np.clip(Q_cmd, cfg.flow_min, cfg.flow_max)


        # 2️⃣ Solve ODE
        sol = solve_ivp(
            lambda t, x: self._derivatives(t, x, I, Q_cmd, T_amb),
            [0, dt],
            self.state,
            method="BDF"
        )
        self.state = sol.y[:, -1]

        # 3️⃣ Numerical stability
        self.state[:8] = np.clip(self.state[:8], 1e-12, None)
        self.time += dt

    # ============================================================
    # OUTPUT PORTS
    # ============================================================
    def get_outputs(self):

        cfg = self.cfg

        (
            C_V2_s, C_V3_s, C_VO2_s, C_VO2plus_s,
            C_V2_t, C_V3_t, _, _,
            T_s, T_t,
            Q, R_m, Q_nom,I_actual
        ) = self.state

        # True SOC
        soc = np.clip(
            C_V2_t / (C_V2_t + C_V3_t + 1e-12),
            0.0, 1.0
        )

        # Safe Nernst
        eps = 1e-12
        num = max(C_VO2plus_s * C_V2_s, eps)
        den = max(C_VO2_s * C_V3_s, eps)

        E = cfg.E0 + (cfg.R*T_s/cfg.F) * np.log(num / den)

        # Ohmic loss
        R_stack = (R_m + cfg.R_contact) * cfg.N_cells
        V_ohmic = R_stack * I_actual

        # ------------------------------------------------------------
        # MASS TRANSPORT (CONCENTRATION) OVERPOTENTIAL - DO NOT REMOVE
        # ------------------------------------------------------------
        if I_actual >= 0: # Discharging
            C_active = min(C_V2_s, C_VO2plus_s)
        else:                      # Charging
            C_active = min(C_V3_s, C_VO2_s)
            
        # Limiting Current based on Flow Rate (Q)
        I_limit = cfg.n * cfg.F * max(Q, 1e-7) * C_active
        
        # Ratio of actual current to limiting current (clamped for math safety)
        ratio = abs(I_actual) / (I_limit + 1e-9)
        ratio = np.clip(ratio, 0.0, 0.999)
        
        # V_conc calculation (This will always be a negative number)
        V_conc = (cfg.R * T_s / (cfg.n * cfg.F)) * np.log(1 - ratio)

        # Final Cell Voltage
        # If discharging (I > 0), subtract V_conc. If charging (I < 0), subtract V_conc to push voltage higher.
        if I_actual >= 0:
            V_cell = E + V_conc  # V_conc is negative, so adding it lowers voltage
        else:
            V_cell = E - V_conc  # Charging requires pushing against concentration limits

        # Final Stack Voltage (Nernst/Mass Transport are per cell, Ohmic is already stack-level)
        voltage = (cfg.N_cells * V_cell) - V_ohmic
        # ------------------------------------------------------------

        pump_power = cfg.pump_power_coeff * Q**3

        return {
            "time": self.time,
            "soc_true": soc,
            "voltage_stack": voltage,
            "temperature_stack": T_s,
            "temperature_tank": T_t,
            "flow_rate": Q,
            "membrane_resistance": R_m,
            "capacity_nominal": Q_nom,
            "pump_power": pump_power,
            "i_limit": I_limit, # Helpful to track in your dashboard
            "current": I_actual
        }