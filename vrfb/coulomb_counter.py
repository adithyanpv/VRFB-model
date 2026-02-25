# coulomb_counter.py

import numpy as np


class CoulombCounter:

    def __init__(self, config):

        self.cfg = config

        # Initial SOC estimate
        self.soc_est = config.initial_soc

        # Optional current sensor bias (can be set externally)
        self.current_bias = 0.0

    # ------------------------------------------------------------
    # Optional: Set current bias (for sensor drift experiments)
    # ------------------------------------------------------------
    def set_current_bias(self, bias):
        self.current_bias = bias

    # ------------------------------------------------------------
    # Initialize manually if needed
    # ------------------------------------------------------------
    def initialize(self, soc_initial):
        self.soc_est = np.clip(
            soc_initial,
            self.cfg.soc_min,
            self.cfg.soc_max
        )

    # ------------------------------------------------------------
    # Update SOC Estimate
    # ------------------------------------------------------------
    def update(self,
               measured_current,
               dt,
               Q_nominal):
        """
        measured_current : Amps
        dt               : seconds
        Q_nominal        : Ah (from vrfb_core output)
        """

        # Apply optional sensor bias
        I = measured_current + self.current_bias

        # Coulomb integration
        # ΔSOC = I * dt / (3600 * Q_nominal)
        delta_soc = (I * dt) / (3600.0 * Q_nominal)

        # Update SOC
        self.soc_est -= delta_soc

        # Bound SOC
        self.soc_est = np.clip(
            self.soc_est,
            self.cfg.soc_min,
            self.cfg.soc_max
        )

        return self.soc_est

    # ------------------------------------------------------------
    def get_soc(self):
        return self.soc_est