# vrfb/bms_controller.py

import numpy as np


class BMSController:

    def __init__(self, config):
        self.cfg = config

    # ============================================================
    # MAIN SUPERVISORY FUNCTION
    # ============================================================
    def apply_protection(self, I_cmd, outputs):

        cfg = self.cfg

        V = outputs["voltage_stack"]
        T = outputs["temperature_stack"]
        SOC = outputs["soc_true"]
        flow = outputs["flow_rate"]
        I_limit = outputs.get("i_limit", None)

        I_safe = I_cmd

        # --------------------------------------------------------
        # 1️⃣ HARD VOLTAGE CUT-OFF
        # --------------------------------------------------------
        if V <= cfg.V_min_stack and I_cmd > 0:
            I_safe = 0.0

        if V >= cfg.V_max_stack and I_cmd < 0:
            I_safe = 0.0

        # --------------------------------------------------------
        # 2️⃣ SOC LIMIT PROTECTION
        # --------------------------------------------------------
        if SOC <= cfg.soc_min and I_cmd > 0:
            I_safe = 0.0

        if SOC >= cfg.soc_max and I_cmd < 0:
            I_safe = 0.0

        # --------------------------------------------------------
        # 3️⃣ THERMAL SHUTDOWN
        # --------------------------------------------------------
        if T >= cfg.T_max:
            I_safe = 0.0

        # --------------------------------------------------------
        # 4️⃣ THERMAL DERATING
        # --------------------------------------------------------
        if cfg.T_derate <= T < cfg.T_max:
            # Linear derating between T_derate and T_max
            scale = (cfg.T_max - T) / (cfg.T_max - cfg.T_derate)
            scale = np.clip(scale, 0.0, 1.0)
            I_safe *= scale

        # --------------------------------------------------------
        # 5️⃣ LIMITING CURRENT DERATING
        # --------------------------------------------------------
        if I_limit is not None and I_limit > 0:
            I_max_allowed = cfg.limiting_current_margin * I_limit
            if abs(I_safe) > I_max_allowed:
                I_safe = np.sign(I_safe) * I_max_allowed

        # --------------------------------------------------------
        # 6️⃣ FLOW-CRITICAL DERATING
        # --------------------------------------------------------
        if flow < cfg.flow_critical:
            # Reduce current proportionally if flow too low
            flow_ratio = flow / cfg.flow_critical
            flow_ratio = np.clip(flow_ratio, 0.0, 1.0)
            I_safe *= flow_ratio

        # --------------------------------------------------------
        # 7️⃣ GLOBAL CURRENT LIMITS
        # --------------------------------------------------------
        I_safe = np.clip(I_safe, cfg.I_min, cfg.I_max)

        return I_safe