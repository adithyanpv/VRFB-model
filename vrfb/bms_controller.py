# bms_controller.py

import numpy as np


class BMSController:
    """
    Battery Management System for a Vanadium Redox Flow Battery.

    Protection hierarchy (applied in strict priority order):
      1. Hard voltage cut-off          — immediate shutdown
      2. SOC window enforcement        — immediate shutdown
      3. Thermal hard shutdown         — immediate shutdown
      4. Thermal derating              — proportional current reduction
      5. Limiting-current derating     — electrochemical safety margin
      6. Flow-critical derating        — mass-transport protection
      7. Global current clip           — absolute hardware limits

    Sign convention:
      I > 0  →  discharge  (positive terminal delivers current)
      I < 0  →  charge     (positive terminal absorbs current)
    """

    def __init__(self, config):
        self.cfg = config

    # ============================================================
    # MAIN SUPERVISORY FUNCTION
    # ============================================================
    def apply_protection(self, I_cmd, outputs):

        cfg = self.cfg

        V       = outputs["voltage_stack"]
        T       = outputs["temperature_stack"]
        SOC     = outputs["soc_true"]
        flow    = outputs["flow_rate"]
        I_limit = outputs.get("i_limit", None)

        I_safe = float(I_cmd)

        # ------------------------------------------------------------
        # 1️⃣  HARD VOLTAGE CUT-OFF
        #     Prevent cell reversal (under-voltage) or electrolyte
        #     decomposition (over-voltage).
        # ------------------------------------------------------------
        if V <= cfg.V_min_stack and I_safe > 0:
            # Under-voltage during discharge → stop discharging
            return 0.0

        if V >= cfg.V_max_stack and I_safe < 0:
            # Over-voltage during charge → stop charging
            return 0.0

        # ------------------------------------------------------------
        # 2️⃣  SOC WINDOW ENFORCEMENT
        #     Keep SOC within [soc_min, soc_max] to protect electrolyte
        #     and prevent irreversible precipitation reactions.
        # ------------------------------------------------------------
        if SOC <= cfg.soc_min and I_safe > 0:
            return 0.0

        if SOC >= cfg.soc_max and I_safe < 0:
            return 0.0

        # ------------------------------------------------------------
        # 3️⃣  THERMAL HARD SHUTDOWN
        #     Above T_max, stop all current immediately.
        # ------------------------------------------------------------
        if T >= cfg.T_max:
            return 0.0

        # ------------------------------------------------------------
        # 4️⃣  THERMAL DERATING
        #     Between T_derate and T_max, linearly scale current down
        #     from 100% → 0% as temperature rises.
        # ------------------------------------------------------------
        if cfg.T_derate <= T < cfg.T_max:
            scale  = (cfg.T_max - T) / (cfg.T_max - cfg.T_derate)
            scale  = np.clip(scale, 0.0, 1.0)
            I_safe *= scale

        # ------------------------------------------------------------
        # 5️⃣  LIMITING CURRENT DERATING
        #     Keep |I| below a safety margin of the mass-transport
        #     limiting current to prevent concentration polarisation
        #     and electrode flooding.
        #
        #     FIX: Guard against near-zero I_limit (can occur at very
        #     low SOC, but SOC check above should prevent reaching here).
        #     Added a minimum threshold (1.0 A) below which this
        #     protection is skipped — the SOC hard stop is the
        #     appropriate guard at that point, not a derating calculation
        #     that could divide by near-zero.
        # ------------------------------------------------------------
        if I_limit is not None and I_limit > 1.0:
            I_max_allowed = cfg.limiting_current_margin * I_limit
            ratio = abs(I_safe) / I_max_allowed
            if ratio > 1.0:
                I_safe /= ratio     # scale down proportionally, preserve sign

        # ------------------------------------------------------------
        # 6️⃣  FLOW-CRITICAL DERATING
        #     If flow drops below the critical threshold, reduce current
        #     proportionally to prevent local electrolyte depletion.
        # ------------------------------------------------------------
        if flow < cfg.flow_critical:
            flow_ratio = flow / cfg.flow_critical
            flow_ratio = np.clip(flow_ratio, 0.0, 1.0)
            I_safe    *= flow_ratio

        # ------------------------------------------------------------
        # 7️⃣  GLOBAL HARDWARE CURRENT LIMITS
        #     Final hard clip — cannot exceed converter / cable ratings.
        # ------------------------------------------------------------
        I_safe = np.clip(I_safe, cfg.I_min, cfg.I_max)

        return I_safe