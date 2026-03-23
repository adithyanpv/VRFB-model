# bms_controller.py

from enum import Enum
import numpy as np


class BMSMode(Enum):
    """
    Operational modes for the VRFB-BMS.

    Transition summary
    ------------------
    STANDBY   → STARTUP    : |I_cmd| > I_DEAD_BAND
    STARTUP   → CHARGE     : flush timer >= FLUSH_DURATION  AND  I_cmd < 0
    STARTUP   → DISCHARGE  : flush timer >= FLUSH_DURATION  AND  I_cmd >= 0
    STARTUP   → STANDBY    : power request cancelled mid-flush
    CHARGE    → STANDBY    : I_cmd >= −I_DEAD_BAND, or SOC >= soc_max, or V >= V_max
    DISCHARGE → STANDBY    : I_cmd <=  I_DEAD_BAND, or SOC <= soc_min, or V <= V_min
    Any       → SHUTDOWN   : T >= T_max  (emergency — sticky until reset())
    SHUTDOWN  → STANDBY    : operator reset only
    """
    STARTUP   = "startup"    # pump flush in progress — no current yet
    CHARGE    = "charge"
    DISCHARGE = "discharge"
    STANDBY   = "standby"
    SHUTDOWN  = "shutdown"   # emergency — requires external reset


class BMSController:
    """
    Battery Management System for a Vanadium Redox Flow Battery.

    Responsibilities
    ----------------
    1. Operational mode state machine  (charge / discharge / standby /
       shutdown / startup-flush)  — updated every timestep via update_mode().

    2. 7-layer current protection hierarchy — applied inside apply_protection()
       only when the mode is CHARGE or DISCHARGE.

    Physical motivation for startup flush
    --------------------------------------
    After a standby period the electrolyte concentrations at the electrode
    surface diverge from the bulk tank values due to self-discharge crossover.
    Running the pump at maximum flow for one minute circulates fresh
    electrolyte from the tanks into the stack before current is applied,
    preventing localised over- or under-charging at the start of a cycle.

    Physical motivation for standby drain
    --------------------------------------
    Extended standby with electrolyte inside the stack accelerates self-
    discharge (V²⁺ reacts spontaneously with VO₂⁺ across the membrane).
    After STANDBY_DRAIN_DELAY seconds the BMS reduces flow to flow_min —
    representing the practical effect of draining the stack into the storage
    tanks (not all VRFB designs support full drain; minimum flow is the
    conservative approximation used here).

    Protection hierarchy (applied in strict priority inside apply_protection)
    --------------------------------------------------------------------------
    1. Hard voltage cut-off          — immediate current block
    2. SOC window enforcement        — immediate current block
    3. Thermal hard shutdown         — immediate current block
    4. Thermal derating              — proportional current reduction
    5. Limiting-current derating     — mass-transport safety margin
    6. Flow-critical derating        — low-flow protection
    7. Global hardware current clip  — absolute converter limits

    Sign convention: I > 0 → discharge,  I < 0 → charge.
    """

    # ── Timing constants ────────────────────────────────────────────────────
    FLUSH_DURATION     = 60.0    # s   pump flush before current is allowed
    STANDBY_DRAIN_DELAY = 1800.0 # s   30 min standby → reduce to drain flow
    I_DEAD_BAND        = 2.0     # A   |I_cmd| below this → remain in standby

    def __init__(self, config):
        self.cfg = config

        self.mode           = BMSMode.STANDBY
        self._standby_timer = 0.0   # s elapsed in current STANDBY period
        self._startup_timer = 0.0   # s elapsed in current STARTUP flush
        self._intended_mode = None  # CHARGE or DISCHARGE — set at STARTUP entry

    # ================================================================
    # OPERATOR INTERFACE
    # ================================================================

    def reset(self):
        """
        Clear SHUTDOWN (or any fault state) and return to STANDBY.
        Must be called by the operator after resolving the fault condition.
        """
        self.mode           = BMSMode.STANDBY
        self._standby_timer = 0.0
        self._startup_timer = 0.0
        self._intended_mode = None

    # ================================================================
    # MODE STATE MACHINE
    # ================================================================

    def update_mode(self, I_cmd: float, outputs: dict, dt: float) -> "BMSMode":
        """
        Advance the mode state machine by dt seconds.

        Must be called BEFORE apply_protection() on every timestep.
        Returns the (possibly updated) current mode.

        Parameters
        ----------
        I_cmd   : commanded current (A), sign convention above
        outputs : dict from vrfb_core.VRFB.get_outputs()
        dt      : timestep (s)
        """
        cfg = self.cfg
        V   = outputs["voltage_stack"]
        T   = outputs["temperature_stack"]
        SOC = outputs["soc_true"]

        # ── Emergency: thermal shutdown — highest priority, any mode ─────
        if T >= cfg.T_max and self.mode is not BMSMode.SHUTDOWN:
            self.mode = BMSMode.SHUTDOWN
            return self.mode

        # ── Mode-specific transitions ─────────────────────────────────────
        if self.mode is BMSMode.SHUTDOWN:
            pass  # sticky — only reset() clears it

        elif self.mode is BMSMode.STANDBY:
            self._standby_timer += dt
            if abs(I_cmd) > self.I_DEAD_BAND:
                # Power request arrived — begin startup flush
                self._intended_mode = (
                    BMSMode.CHARGE if I_cmd < 0 else BMSMode.DISCHARGE
                )
                self._startup_timer = 0.0
                self.mode = BMSMode.STARTUP

        elif self.mode is BMSMode.STARTUP:
            # Check if power request was cancelled mid-flush
            if abs(I_cmd) <= self.I_DEAD_BAND:
                self.mode = BMSMode.STANDBY
                self._standby_timer = 0.0
                return self.mode

            self._startup_timer += dt
            if self._startup_timer >= self.FLUSH_DURATION:
                # Flush complete — engage intended mode
                # Re-evaluate direction in case I_cmd sign flipped during flush
                self._intended_mode = (
                    BMSMode.CHARGE if I_cmd < 0 else BMSMode.DISCHARGE
                )
                self.mode = self._intended_mode
                self._standby_timer = 0.0

        elif self.mode is BMSMode.CHARGE:
            if (I_cmd >= -self.I_DEAD_BAND     # power request gone
                    or SOC >= cfg.soc_max       # SOC ceiling reached
                    or V   >= cfg.V_max_stack): # voltage ceiling reached
                self.mode = BMSMode.STANDBY
                self._standby_timer = 0.0

        elif self.mode is BMSMode.DISCHARGE:
            if (I_cmd <=  self.I_DEAD_BAND     # power request gone
                    or SOC <= cfg.soc_min       # SOC floor reached
                    or V   <= cfg.V_min_stack): # voltage floor reached
                self.mode = BMSMode.STANDBY
                self._standby_timer = 0.0

        return self.mode

    # ================================================================
    # FLOW OVERRIDE
    # ================================================================

    def get_flow_override(self) -> "float | None":
        """
        Returns a flow-rate override (m³/s) driven by mode requirements,
        or None if the caller's own Q_cmd should be used.

        STARTUP  → flow_max  (maximum flush to clear electrode surface)
        STANDBY (drain phase) → flow_min  (drain / minimise self-discharge)
        All other modes → None  (no override)
        """
        if self.mode is BMSMode.STARTUP:
            return self.cfg.flow_max

        if (self.mode is BMSMode.STANDBY
                and self._standby_timer > self.STANDBY_DRAIN_DELAY):
            return self.cfg.flow_min

        return None

    # ================================================================
    # MAIN PROTECTION HIERARCHY
    # ================================================================

    def apply_protection(self, I_cmd: float, outputs: dict) -> float:
        """
        Apply the 7-layer BMS protection hierarchy.

        Returns the safe current to command (A).

        In STANDBY / STARTUP / SHUTDOWN modes this always returns 0.0
        — the protection hierarchy is superseded by the mode gate.

        Parameters
        ----------
        I_cmd   : commanded current (A)
        outputs : dict from vrfb_core.VRFB.get_outputs()
        """
        # ── Mode gate: only CHARGE or DISCHARGE allow non-zero current ───
        if self.mode not in (BMSMode.CHARGE, BMSMode.DISCHARGE):
            return 0.0

        cfg = self.cfg

        V       = outputs["voltage_stack"]
        T       = outputs["temperature_stack"]
        SOC     = outputs["soc_true"]
        flow    = outputs["flow_rate"]
        I_limit = outputs.get("i_limit", None)

        I_safe = float(I_cmd)

        # ── 1. Hard voltage cut-off ──────────────────────────────────────
        if V <= cfg.V_min_stack and I_safe > 0:
            return 0.0
        if V >= cfg.V_max_stack and I_safe < 0:
            return 0.0

        # ── 2. SOC window enforcement ────────────────────────────────────
        if SOC <= cfg.soc_min and I_safe > 0:
            return 0.0
        if SOC >= cfg.soc_max and I_safe < 0:
            return 0.0

        # ── 3. Thermal hard shutdown (belt-and-suspenders) ───────────────
        if T >= cfg.T_max:
            return 0.0

        # ── 4. Thermal derating ──────────────────────────────────────────
        if cfg.T_derate <= T < cfg.T_max:
            scale  = (cfg.T_max - T) / (cfg.T_max - cfg.T_derate)
            scale  = np.clip(scale, 0.0, 1.0)
            I_safe *= scale

        # ── 5. Limiting-current derating ─────────────────────────────────
        # Guard against near-zero I_limit (very low SOC corner case —
        # the SOC hard stop above should catch this first).
        if I_limit is not None and I_limit > 1.0:
            I_max_allowed = cfg.limiting_current_margin * I_limit
            ratio = abs(I_safe) / I_max_allowed
            if ratio > 1.0:
                I_safe /= ratio     # scale down preserving sign

        # ── 6. Flow-critical derating ────────────────────────────────────
        if flow < cfg.flow_critical:
            flow_ratio = np.clip(flow / cfg.flow_critical, 0.0, 1.0)
            I_safe    *= flow_ratio

        # ── 7. Global hardware current limits ────────────────────────────
        I_safe = np.clip(I_safe, cfg.I_min, cfg.I_max)

        return I_safe

    # ================================================================
    # CONVENIENCE PROPERTIES
    # ================================================================

    @property
    def mode_name(self) -> str:
        return self.mode.value

    @property
    def standby_timer(self) -> float:
        """Seconds spent in the current STANDBY period."""
        return self._standby_timer

    @property
    def startup_progress(self) -> float:
        """Flush progress fraction [0, 1] during STARTUP; 0 otherwise."""
        if self.mode is BMSMode.STARTUP:
            return min(self._startup_timer / self.FLUSH_DURATION, 1.0)
        return 0.0

    @property
    def in_drain(self) -> bool:
        """True when STANDBY has lasted long enough to enter drain flow."""
        return (
            self.mode is BMSMode.STANDBY
            and self._standby_timer > self.STANDBY_DRAIN_DELAY
        )