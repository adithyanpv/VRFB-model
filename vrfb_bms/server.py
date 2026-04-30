# vrfb_bms/server.py
"""
VRFB Battery Management System — Real-Time Web Server 1.7
======================================================
FastAPI + WebSocket server that runs the VRFB digital twin at 1 Hz,
applies the 7-layer BMS with operational mode state machine,
runs REN SOC inference every step, and broadcasts live data
(including half-cell SOC and imbalance) to all connected dashboard clients.

USAGE:
  uvicorn vrfb_bms.server:app --reload --port 8000
  open http://localhost:8000
"""

import asyncio
import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from vrfb.config          import VRFBConfig
from vrfb.vrfb_core       import VRFB
from vrfb.bms_controller  import BMSController, BMSMode
from vrfb.sensor_model    import SensorModel
from vrfb.coulomb_counter import CoulombCounter
from ren.ren_model        import REN

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR  = os.path.join(BASE_DIR, "static")

HIDDEN_DIM     = 128
ALPHA          = 0.5
DEVICE         = torch.device("cpu")
SIM_HZ         = 1.0
STEPS_PER_TICK = 10
HISTORY_LEN    = 300
EMA_TAU_FAST = 30     # used for BMS protection decisions (responsive)
EMA_TAU_DISP = 120    # used for dashboard display (smooth visual)
INIT_DECAY_STEPS = 1800.0 


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION MANAGER  — only manages WebSocket connections, nothing else
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg  = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE  — all physics, BMS, REN logic lives here
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationEngine:

    def __init__(self):
        self.cfg = VRFBConfig()

        proj_root   = os.path.dirname(BASE_DIR)
        model_path  = os.path.join(proj_root, "ren", "ren_soc_best.pth")
        scaler_path = os.path.join(proj_root, "ren", "scaler.pkl")

        # ── Load REN model ────────────────────────────────────────────────
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"REN model not found at:\n  {model_path}\n"
                "Run train_ren.py first."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found at:\n  {scaler_path}\n"
                "Run train_ren.py first."
            )

        self.model = REN(
            input_dim        = 6,  # 5 sensors + soc_cc + transport_ratio_approx
            hidden_dim       = HIDDEN_DIM,
            output_dim       = 1,
            alpha            = ALPHA,
            dropout          = 0.1,
            n_power_iters    = 10,
            use_feedthrough  = False,
            use_current_gate = True,
            current_feat_idx = 1,
        ).to(DEVICE)
        self.model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True)
        )
        self.model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # I_limit approximation constants (hardware-estimable, no ODE access)
        self._il_const = 1 * 96485.0 * 2e-5 * 0.15 * 1600.0 * 0.5  # ~231.6 A
        self._q_ref    = 20.0 / 60000.0   # 20 LPM in m³/s

        # Initialise simulation state and REN hidden state
        self._init_sim()
        self._reset_ren_state()

    # ─────────────────────────────────────────────────────────────────────────

    # ── IFC physics constants (Faraday-based optimal flow calculation) ─────────
    # Q_min = |I| / (n * F * C_v * SOC_eff)
    # Q_optimal = Q_min * IFC_FLOW_FACTOR
    # Clamp output to [IFC_Q_MIN_LPM, IFC_Q_MAX_LPM]
    IFC_FLOW_FACTOR = 170.0   # was 4.0 — Faraday minimum is the absolute bulk lower
                              # bound. VRFB stacks need ~150-200× this for electrode
                              # surface replenishment and transport uniformity.
                              # Verification at nominal conditions (150A, SOC=0.50):
                              #   Q_min = 150/(96485×1600×0.5) = 1.94e-6 m³/s
                              #   Q_opt = 1.94e-6 × 170 = 3.3e-4 m³/s = 19.8 LPM ✓
                              # At low SOC (0.30): Q_opt = 33 LPM ✓
                              # Near floor (0.05): Q_opt = 198 LPM → clamped to 55 ✓
    IFC_Q_MIN_LPM   = 8.0     # physical lower bound (critical threshold, unchanged)
    IFC_Q_MAX_LPM   = 55.0    # physical upper bound (unchanged)
    IFC_SOC_FLOOR   = 0.05    # matches soc_min in config

    def _init_sim(self):
        """Initialise / re-initialise simulation objects only (not the model)."""
        self.battery    = VRFB(self.cfg)
        self.bms        = BMSController(self.cfg)
        self.sensor     = SensorModel(self.cfg)
        self.cc         = CoulombCounter(self.cfg)
        self.cc.initialize(self.cfg.initial_soc)
        self.I_cmd        = 0.0
        self.Q_cmd        = self.cfg.initial_flow
        self.T_amb        = self.cfg.initial_temperature
        self.auto_flow    = False    # IFC closed-loop control flag
        self.history      = []
        self.step_count   = 0
        self.t_start      = time.time()
        self._prev_soc_cc      = self.cfg.initial_soc
        self._prev_bms_mode    = "standby"
        self._transition_steps = 0
        self._bias_est         = 0.0
        self._cumulative_ah    = 0.0
        self._init_soc_for_decay = self.cfg.initial_soc  # anchor for soc_init_decay
        self._decay_step         = 0

    def _reset_ren_state(self):
        with torch.no_grad():
            self.z_ren = self.model.z0.detach().clone()
            self.soc_ren_ema_fast    = self.cfg.initial_soc
            self.soc_ren_ema_disp    = self.cfg.initial_soc
            self._prev_bms_mode      = "standby"
            self._transition_steps   = 0
            self._decay_step         = 0
            self._init_soc_for_decay = self.cfg.initial_soc  # use config initial SOC

    # ─────────────────────────────────────────────────────────────────────────
    # INTELLIGENT FLOW CONTROLLER (IFC) — Faraday-based optimal flow
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_ifc_flow(self) -> float:
        """
        Calculate the optimal pump flow rate from Faraday's law using the
        REN fast-EMA SOC estimate (soc_ren_ema_fast) as the reactant gauge.

        Physics:
          The minimum electrolyte flow to avoid concentration starvation at
          the electrode surface is set by the molar flux balance:

            Q_min [m³/s] = |I| / (n × F × C_v × SOC_eff)

          where:
            I       = stack current (A)  — from last measured value
            n       = 1  (one electron per vanadium ion)
            F       = 96485  C/mol
            C_v     = total vanadium concentration  (mol/m³, from config)
            SOC_eff = available reactant fraction:
                        discharging → soc_ren_ema_fast   (V²⁺ on neg side)
                        charging    → 1 - soc_ren_ema_fast (V³⁺ on neg side)

          Q_optimal = Q_min × IFC_FLOW_FACTOR  (safety margin, default 4×)

        When I = 0 (standby), returns the physical minimum flow (idle circulation).

        Returns: flow rate in m³/s, clamped to [IFC_Q_MIN_LPM, IFC_Q_MAX_LPM].
        """
        I_abs   = abs(self.I_cmd)
        soc_ai  = float(np.clip(self.soc_ren_ema_fast, 0.0, 1.0))

        if self._decay_step < 3600:
            return 20.0 * self.cfg.LPM_to_m3s 

        if I_abs < 1.0:
            # Standby — run at minimum to maintain electrolyte circulation
            return self.IFC_Q_MIN_LPM * self.cfg.LPM_to_m3s
        
        

        # Determine which reactant is being consumed
        if self.I_cmd > 0:
            # Discharging: V²⁺ (negative side) is consumed; SOC_eff = SOC
            soc_eff = max(soc_ai, self.IFC_SOC_FLOOR)
        else:
            # Charging: V³⁺ (negative side) is consumed; SOC_eff = 1 - SOC
            soc_eff = max(1.0 - soc_ai, self.IFC_SOC_FLOOR)

        # Faraday minimum flow (m³/s)
        Q_min     = I_abs / (self.cfg.n * self.cfg.F * self.cfg.C_total * soc_eff)
        Q_optimal = Q_min * self.IFC_FLOW_FACTOR

        # Convert limits to m³/s and clamp
        Q_lo = self.IFC_Q_MIN_LPM * self.cfg.LPM_to_m3s
        Q_hi = self.IFC_Q_MAX_LPM * self.cfg.LPM_to_m3s
        return float(np.clip(Q_optimal, Q_lo, Q_hi))

    # ─────────────────────────────────────────────────────────────────────────
    # BMS FLAGS
    # ─────────────────────────────────────────────────────────────────────────

    def _bms_flags(self, out: dict, I_cmd: float, I_safe: float) -> dict:
        cfg    = self.cfg
        mode   = self.bms.mode
        active = mode in (BMSMode.CHARGE, BMSMode.DISCHARGE)

        return {
            "mode_standby"  : mode is BMSMode.STANDBY,
            "mode_startup"  : mode is BMSMode.STARTUP,
            "mode_shutdown" : mode is BMSMode.SHUTDOWN,
            "mode_charge"   : mode is BMSMode.CHARGE,
            "mode_discharge": mode is BMSMode.DISCHARGE,
            "voltage_cutoff": active and bool(
                (out["voltage_stack"] <= cfg.V_min_stack and I_cmd > 0) or
                (out["voltage_stack"] >= cfg.V_max_stack and I_cmd < 0)
            ),
            "soc_window": active and bool(
                (out["soc_true"] <= cfg.soc_min and I_cmd > 0) or
                (out["soc_true"] >= cfg.soc_max and I_cmd < 0)
            ),
            "thermal_hard"  : bool(out["temperature_stack"] >= cfg.T_max),
            "thermal_derate": bool(cfg.T_derate <= out["temperature_stack"] < cfg.T_max),
            "i_limit_derate": active and bool(
                out.get("i_limit", 999) > 1.0 and
                abs(I_cmd) > cfg.limiting_current_margin * out.get("i_limit", 999)
            ),
            "flow_derate"   : bool(out["flow_rate"] < cfg.flow_critical),
            "imbalance_warn": bool(out["soc_imbalance"] > 0.05),
            "imbalance_crit": bool(out["soc_imbalance"] > 0.10),
            "drain_active"  : self.bms.in_drain,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # REN INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _ren_step(self, measured: dict, soc_cc: float) -> float:
        """
        Direct SOC inference — v6.

        Features (6): v_ocv_approx, current, T_stack, T_tank, flow_rate, soc_init_decay
        soc_cc is NOT a feature — it is passed in only to be available if needed elsewhere.

        soc_init_decay = _init_soc_for_decay * exp(-_decay_step / INIT_DECAY_STEPS)
        _init_soc_for_decay is reset to the Nernst-estimated SOC at every BMS mode
        transition (standby→charge, standby→discharge) so the prior is always fresh.

        Returns: direct SOC estimate in [0, 1] — NOT soc_cc + correction.
        """
        import math

        I = measured["current"]
        Q = measured["flow_rate"]

        _R_STACK = (self.cfg.R_membrane_initial + self.cfg.R_contact) * self.cfg.N_cells
        # Approximate concentration overpotential correction using flow rate.
        # At reference flow (20 LPM), this term is zero.
        # At low flow, adds correction to compensate for larger concentration drop.
        # At high flow (45 LPM), subtracts slightly to avoid over-correction.
        # This keeps v_ocv closer to true V_OCV regardless of pump speed.
        Q_ref        = self._q_ref                          # 20 LPM in m³/s
        Q_actual     = max(measured["flow_rate"], 1e-6)
        # Concentration overpotential scales as flow^(-0.4) from vrfb_core.py
        conc_corr    = 0.008 * ((Q_ref / Q_actual) ** 0.4 - 1.0) * np.sign(I)
        v_ocv        = measured["voltage"] + I * _R_STACK + conc_corr

        
        

        soc_init_decay = self._init_soc_for_decay * math.exp(
            -self._decay_step / INIT_DECAY_STEPS
        )
        self._decay_step += 1

        x = np.array([[
            float(v_ocv),
            I,
            measured["temperature"],
            measured["temperature_tank"],
            Q,
            float(soc_init_decay),     # replaces soc_cc
        ]], dtype=np.float32)

        x_s   = self.scaler.transform(x).astype(np.float32)
        x_t   = torch.tensor(x_s).unsqueeze(0).to(DEVICE)
        I_raw = torch.tensor([[[I]]], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            y_seq, self.z_ren = self.model(x_t, z=self.z_ren, x_raw=I_raw)

        # Direct SOC — model output IS the SOC estimate, not a correction delta
        return float(np.clip(float(y_seq.squeeze()), 0.0, 1.0))
    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE SIMULATION STEP
    # ─────────────────────────────────────────────────────────────────────────

    def step(self) -> dict:
        dt  = self.cfg.dt_default
        out = self.battery.get_outputs()

        # 1. Mode state machine
        self.bms.update_mode(self.I_cmd, out, dt)

        # 2. Flow selection:
        #    Priority: BMS override > IFC auto-flow > manual Q_cmd
        flow_override = self.bms.get_flow_override()
        if flow_override is not None:
            # BMS mode (startup flush / standby drain) always takes priority
            Q_eff          = flow_override
            ifc_active     = False
        elif self.auto_flow:
            # IFC closed-loop: AI SOC estimate drives pump speed
            Q_eff          = self._compute_ifc_flow()
            ifc_active     = True
        else:
            # Manual control from user slider
            Q_eff          = self.Q_cmd
            ifc_active     = False

        # 3. Protection hierarchy → safe current
        I_safe = self.bms.apply_protection(self.I_cmd, out)

        # 4. BMS flags for dashboard
        flags = self._bms_flags(out, self.I_cmd, I_safe)

        # 5. Advance physics
        self.battery.step(I_safe, Q_eff, self.T_amb, dt)
        out = self.battery.get_outputs()

        # 6. Sensor measurement
        measured = self.sensor.measure(out)

        # 7. Coulomb counter
        soc_cc = self.cc.update(
            measured["current"], dt, out["capacity_nominal"]
        )

        # 8. REN inference + dual EMA smoothing
        # _ren_step returns raw soc_cc + correction (no smoothing)
        raw = self._ren_step(measured, soc_cc)

        # Transition dampening: when BMS mode changes (e.g., standby→discharge),
        # the voltage reading transiently shows post-rest OCV which the model
        # interprets as higher SOC → spike. Dampen correction for first 60 steps.
        cur_mode = self.bms.mode_name
        if cur_mode != self._prev_bms_mode:
            self._transition_steps = 0
        self._prev_bms_mode = cur_mode
        self._transition_steps += 1

        if self._transition_steps < 180 and cur_mode in ("discharge", "charge"):
            # Extended 3-minute blend — gives model time to converge from OCV
            # after current reversal. Prevents the charge-onset overshoot.
            blend = self._transition_steps / 180.0
            raw   = (1.0 - blend) * self.soc_ren_ema_disp + blend * raw

        a_fast = 1.0 / EMA_TAU_FAST
        a_disp = 1.0 / EMA_TAU_DISP
        if self.bms.mode is BMSMode.STANDBY and self._transition_steps > 60:
            pass  # hold EMA values — do not update
        else:
            self.soc_ren_ema_fast = (1.0 - a_fast) * self.soc_ren_ema_fast + a_fast * raw
            self.soc_ren_ema_disp = (1.0 - a_disp) * self.soc_ren_ema_disp + a_disp * raw

        soc_ren     = float(np.clip(self.soc_ren_ema_disp, 0.0, 1.0))
        soc_ren_bms = float(np.clip(self.soc_ren_ema_fast, 0.0, 1.0))
        soc_true    = out["soc_true"]

        self.step_count += 1

        derate_pct = (
            (1.0 - abs(I_safe) / max(abs(self.I_cmd), 1e-3)) * 100.0
            if abs(self.I_cmd) > 1 else 0.0
        )

        snap = {
            "type"                : "step",
            "t"                   : self.step_count,
            "elapsed_s"           : round(time.time() - self.t_start, 1),
            "sim_time_s"          : self.step_count,
            "soc_true"            : round(soc_true, 5),
            "soc_cc"              : round(float(soc_cc), 5),
            "soc_ren"             : round(soc_ren, 5),
            "soc_ren_bms"         : round(soc_ren_bms, 5),   # BMS (τ=30, responsive)
            "soc_ren_raw"         : round(raw, 5),
            "err_cc"              : round(abs(float(soc_cc) - soc_true), 5),
            "err_ren"             : round(abs(soc_ren - soc_true), 5),
            "soc_neg"             : round(out["soc_neg"],        5),
            "soc_pos"             : round(out["soc_pos"],        5),
            "soc_system"          : round(out["soc_system"],     5),
            "soc_imbalance"       : round(out["soc_imbalance"],  5),
            "voltage"             : round(measured["voltage"],   3),
            "current"             : round(measured["current"],   2),
            "current_cmd"         : round(self.I_cmd,            1),
            "current_safe"        : round(I_safe,                2),
            "i_limit"             : round(out["i_limit"],        1),
            "transport_ratio"     : round(out["transport_ratio"], 4),
            "temp_stack"          : round(measured["temperature"],      2),
            "temp_tank"           : round(measured["temperature_tank"], 2),
            "temp_ambient"        : round(self.T_amb,                   1),
            "flow_lpm"            : round(out["flow_rate"] * 60000,     2),
            "flow_override"       : flow_override is not None,
            "ifc_active"          : ifc_active,
            "ifc_flow_lpm"        : round(Q_eff * 60000, 2) if ifc_active else None,
            "capacity_ah"         : round(out["capacity_nominal"],      1),
            "bms_mode"            : self.bms.mode_name,
            "bms_standby_timer_s" : round(self.bms.standby_timer, 1),
            "bms_startup_pct"     : round(self.bms.startup_progress * 100, 1),
            "bms_drain_active"    : self.bms.in_drain,
            "bms_flags"           : flags,
            "bms_derate_pct"      : round(derate_pct, 1),
        }

        self.history.append({
            k: snap[k] for k in [
                "t", "soc_true", "soc_cc", "soc_ren",
                "voltage", "current", "err_cc", "err_ren",
                "soc_neg", "soc_pos", "soc_imbalance",
            ]
        })
        if len(self.history) > HISTORY_LEN:
            self.history.pop(0)

        return snap

    # ─────────────────────────────────────────────────────────────────────────
    # CONTROL INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def set_control(
        self,
        I_cmd:     Optional[float] = None,
        flow_lpm:  Optional[float] = None,
        T_amb:     Optional[float] = None,
        auto_flow: Optional[bool]  = None,
        reset:     bool = False,
    ):
        if reset:
            self._init_sim()
            self._reset_ren_state()
            return
        if I_cmd     is not None:
            self.I_cmd     = float(np.clip(I_cmd, -200, 200))
        if flow_lpm  is not None:
            self.Q_cmd     = float(np.clip(flow_lpm, 5, 60)) * self.cfg.LPM_to_m3s
        if T_amb     is not None:
            self.T_amb     = float(np.clip(T_amb, 278, 323))
        if auto_flow is not None:
            self.auto_flow = bool(auto_flow)

    def get_history(self) -> list:
        return self.history


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

# manager is safe at module level — it has no imports that can fail
manager = ConnectionManager()

# engine is created inside lifespan so startup errors show the real exception
engine: Optional[SimulationEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SimulationEngine()
    task   = asyncio.create_task(_simulation_loop())
    yield
    task.cancel()


app = FastAPI(title="VRFB BMS", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/history")
async def get_history():
    return {"history": engine.get_history()}


@app.post("/control")
async def control(body: dict):
    engine.set_control(
        I_cmd     = body.get("I_cmd"),
        flow_lpm  = body.get("flow_lpm"),
        T_amb     = body.get("T_amb"),
        auto_flow = body.get("auto_flow"),
        reset     = body.get("reset", False),
    )
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_text(json.dumps({
            "type": "history",
            "history": engine.get_history()
        }))
    except Exception:
        pass
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


async def _simulation_loop():
    interval = 1.0 / SIM_HZ
    while True:
        t0 = asyncio.get_event_loop().time()
        for _ in range(STEPS_PER_TICK):
            snap = engine.step()
        snap["type"] = "step"
        await manager.broadcast(snap)
        elapsed = asyncio.get_event_loop().time() - t0
        await asyncio.sleep(max(0.0, interval - elapsed))