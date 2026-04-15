# vrfb_bms/server.py
"""
VRFB Battery Management System — Real-Time Web Server
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
            input_dim        = 7,
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

    def _init_sim(self):
        """Initialise / re-initialise simulation objects only (not the model)."""
        self.battery    = VRFB(self.cfg)
        self.bms        = BMSController(self.cfg)
        self.sensor     = SensorModel(self.cfg)
        self.cc         = CoulombCounter(self.cfg)
        self.cc.initialize(self.cfg.initial_soc)
        self.I_cmd      = 0.0
        self.Q_cmd      = self.cfg.initial_flow
        self.T_amb      = self.cfg.initial_temperature
        self.history    = []
        self.step_count = 0
        self.t_start    = time.time()

    def _reset_ren_state(self):
        """Reset REN hidden state and EMA — use learned z0, not zeros."""
        with torch.no_grad():
            self.z_ren = self.model.z0.detach().clone()
        self.soc_ren_ema_fast = self.cfg.initial_soc   # for BMS
        self.soc_ren_ema_disp = self.cfg.initial_soc   # for display

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
        7-feature hybrid inference: 5 sensors + soc_cc + transport_ratio_approx.
 
        GATE FIX: raw current (Amps) passed as x_raw so gate closes at I=0A.
        Previously no x_raw was passed — gate used scaled current and never
        closed at zero current (gate ≈ 0.44 instead of 0.0 at I=0A).
        """
        Q  = measured["flow_rate"]
        I  = measured["current"]                           # raw Amps
        il = self._il_const * (max(Q, 1e-6) / self._q_ref) ** 0.4
        tr = abs(I) / max(il, 1.0)                        # 0 when I = 0
 
        x = np.array([[
            measured["voltage"],
            I,
            measured["temperature"],
            measured["temperature_tank"],
            Q,
            float(soc_cc),
            tr,
        ]], dtype=np.float32)
 
        x_s = self.scaler.transform(x).astype(np.float32)
        x_t = torch.tensor(x_s).unsqueeze(0).to(DEVICE)
 
        # Raw current as (1, 1, 1) tensor in Amps — gate uses this, not scaled value
        I_raw = torch.tensor([[[I]]], dtype=torch.float32).to(DEVICE)
 
        with torch.no_grad():
            y_seq, self.z_ren = self.model(x_t, z=self.z_ren, x_raw=I_raw)
 
        correction = float(y_seq.squeeze())
        soc_ren    = float(np.clip(soc_cc + correction, 0.0, 1.0))
        return soc_ren
    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE SIMULATION STEP
    # ─────────────────────────────────────────────────────────────────────────

    def step(self) -> dict:
        dt  = self.cfg.dt_default
        out = self.battery.get_outputs()

        # 1. Mode state machine
        self.bms.update_mode(self.I_cmd, out, dt)

        # 2. Flow override (startup flush / standby drain)
        flow_override = self.bms.get_flow_override()
        Q_eff = flow_override if flow_override is not None else self.Q_cmd

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

        # 8. REN inference + EMA smoothing
        soc_ren      = self._ren_step(measured, soc_cc)
        raw = soc_ren
        
        soc_true = out["soc_true"]

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
        I_cmd:    Optional[float] = None,
        flow_lpm: Optional[float] = None,
        T_amb:    Optional[float] = None,
        reset:    bool = False,
    ):
        if reset:
            # Rebuild simulation state only — model stays loaded, no disk I/O
            self._init_sim()
            self._reset_ren_state()
            return
        if I_cmd    is not None:
            self.I_cmd = float(np.clip(I_cmd, -200, 200))
        if flow_lpm is not None:
            self.Q_cmd = float(np.clip(flow_lpm, 5, 60)) * self.cfg.LPM_to_m3s
        if T_amb    is not None:
            self.T_amb = float(np.clip(T_amb, 278, 323))

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
        I_cmd    = body.get("I_cmd"),
        flow_lpm = body.get("flow_lpm"),
        T_amb    = body.get("T_amb"),
        reset    = body.get("reset", False),
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