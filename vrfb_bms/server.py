# vrfb_bms/server.py
"""
VRFB Battery Management System — Real-Time Web Server
======================================================
FastAPI + WebSocket server that runs the VRFB digital twin at 1 Hz,
applies the 7-layer BMS with operational mode state machine,
runs REN SOC inference every step, and broadcasts live data
(including half-cell SOC and imbalance) to all connected dashboard clients.

USAGE:
  pip install fastapi uvicorn
  uvicorn vrfb_bms.server:app --reload --port 8000
  open http://localhost:8000
"""

import asyncio
import json
import os
import pickle
import time
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
SCALER_PATH = os.path.join(os.path.dirname(BASE_DIR), "ren", "scaler.pkl")
MODEL_PATH  = os.path.join(os.path.dirname(BASE_DIR), "ren", "ren_soc_best.pth")

HIDDEN_DIM  = 128
ALPHA       = 0.5    # match train_ren.py v3
DEVICE      = torch.device("cpu")
SIM_HZ         = 1.0    # WebSocket broadcast frequency (Hz)
STEPS_PER_TICK = 10     # Sim steps per broadcast — 10× realtime speed
HISTORY_LEN    = 300


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION MANAGER
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
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationEngine:

    def __init__(self):
        self.cfg     = VRFBConfig()
        self.battery = VRFB(self.cfg)
        self.bms     = BMSController(self.cfg)
        self.sensor  = SensorModel(self.cfg)
        self.cc      = CoulombCounter(self.cfg)
        self.cc.initialize(self.cfg.initial_soc)

        # REN model
        self.model = REN(
            input_dim     = 5,          # V, I, T_stack, T_tank, Q
            hidden_dim    = HIDDEN_DIM,
            output_dim    = 1,
            alpha         = 0.5,        # match train_ren.py v3
            dropout       = 0.1,        # keep > 0 for MC uncertainty
            n_power_iters = 10,
        ).to(DEVICE)
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE)
        )
        self.model.eval()

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        self.z_ren      = torch.zeros(1, HIDDEN_DIM, device=DEVICE)
        self.I_cmd      = 0.0          # operator-commanded current (A)
        self.Q_cmd      = self.cfg.initial_flow   # operator-commanded flow (m³/s)
        self.T_amb      = self.cfg.initial_temperature
        self.history    = []
        self.step_count = 0
        self.t_start    = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # BMS FLAGS  (for dashboard layer indicators)
    # ─────────────────────────────────────────────────────────────────────────

    def _bms_flags(self, out: dict, I_cmd: float, I_safe: float) -> dict:
        cfg  = self.cfg
        mode = self.bms.mode

        # Mode-level gates
        in_standby  = mode is BMSMode.STANDBY
        in_startup  = mode is BMSMode.STARTUP
        in_shutdown = mode is BMSMode.SHUTDOWN

        # Protection-layer activations (only meaningful in CHARGE/DISCHARGE)
        active = mode in (BMSMode.CHARGE, BMSMode.DISCHARGE)

        return {
            # ── Mode state ────────────────────────────────────────────────
            "mode_standby"  : in_standby,
            "mode_startup"  : in_startup,
            "mode_shutdown" : in_shutdown,
            "mode_charge"   : mode is BMSMode.CHARGE,
            "mode_discharge": mode is BMSMode.DISCHARGE,
            # ── 7-layer protection ────────────────────────────────────────
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
            # ── Electrolyte health ────────────────────────────────────────
            "imbalance_warn": bool(out["soc_imbalance"] > 0.05),
            "imbalance_crit": bool(out["soc_imbalance"] > 0.10),
            # ── Standby drain ─────────────────────────────────────────────
            "drain_active"  : self.bms.in_drain,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # REN INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _ren_step(self, measured: dict, out: dict) -> float:
        """
        REN inference step using only physically measurable sensor signals.
        No Coulomb counter output, no derived concentrations.
        """
        x = np.array([[
            measured["voltage"],             # stack voltage [V]
            measured["current"],             # current [A]
            measured["temperature"],         # stack thermocouple [K]
            measured["temperature_tank"],    # tank thermocouple [K]
            measured["flow_rate"],           # flow meter [m3/s]
        ]], dtype=np.float32)
        x_s = self.scaler.transform(x).astype(np.float32)
        x_t = torch.tensor(x_s).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            y_seq, self.z_ren = self.model(x_t, z=self.z_ren)
        return float(y_seq.squeeze())

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE SIMULATION STEP
    # ─────────────────────────────────────────────────────────────────────────

    def step(self) -> dict:
        dt  = self.cfg.dt_default
        out = self.battery.get_outputs()

        # ── 1. Advance mode state machine ─────────────────────────────────
        mode = self.bms.update_mode(self.I_cmd, out, dt)

        # ── 2. Flow override from mode (flush / drain) ────────────────────
        flow_override = self.bms.get_flow_override()
        Q_eff = flow_override if flow_override is not None else self.Q_cmd

        # ── 3. 7-layer protection → safe current ─────────────────────────
        I_safe = self.bms.apply_protection(self.I_cmd, out)

        # ── 4. Collect BMS flags (pre-step state for display) ─────────────
        flags = self._bms_flags(out, self.I_cmd, I_safe)

        # ── 5. Advance physics ────────────────────────────────────────────
        self.battery.step(I_safe, Q_eff, self.T_amb, dt)
        out = self.battery.get_outputs()

        # ── 6. Sensor measurement ─────────────────────────────────────────
        measured = self.sensor.measure(out)

        # ── 7. Coulomb counter ────────────────────────────────────────────
        soc_cc = self.cc.update(
            measured["current"], dt, out["capacity_nominal"]
        )

        # ── 8. REN inference ──────────────────────────────────────────────
        soc_ren  = self._ren_step(measured, out)
        soc_true = out["soc_true"]

        self.step_count += 1

        # ── 9. Build snapshot for WebSocket broadcast ─────────────────────
        derate_pct = (
            (1.0 - abs(I_safe) / max(abs(self.I_cmd), 1e-3)) * 100.0
            if abs(self.I_cmd) > 1 else 0.0
        )

        snap = {
            "type"           : "step",
            # ── Timing ────────────────────────────────────────────────────
            "t"              : self.step_count,
            "elapsed_s"      : round(time.time() - self.t_start, 1),
            "sim_time_s"     : self.step_count,
            # ── SOC — estimators ──────────────────────────────────────────
            "soc_true"       : round(soc_true, 5),
            "soc_cc"         : round(float(soc_cc), 5),
            "soc_ren"        : round(soc_ren, 5),
            "err_cc"         : round(abs(float(soc_cc) - soc_true), 5),
            "err_ren"        : round(abs(soc_ren - soc_true), 5),
            # ── SOC — half-cell breakdown ─────────────────────────────────
            "soc_neg"        : round(out["soc_neg"],       5),
            "soc_pos"        : round(out["soc_pos"],       5),
            "soc_system"     : round(out["soc_system"],    5),
            "soc_imbalance"  : round(out["soc_imbalance"], 5),
            # ── Electrical ────────────────────────────────────────────────
            "voltage"        : round(measured["voltage"],  3),
            "current"        : round(measured["current"],  2),
            "current_cmd"    : round(self.I_cmd,           1),
            "current_safe"   : round(I_safe,               2),
            "i_limit"        : round(out["i_limit"],       1),
            "transport_ratio": round(out["transport_ratio"], 4),
            # ── Thermal ───────────────────────────────────────────────────
            "temp_stack"     : round(measured["temperature"],   2),
            "temp_tank"      : round(out["temperature_tank"],   2),
            "temp_ambient"   : round(self.T_amb,                1),
            # ── Hydraulic ─────────────────────────────────────────────────
            "flow_lpm"       : round(out["flow_rate"] * 60000,  2),
            "flow_override"  : flow_override is not None,
            # ── Degradation ───────────────────────────────────────────────
            "capacity_ah"    : round(out["capacity_nominal"],   1),
            # ── Mode state machine ────────────────────────────────────────
            "bms_mode"            : self.bms.mode_name,
            "bms_standby_timer_s" : round(self.bms.standby_timer, 1),
            "bms_startup_pct"     : round(self.bms.startup_progress * 100, 1),
            "bms_drain_active"    : self.bms.in_drain,
            # ── Protection flags & derate ─────────────────────────────────
            "bms_flags"      : flags,
            "bms_derate_pct" : round(derate_pct, 1),
        }

        # Trim history to last HISTORY_LEN steps
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
            self.__init__()
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

app     = FastAPI(title="VRFB BMS")
manager = ConnectionManager()
engine  = SimulationEngine()

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


@app.on_event("startup")
async def start_simulation():
    asyncio.create_task(_simulation_loop())


async def _simulation_loop():
    interval = 1.0 / SIM_HZ
    while True:
        t0 = asyncio.get_event_loop().time()
        for _ in range(STEPS_PER_TICK):
            snap = engine.step()
        snap["type"] = "step"
        await manager.broadcast(snap)
        dt = asyncio.get_event_loop().time() - t0
        await asyncio.sleep(max(0, interval - dt))