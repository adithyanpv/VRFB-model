# vrfb_bms/server.py
"""
VRFB Battery Management System — Real-Time Web Server
======================================================
FastAPI + WebSocket server that runs the VRFB digital twin at 1 Hz,
applies the 7-layer BMS, runs REN SOC inference every step, and
broadcasts live data to all connected dashboard clients.

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
from vrfb.bms_controller  import BMSController
from vrfb.sensor_model    import SensorModel
from vrfb.coulomb_counter import CoulombCounter
from ren.ren_model        import REN

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR  = os.path.join(BASE_DIR, "static")
SCALER_PATH = os.path.join(os.path.dirname(BASE_DIR), "ren", "scaler.pkl")
MODEL_PATH  = os.path.join(os.path.dirname(BASE_DIR), "ren", "ren_soc_best.pth")

HIDDEN_DIM  = 128
ALPHA       = 0.95
DEVICE      = torch.device("cpu")
SIM_HZ         = 1.0    # WebSocket broadcast frequency (Hz)
STEPS_PER_TICK = 10     # Sim steps per broadcast — 10x realtime speed
                        # 1 real second = 10 simulated seconds
                        # SOC visible change in ~30s instead of 5 min
HISTORY_LEN    = 300


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


class SimulationEngine:
    def __init__(self):
        self.cfg     = VRFBConfig()
        self.battery = VRFB(self.cfg)
        self.bms     = BMSController(self.cfg)
        self.sensor  = SensorModel(self.cfg)
        self.cc      = CoulombCounter(self.cfg)
        self.cc.initialize(self.cfg.initial_soc)

        self.model = REN(input_dim=8, hidden_dim=HIDDEN_DIM,
                         output_dim=1, alpha=ALPHA, dropout=0.0).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        self.z_ren      = torch.zeros(1, HIDDEN_DIM, device=DEVICE)
        self.I_cmd      = 120.0
        self.Q_cmd      = self.cfg.initial_flow
        self.T_amb      = self.cfg.initial_temperature
        self.history    = []
        self.step_count = 0
        self.t_start    = time.time()
        self.bms_flags  = {}

    def _bms_flags(self, out, I_cmd):
        cfg = self.cfg
        return {
            "voltage_cutoff": bool((out["voltage_stack"] <= cfg.V_min_stack and I_cmd > 0) or
                                   (out["voltage_stack"] >= cfg.V_max_stack and I_cmd < 0)),
            "soc_window":     bool((out["soc_true"] <= cfg.soc_min and I_cmd > 0) or
                                   (out["soc_true"] >= cfg.soc_max and I_cmd < 0)),
            "thermal_hard":   bool(out["temperature_stack"] >= cfg.T_max),
            "thermal_derate": bool(cfg.T_derate <= out["temperature_stack"] < cfg.T_max),
            "i_limit_derate": bool(out.get("i_limit", 999) > 1.0 and
                                   abs(I_cmd) > cfg.limiting_current_margin * out.get("i_limit", 999)),
            "flow_derate":    bool(out["flow_rate"] < cfg.flow_critical),
        }

    def _ren_step(self, measured, soc_cc, out):
        x = np.array([[
            measured["voltage"],
            measured["current"],
            measured["temperature"],
            out["temperature_tank"],
            out["flow_rate"],
            soc_cc,
            out["i_limit"],
            out["transport_ratio"],
        ]], dtype=np.float32)
        x_s = self.scaler.transform(x).astype(np.float32)
        x_t = torch.tensor(x_s).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            y_seq, self.z_ren = self.model(x_t, z=self.z_ren)
        return float(y_seq.squeeze())

    def step(self):
        dt     = self.cfg.dt_default
        out    = self.battery.get_outputs()
        I_safe = self.bms.apply_protection(self.I_cmd, out)
        self.bms_flags = self._bms_flags(out, self.I_cmd)

        self.battery.step(I_safe, self.Q_cmd, self.T_amb, dt)
        out      = self.battery.get_outputs()
        measured = self.sensor.measure(out)

        soc_cc  = self.cc.update(measured["current"], dt, out["capacity_nominal"])
        soc_ren = self._ren_step(measured, soc_cc, out)
        soc_true = out["soc_true"]

        self.step_count += 1

        snap = {
            "type":            "step",
            "t":               self.step_count,
            "elapsed_s":       round(time.time() - self.t_start, 1),
            "sim_time_s":      self.step_count,   # simulated seconds
            "soc_true":        round(soc_true, 5),
            "soc_cc":          round(float(soc_cc), 5),
            "soc_ren":         round(soc_ren, 5),
            "err_cc":          round(abs(float(soc_cc)  - soc_true), 5),
            "err_ren":         round(abs(soc_ren - soc_true), 5),
            "voltage":         round(measured["voltage"], 3),
            "current":         round(measured["current"], 2),
            "current_cmd":     round(self.I_cmd, 1),
            "current_safe":    round(I_safe, 2),
            "temp_stack":      round(measured["temperature"], 2),
            "temp_tank":       round(out["temperature_tank"], 2),
            "temp_ambient":    round(self.T_amb, 1),
            "flow_lpm":        round(out["flow_rate"] * 60000, 2),
            "i_limit":         round(out["i_limit"], 1),
            "transport_ratio": round(out["transport_ratio"], 4),
            "capacity_ah":     round(out["capacity_nominal"], 1),
            "bms_flags":       self.bms_flags,
            "bms_derate_pct":  round((1 - abs(I_safe) / max(abs(self.I_cmd), 1e-3)) * 100, 1)
                               if abs(self.I_cmd) > 1 else 0.0,
        }

        self.history.append({k: snap[k] for k in
            ["t","soc_true","soc_cc","soc_ren","voltage","current","err_cc","err_ren"]})
        if len(self.history) > HISTORY_LEN:
            self.history.pop(0)

        return snap

    def set_control(self, I_cmd=None, flow_lpm=None, T_amb=None, reset=False):
        if reset:
            self.__init__(); return
        if I_cmd    is not None: self.I_cmd = float(np.clip(I_cmd, -200, 200))
        if flow_lpm is not None: self.Q_cmd = float(np.clip(flow_lpm, 5, 60)) * self.cfg.LPM_to_m3s
        if T_amb    is not None: self.T_amb = float(np.clip(T_amb, 278, 323))

    def get_history(self): return self.history


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
        await ws.send_text(json.dumps({"type": "history", "history": engine.get_history()}))
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
        # Run multiple physics steps per broadcast tick
        for _ in range(STEPS_PER_TICK):
            snap = engine.step()
        snap["type"] = "step"
        await manager.broadcast(snap)
        dt = asyncio.get_event_loop().time() - t0
        await asyncio.sleep(max(0, interval - dt))
