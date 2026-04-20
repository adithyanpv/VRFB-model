# vrfb_bms/server.py
"""
VRFB BMS — Real-Time Web Server  v2.0  (Pure Observer)
=======================================================

CHANGES FROM v1.x
-----------------
  - input_dim reduced from 9 to 6 (pure observer architecture)
  - PI observer / bias integrator completely removed from _ren_step
  - cumulative_ah_norm removed (OOD after training horizon)
  - bias_est removed (caused runaway feedback at long duration)
  - transport_ratio_approx removed (redundant, derived)
  - _ren_step now uses 6 raw physical features only
  - Final SOC = clip(soc_cc + ren_correction, 0, 1)
  - Transition dampening retained (suppresses post-rest OCV spike)
  - Dual EMA retained (fast tau=30 for BMS, display tau=120 for chart)
  - soc_ren_hybrid broadcast retained (fast EMA on chart)

USAGE:
  uvicorn vrfb_bms.server:app --reload --port 8000
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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

HIDDEN_DIM     = 128
ALPHA          = 0.5
DEVICE         = torch.device("cpu")
SIM_HZ         = 1.0
STEPS_PER_TICK = 10
HISTORY_LEN    = 300
EMA_TAU_FAST   = 30    # responsive — BMS decisions + soc_ren_hybrid on chart
EMA_TAU_DISP   = 120   # smooth visual — soc_ren on chart


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

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


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class SimulationEngine:

    def __init__(self):
        self.cfg = VRFBConfig()

        proj_root   = os.path.dirname(BASE_DIR)
        model_path  = os.path.join(proj_root, "ren", "ren_soc_best.pth")
        scaler_path = os.path.join(proj_root, "ren", "scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"REN model not found: {model_path}\nRun train_ren.py first."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found: {scaler_path}\nRun train_ren.py first."
            )

        # Pure observer: 6 features — voltage, current, T_stack, T_tank, flow, soc_cc
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

        self._init_sim()
        self._reset_ren_state()

    # ─────────────────────────────────────────────────────────────────────────

    def _init_sim(self):
        self.battery          = VRFB(self.cfg)
        self.bms              = BMSController(self.cfg)
        self.sensor           = SensorModel(self.cfg)
        self.cc               = CoulombCounter(self.cfg)
        self.cc.initialize(self.cfg.initial_soc)
        self.I_cmd            = 0.0
        self.Q_cmd            = self.cfg.initial_flow
        self.T_amb            = self.cfg.initial_temperature
        self.history          = []
        self.step_count       = 0
        self.t_start          = time.time()
        self._prev_bms_mode   = "standby"
        self._transition_steps = 0
        # No bias integrator, no cumulative_ah — pure observer

    def _reset_ren_state(self):
        with torch.no_grad():
            self.z_ren = self.model.z0.detach().clone()
        self.soc_ren_ema_fast  = self.cfg.initial_soc
        self.soc_ren_ema_disp  = self.cfg.initial_soc
        self._prev_bms_mode    = "standby"
        self._transition_steps = 0

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
    # REN INFERENCE  — Pure Observer, 6 features, no feedback loop
    # ─────────────────────────────────────────────────────────────────────────

    def _ren_step(self, measured: dict, soc_cc: float) -> float:
        """
        6-feature pure observer inference.

        Features: voltage, current, T_stack, T_tank, flow_rate, soc_cc

        No PI observer. No bias_est. No cumulative_ah_norm.
        The REN hidden state z is the sole non-linear integrator.

        final_soc = clip(soc_cc + ren_correction, 0, 1)
        """
       # Ohmic-corrected OCV (same constant used in dataset_gen.py)
        _R_STACK = (0.0015 + 0.0005) * 40   # 0.08 Ω
        v_ocv    = measured["voltage"] - measured["current"] * _R_STACK

        x = np.array([[
            measured["voltage"],
            measured["current"],
            measured["temperature"],
            measured["temperature_tank"],
            measured["flow_rate"],
            float(soc_cc),
            float(v_ocv),
        ]], dtype=np.float32)

        x_s   = self.scaler.transform(x).astype(np.float32)
        x_t   = torch.tensor(x_s).unsqueeze(0).to(DEVICE)

        # Gate uses raw Amps — reconstruct from measured current
        I_raw = torch.tensor([[[measured["current"]]]], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            y_seq, self.z_ren = self.model(x_t, z=self.z_ren, x_raw=I_raw)

        correction = float(y_seq.squeeze())
        return float(np.clip(float(soc_cc) + correction, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE SIMULATION STEP
    # ─────────────────────────────────────────────────────────────────────────

    def step(self) -> dict:
        dt  = self.cfg.dt_default
        out = self.battery.get_outputs()

        self.bms.update_mode(self.I_cmd, out, dt)
        flow_override = self.bms.get_flow_override()
        Q_eff  = flow_override if flow_override is not None else self.Q_cmd
        I_safe = self.bms.apply_protection(self.I_cmd, out)
        flags  = self._bms_flags(out, self.I_cmd, I_safe)

        self.battery.step(I_safe, Q_eff, self.T_amb, dt)
        out = self.battery.get_outputs()

        measured = self.sensor.measure(out)
        soc_cc   = self.cc.update(measured["current"], dt, out["capacity_nominal"])

        # REN inference — pure observer, no feedback
        raw = self._ren_step(measured, soc_cc)

        # Transition dampening: suppress post-rest OCV spike when mode changes
        cur_mode = self.bms.mode_name
        if cur_mode != self._prev_bms_mode:
            self._transition_steps = 0
        self._prev_bms_mode = cur_mode
        self._transition_steps += 1

        if self._transition_steps < 60 and cur_mode in ("discharge", "charge"):
            blend = self._transition_steps / 60.0
            raw   = (1.0 - blend) * self.soc_ren_ema_disp + blend * raw

        # Dual EMA
        a_fast = 1.0 / EMA_TAU_FAST
        a_disp = 1.0 / EMA_TAU_DISP
        self.soc_ren_ema_fast = (1.0 - a_fast) * self.soc_ren_ema_fast + a_fast * raw
        self.soc_ren_ema_disp = (1.0 - a_disp) * self.soc_ren_ema_disp + a_disp * raw

        soc_ren        = float(np.clip(self.soc_ren_ema_disp, 0.0, 1.0))  # dashboard
        soc_ren_hybrid = float(np.clip(self.soc_ren_ema_fast, 0.0, 1.0))  # BMS / chart
        soc_true       = out["soc_true"]

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
            # SOC estimators
            "soc_true"            : round(soc_true, 5),
            "soc_cc"              : round(float(soc_cc), 5),
            "soc_ren"             : round(soc_ren, 5),           # display EMA tau=120
            "soc_ren_hybrid"      : round(soc_ren_hybrid, 5),    # fast EMA tau=30
            "soc_ren_raw"         : round(raw, 5),               # pre-EMA (diagnostic)
            # Errors
            "err_cc"              : round(abs(float(soc_cc) - soc_true), 5),
            "err_ren"             : round(abs(soc_ren - soc_true), 5),
            "err_ren_hybrid"      : round(abs(soc_ren_hybrid - soc_true), 5),
            # Half-cell SOC
            "soc_neg"             : round(out["soc_neg"],       5),
            "soc_pos"             : round(out["soc_pos"],       5),
            "soc_system"          : round(out["soc_system"],    5),
            "soc_imbalance"       : round(out["soc_imbalance"], 5),
            # Electrical
            "voltage"             : round(measured["voltage"],  3),
            "current"             : round(measured["current"],  2),
            "current_cmd"         : round(self.I_cmd,           1),
            "current_safe"        : round(I_safe,               2),
            "i_limit"             : round(out["i_limit"],       1),
            "transport_ratio"     : round(out["transport_ratio"], 4),
            # Thermal
            "temp_stack"          : round(measured["temperature"],      2),
            "temp_tank"           : round(measured["temperature_tank"], 2),
            "temp_ambient"        : round(self.T_amb,                   1),
            # Hydraulic
            "flow_lpm"            : round(out["flow_rate"] * 60000,     2),
            "flow_override"       : flow_override is not None,
            # Degradation
            "capacity_ah"         : round(out["capacity_nominal"],      1),
            # BMS
            "bms_mode"            : self.bms.mode_name,
            "bms_standby_timer_s" : round(self.bms.standby_timer, 1),
            "bms_startup_pct"     : round(self.bms.startup_progress * 100, 1),
            "bms_drain_active"    : self.bms.in_drain,
            "bms_flags"           : flags,
            "bms_derate_pct"      : round(derate_pct, 1),
        }

        self.history.append({
            k: snap[k] for k in [
                "t", "soc_true", "soc_cc", "soc_ren", "soc_ren_hybrid",
                "voltage", "current", "err_cc", "err_ren", "err_ren_hybrid",
                "soc_neg", "soc_pos", "soc_imbalance",
            ]
        })
        if len(self.history) > HISTORY_LEN:
            self.history.pop(0)

        return snap

    def set_control(
        self,
        I_cmd:    Optional[float] = None,
        flow_lpm: Optional[float] = None,
        T_amb:    Optional[float] = None,
        reset:    bool = False,
    ):
        if reset:
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


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

manager = ConnectionManager()
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
            "type": "history", "history": engine.get_history()
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