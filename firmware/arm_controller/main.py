from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import ArmConfig, default_config_path
from .servo_driver import make_driver
from .state import ArmState

log = logging.getLogger("arm_controller")


class Hub:
    """Tracks connected websocket clients and fans out state broadcasts."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients)
        text = json.dumps(payload)
        for ws in targets:
            try:
                await ws.send_text(text)
            except Exception:
                await self.remove(ws)


def _snapshot_payload(state: ArmState) -> dict[str, Any]:
    snap = state.snapshot()
    return {
        "type": "state",
        "enabled": snap.enabled,
        "hardware": snap.hardware,
        "current": snap.current,
        "target": snap.target,
        "limits": {k: list(v) for k, v in snap.limits.items()},
        "modes": snap.modes,
        "speeds": snap.speeds,
    }


async def _control_loop(state: ArmState, hz: float, stop: asyncio.Event) -> None:
    period = 1.0 / hz
    while not stop.is_set():
        await state.tick()
        await asyncio.sleep(period)


async def _broadcast_loop(state: ArmState, hub: Hub, hz: float, stop: asyncio.Event) -> None:
    period = 1.0 / hz
    while not stop.is_set():
        await hub.broadcast(_snapshot_payload(state))
        await asyncio.sleep(period)


def create_app(config_path: Path) -> FastAPI:
    config = ArmConfig.load(config_path)
    driver = make_driver(config.i2c_address, config.pwm_frequency)
    state = ArmState(config, driver)
    hub = Hub()
    stop = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        control_task = asyncio.create_task(_control_loop(state, config.control_rate_hz, stop))
        broadcast_task = asyncio.create_task(_broadcast_loop(state, hub, config.broadcast_rate_hz, stop))
        try:
            yield
        finally:
            stop.set()
            for t in (control_task, broadcast_task):
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            state.shutdown()

    app = FastAPI(title="Arm Controller", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        return JSONResponse(
            {
                "joints": {
                    name: {
                        "channel": j.channel,
                        "min_deg": j.min_deg,
                        "max_deg": j.max_deg,
                        "home_deg": j.home_deg,
                        "mode": j.mode,
                    }
                    for name, j in config.joints.items()
                },
                "hardware": driver.hardware,
            }
        )

    @app.get("/api/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(_snapshot_payload(state))

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await hub.add(ws)
        await ws.send_text(json.dumps(_snapshot_payload(state)))
        try:
            while True:
                raw = await ws.receive_text()
                await _handle_message(raw, state)
        except WebSocketDisconnect:
            pass
        finally:
            await hub.remove(ws)

    dashboard_dist = Path(os.environ.get("ARM_DASHBOARD_DIST", "")).expanduser()
    if dashboard_dist and dashboard_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dashboard_dist), html=True), name="dashboard")

    return app


async def _handle_message(raw: str, state: ArmState) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("bad json from client: %r", raw[:120])
        return
    kind = msg.get("type")
    if kind == "set_target":
        joint = msg.get("joint")
        deg = msg.get("degrees")
        if isinstance(joint, str) and isinstance(deg, (int, float)):
            try:
                await state.set_target(joint, float(deg))
            except KeyError:
                log.warning("unknown joint: %s", joint)
    elif kind == "set_targets":
        joints = msg.get("joints") or {}
        if isinstance(joints, dict):
            cleaned = {k: float(v) for k, v in joints.items() if isinstance(v, (int, float))}
            await state.set_targets(cleaned)
    elif kind == "set_speed":
        joint = msg.get("joint")
        speed = msg.get("speed")
        if isinstance(joint, str) and isinstance(speed, (int, float)):
            try:
                await state.set_speed(joint, float(speed))
            except (KeyError, ValueError):
                log.warning("invalid speed request: %s %r", joint, speed)
    elif kind == "home":
        await state.home()
    elif kind == "set_enabled":
        await state.set_enabled(bool(msg.get("enabled")))
    else:
        log.warning("unknown message type: %s", kind)


def run() -> None:
    parser = argparse.ArgumentParser(description="Arm controller HTTP + WS server.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import uvicorn

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    run()
