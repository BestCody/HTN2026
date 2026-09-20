"""Authenticated laptop-to-Pi robot-control link with local motor authority."""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import re
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from .cloud import JsonHttpClient
from .physical import (
    ActionChunk,
    ArmTelemetry,
    CandidatePlan,
    ControlInput,
    ControlReport,
    DetectedObject,
    FinalPlan,
    GraspStability,
    MotionTrajectory,
    PlanStep,
    PolicyPlan,
    SimulationOutcome,
    TrajectoryPoint,
    WorldState,
)
from .production import LazyPCA9685Control
from .robot_config import RobotModel

ROBOT_LINK_SCHEMA_VERSION = 1
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
MAX_ROBOT_REQUEST_BYTES = 2 * 1024 * 1024


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Robot-link {name} must be a JSON object")
    return value


def _step(value: Any) -> PlanStep:
    value = _mapping(value, "step")
    return PlanStep(
        value["id"],
        value["action"],
        tuple(value["arms"]),
        value["duration_seconds"],
        value.get("target_object_id"),
        value.get("parameters"),
    )


def _chunk(value: Any) -> ActionChunk:
    value = _mapping(value, "action chunk")
    return ActionChunk(
        value["id"],
        value["step_id"],
        value["skill"],
        tuple(value["arms"]),
        value["duration_seconds"],
        value.get("target_object_id"),
        tuple(value["target_pose"]),
        value["gripper_width_m"],
        value["force_limit_n"],
    )


def _world(value: Any) -> WorldState:
    value = _mapping(value, "world")
    objects = tuple(
        DetectedObject(
            item["id"],
            item["label"],
            item["confidence"],
            tuple(item["position_m"]),
            item.get("estimated_mass_kg"),
            item.get("attributes"),
        )
        for item in value.get("objects", ())
    )
    return WorldState(
        objects,
        value.get("workspace", {}),
        tuple(value.get("hazards", ())),
        value.get("observed_at", 0.0),
        value.get("geometry", {}),
        value.get("coordinate_frame", "robot_base"),
        value.get("up_axis", "z"),
    )


def _trajectory(value: Any) -> MotionTrajectory:
    value = _mapping(value, "trajectory")
    points = tuple(
        TrajectoryPoint(
            item["time_s"],
            {arm: tuple(joints) for arm, joints in item["joint_positions"].items()},
            item.get("chunk_id"),
            item.get("gripper_widths_m", {}),
        )
        for item in value["points"]
    )
    return MotionTrajectory(value["plan_id"], points, value["duration_seconds"])


def _validate_pi_trajectory(
    trajectory: MotionTrajectory,
    chunk: ActionChunk,
    model: RobotModel,
) -> None:
    """Independently enforce calibrated position and velocity limits on the Pi."""

    previous: dict[str, tuple[float, tuple[float, ...], float]] = {}
    expected_arms = set(chunk.arms)
    for point in trajectory.points:
        if set(point.joint_positions) != expected_arms:
            raise ValueError("Pi trajectory arms do not match the action chunk")
        if set(point.gripper_widths_m) != expected_arms:
            raise ValueError("Pi trajectory gripper commands do not match the action chunk")
        for arm in chunk.arms:
            joints = point.joint_positions[arm]
            if len(joints) != len(model.kinematic_joints):
                raise ValueError("Pi trajectory joint count does not match the robot model")
            for joint, value in zip(model.kinematic_joints, joints, strict=True):
                if not joint.lower_rad <= value <= joint.upper_rad:
                    raise ValueError(f"Pi trajectory exceeds calibrated limit for {joint.name}")
            width = point.gripper_widths_m[arm]
            if not 0 <= width <= model.max_gripper_width_m:
                raise ValueError("Pi trajectory exceeds the calibrated gripper width")
            if arm in previous:
                previous_time, previous_joints, previous_width = previous[arm]
                delta = point.time_s - previous_time
                if delta <= 0:
                    raise ValueError("Pi trajectory timestamps must increase")
                for joint, start, end in zip(
                    model.kinematic_joints, previous_joints, joints, strict=True
                ):
                    speed = abs(end - start) / delta
                    if speed > joint.max_velocity_rad_s + 1e-9:
                        raise ValueError(
                            f"Pi trajectory exceeds calibrated velocity for {joint.name}"
                        )
                gripper_speed = abs(width - previous_width) / delta
                if gripper_speed > model.max_gripper_velocity_m_s + 1e-9:
                    raise ValueError("Pi trajectory exceeds calibrated gripper velocity")
            previous[arm] = (point.time_s, joints, width)
    if not math.isclose(
        trajectory.duration_seconds,
        chunk.duration_seconds,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("Pi trajectory duration does not match the action chunk")


def decode_control_command(value: Any, model: RobotModel) -> tuple[str, ControlInput]:
    """Decode and revalidate one bounded action chunk on the Pi."""

    value = _mapping(value, "command")
    if value.get("schema_version") != ROBOT_LINK_SCHEMA_VERSION:
        raise ValueError("Robot-link schema_version is unsupported")
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise ValueError("Robot-link request_id is invalid")
    if value.get("robot_model_id") != model.model_id:
        raise ValueError("Robot-link model ID does not match the Pi robot")
    if value.get("robot_source_sha256") != model.source_sha256:
        raise ValueError("Robot-link geometry digest does not match the Pi robot")
    plan_id = value.get("plan_id")
    step = _step(value.get("step"))
    chunk = _chunk(value.get("chunk"))
    if chunk.step_id != step.id:
        raise ValueError("Robot-link action chunk does not match its plan step")
    candidate = CandidatePlan(plan_id, (step,), value.get("rationale", "validated action chunk"))
    simulation_raw = _mapping(value.get("simulation"), "simulation")
    simulation = SimulationOutcome(
        simulation_raw["plan_id"],
        simulation_raw["score"],
        simulation_raw["safe"],
        simulation_raw["horizon_seconds"],
        tuple(simulation_raw.get("risks", ())),
        simulation_raw.get("predicted"),
    )
    final = FinalPlan(candidate, simulation, value.get("summary", "validated action chunk"))
    policy = PolicyPlan(plan_id, value["policy"], (chunk,))
    tactile_raw = value.get("tactile")
    tactile = (
        None
        if tactile_raw is None
        else GraspStability(tactile_raw["stable"], tactile_raw["score"])
    )
    trajectory = _trajectory(value.get("trajectory"))
    if model.motion_ready:
        _validate_pi_trajectory(trajectory, chunk, model)
    return request_id, ControlInput(
        final,
        _world(value.get("world")),
        True,
        policy,
        trajectory,
        tactile,
    )


def _control_report(value: Any, *, plan_id: str, request_id: str) -> ControlReport:
    value = _mapping(value, "response")
    if value.get("schema_version") != ROBOT_LINK_SCHEMA_VERSION:
        raise ValueError("Pi robot response schema_version is unsupported")
    if value.get("request_id") != request_id:
        raise ValueError("Pi robot response request_id does not match")
    control = _mapping(value.get("control"), "control report")
    if control.get("plan_id") != plan_id:
        raise ValueError("Pi robot response plan ID does not match")
    telemetry = tuple(
        ArmTelemetry(
            item["arm"],
            item["step_id"],
            item["success"],
            item.get("measured_mass_kg"),
            item.get("issue"),
            item.get("normal_force_n"),
            item.get("slip_probability"),
        )
        for item in control.get("telemetry", ())
    )
    return ControlReport(
        control["plan_id"],
        control["executed"],
        control["success"],
        telemetry,
        tuple(control.get("issues", ())),
    )


class PiRobotControlComponent:
    """Laptop-side component that sends only prevalidated chunks to the Pi."""

    def __init__(
        self,
        base_url: str,
        token: str,
        model: RobotModel,
        *,
        http: JsonHttpClient | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Pi robot URL must be non-empty")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Pi robot bearer token must be non-empty")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.model = model
        self.http = http or JsonHttpClient(
            timeout_seconds=300,
            attempts=1,
            allow_private_http=True,
        )

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _stop_after_failure(self) -> tuple[str, ...]:
        try:
            response = self.http.post(
                f"{self.base_url}/v1/stop",
                {"schema_version": ROBOT_LINK_SCHEMA_VERSION, "request_id": uuid4().hex},
                self._headers,
            )
            issues = response.get("issues", ()) if isinstance(response, dict) else ()
            return tuple(issues)
        except Exception as exc:
            return (f"Pi emergency-stop request failed: {str(exc) or type(exc).__name__}",)

    def emergency_stop(self) -> tuple[str, ...]:
        return self._stop_after_failure()

    def stop(self) -> None:
        issues = self.emergency_stop()
        if issues:
            raise RuntimeError("; ".join(issues))

    def close(self) -> None:
        return None

    def run(self, request: ControlInput) -> ControlReport:
        if not isinstance(request, ControlInput):
            raise TypeError("Pi robot control expects ControlInput")
        if not request.execute:
            return ControlReport(request.plan.candidate.id, False, True, ())
        if request.policy is None or request.trajectory is None:
            raise ValueError("Pi robot execution requires a validated policy and trajectory")
        chunks_by_step = {
            step.id: tuple(chunk for chunk in request.policy.chunks if chunk.step_id == step.id)
            for step in request.plan.candidate.steps
        }
        telemetry: list[ArmTelemetry] = []
        issues: list[str] = []
        completed_actions: list[str] = []
        for step in request.plan.candidate.steps:
            try:
                if request.execution_guard is not None:
                    request.execution_guard(step, tuple(completed_actions))
            except Exception as exc:
                issues.append(str(exc).strip() or type(exc).__name__)
                issues.extend(self._stop_after_failure())
                return ControlReport(
                    request.plan.candidate.id,
                    True,
                    False,
                    tuple(telemetry),
                    tuple(dict.fromkeys(issues)),
                )
            for chunk in chunks_by_step[step.id]:
                request_id = uuid4().hex
                chunk_step = PlanStep(
                    step.id,
                    step.action,
                    step.arms,
                    chunk.duration_seconds,
                    step.target_object_id,
                    step.parameters,
                )
                payload = {
                    "schema_version": ROBOT_LINK_SCHEMA_VERSION,
                    "request_id": request_id,
                    "robot_model_id": self.model.model_id,
                    "robot_source_sha256": self.model.source_sha256,
                    "plan_id": request.plan.candidate.id,
                    "rationale": request.plan.candidate.rationale,
                    "summary": request.plan.summary,
                    "step": asdict(chunk_step),
                    "chunk": asdict(chunk),
                    "policy": request.policy.policy,
                    "trajectory": asdict(request.trajectory.for_chunk(chunk.id)),
                    "simulation": asdict(request.plan.simulation),
                    "world": asdict(request.world),
                    "tactile": asdict(request.tactile) if request.tactile is not None else None,
                }
                try:
                    response = self.http.post(
                        f"{self.base_url}/v1/control", payload, self._headers
                    )
                    report = _control_report(
                        response,
                        plan_id=request.plan.candidate.id,
                        request_id=request_id,
                    )
                except Exception as exc:
                    issues.append(str(exc).strip() or type(exc).__name__)
                    issues.extend(self._stop_after_failure())
                    return ControlReport(
                        request.plan.candidate.id,
                        True,
                        False,
                        tuple(telemetry),
                        tuple(dict.fromkeys(issues)),
                    )
                telemetry.extend(report.telemetry)
                if not report.success:
                    issues.extend(report.issues)
                    issues.extend(self._stop_after_failure())
                    return ControlReport(
                        request.plan.candidate.id,
                        True,
                        False,
                        tuple(telemetry),
                        tuple(dict.fromkeys(issues)),
                    )
            completed_actions.append(step.action)
        return ControlReport(
            request.plan.candidate.id,
            True,
            bool(telemetry),
            tuple(telemetry),
        )


class PiRobotControllerApplication:
    """Pi-side command gate; starts without touching I2C until execution is enabled."""

    def __init__(self, model: RobotModel, *, enable_motion: bool = False) -> None:
        if not isinstance(enable_motion, bool):
            raise TypeError("enable_motion must be boolean")
        self.model = model
        self.enable_motion = enable_motion
        self._controller = (
            LazyPCA9685Control(model) if enable_motion and model.motion_ready else None
        )
        self._seen: set[str] = set()
        self._seen_order: list[str] = []
        self._state_lock = Lock()
        self._motion_lock = Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready" if self._controller is not None else "locked",
            "robot_model_id": self.model.model_id,
            "robot_source_sha256": self.model.source_sha256,
            "motion_enabled": self.enable_motion,
            "motion_ready": self.model.motion_ready,
            "installed_arms": list(self.model.servo_controller.installed_arms),
            "i2c_address": self.model.servo_controller.i2c_address,
            "pwm_frequency_hz": self.model.servo_controller.pwm_frequency_hz,
            "readiness_issues": list(self.model.readiness_issues),
        }

    def control(self, value: Any) -> dict[str, Any]:
        if not self.enable_motion:
            raise PermissionError("Pi robot motion is disabled; calibrate and restart explicitly")
        if self._controller is None:
            raise PermissionError("Pi robot model is not motion-ready")
        request_id, command = decode_control_command(value, self.model)
        with self._state_lock:
            if request_id in self._seen:
                raise RuntimeError("Duplicate robot-control request was rejected")
            self._seen.add(request_id)
            self._seen_order.append(request_id)
            if len(self._seen_order) > 1024:
                self._seen.discard(self._seen_order.pop(0))
        if not self._motion_lock.acquire(blocking=False):
            raise RuntimeError("Pi robot controller is busy")
        try:
            report = self._controller.run(command)
        finally:
            self._motion_lock.release()
        return {
            "schema_version": ROBOT_LINK_SCHEMA_VERSION,
            "request_id": request_id,
            "robot_model_id": self.model.model_id,
            "control": asdict(report),
        }

    def emergency_stop(self) -> tuple[str, ...]:
        if self._controller is None:
            return ()
        return self._controller.emergency_stop()

    def close(self) -> None:
        if self._controller is not None:
            self._controller.close()


def pi_robot_handler_type(
    application: PiRobotControllerApplication,
    token: str,
) -> type[BaseHTTPRequestHandler]:
    """Build the authenticated Pi HTTP handler around one controller application."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "MoIRAPiRobot/1"

        def _json(self, status: HTTPStatus, value: Any) -> None:
            encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied, f"Bearer {token}")

        def _request(self) -> Any:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_ROBOT_REQUEST_BYTES:
                raise ValueError("request body size is invalid")
            return json.loads(self.rfile.read(length))

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            self._json(HTTPStatus.OK, application.health())

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            try:
                value = self._request()
                if self.path == "/v1/control":
                    response = application.control(value)
                elif self.path == "/v1/stop":
                    request_id = value.get("request_id") if isinstance(value, dict) else None
                    response = {
                        "schema_version": ROBOT_LINK_SCHEMA_VERSION,
                        "request_id": request_id,
                        "issues": application.emergency_stop(),
                    }
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                    return
            except PermissionError as exc:
                self._json(HTTPStatus.LOCKED, {"error": str(exc)})
                return
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            except Exception as exc:
                application.emergency_stop()
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": type(exc).__name__, "message": str(exc)},
                )
                return
            self._json(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} {format % args}")

    return Handler


def controller_main(argv: list[str] | None = None) -> int:
    """Run the Pi service; motor output remains locked unless explicitly enabled."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Permit calibrated motion; omitted by default so the service remains locked",
    )
    parser.add_argument("--env-file", type=Path, default=Path.cwd() / ".env")
    args = parser.parse_args(argv)
    load_dotenv(args.env_file, override=False)
    token = os.environ.get("MOIRA_ROBOT_TOKEN")
    if not token:
        parser.error("MOIRA_ROBOT_TOKEN is required")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    from .robot_config import load_robot_model

    model = load_robot_model(args.model, verify_source=True)
    application = PiRobotControllerApplication(model, enable_motion=args.enable_motion)
    server = ThreadingHTTPServer((args.host, args.port), pi_robot_handler_type(application, token))
    print(
        f"MoIRA Pi robot service listening on http://{args.host}:{args.port}; "
        f"motion_enabled={application.enable_motion}; motion_ready={model.motion_ready}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        application.emergency_stop()
        application.close()
        server.server_close()
    return 0
