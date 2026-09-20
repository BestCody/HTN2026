"""MuJoCo digital-twin demo that preserves the production MExT routing path.

The simulator replaces only physical camera and motor I/O.  Speech, visual
grounding, component routing, candidate planning, world prediction, safety,
confirmation, outcome verification, and journaling use the same contracts as
the hardware workflow.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from .human_interaction import HumanAwarePhysicalSession, PlanProposal
from .physical import (
    ActionChunk,
    ArmTelemetry,
    CameraFrame,
    ClarificationResult,
    EmergencyStopResult,
    MotionTrajectory,
    PhysicalAIResult,
    PipelineEvent,
    RobotState,
    WorldState,
)
from .software_integration import (
    SoftwareIntegrationConfig,
    build_software_integration_session,
    load_software_integration_config,
)

JOINT_NAMES = ("J1_BASE_YAW", "J2_SHOULDER")
GRIPPER_JOINT = "J3_GRIPPER"
GRIPPER_MIRROR_JOINT = "J3_GRIPPER_MIRROR"


@dataclass(frozen=True)
class SimulationDemoConfig:
    path: Path
    pipeline: SoftwareIntegrationConfig
    mujoco_model: Path
    output_dir: Path
    width: int
    height: int
    fps: int
    playback_speed: float


@dataclass(frozen=True)
class SimulationArtifacts:
    animation: Path
    initial_overview: Path
    final_overview: Path
    final_arm_pov: Path
    response_audio: Path | None


@dataclass(frozen=True)
class SimulationDemoRun:
    result: PhysicalAIResult
    artifacts: SimulationArtifacts
    routed_components: tuple[str, ...]
    world_models: tuple[str, ...]


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def load_simulation_demo_config(path: str | Path) -> SimulationDemoConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("simulation demo schema_version must be 1")
    simulation = raw.get("simulation")
    if not isinstance(simulation, dict):
        raise TypeError("simulation demo needs a simulation object")
    root = config_path.parent

    def resolved(name: str) -> Path:
        value = simulation.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"simulation.{name} must be a path")
        return (root / value).resolve()

    model_path = resolved("mujoco_model")
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo demo model does not exist: {model_path}")
    return SimulationDemoConfig(
        path=config_path,
        pipeline=load_software_integration_config(config_path),
        mujoco_model=model_path,
        output_dir=resolved("output_dir"),
        width=_positive_int(simulation.get("width", 640), "simulation.width"),
        height=_positive_int(simulation.get("height", 480), "simulation.height"),
        fps=_positive_int(simulation.get("fps", 10), "simulation.fps"),
        playback_speed=_positive_number(
            simulation.get("playback_speed", 1.0), "simulation.playback_speed"
        ),
    )


class MujocoDemoScene:
    """Thread-safe CAD-arm scene, renderer, and simulated sensor state."""

    def __init__(
        self,
        model_path: Path,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
        live_preview: bool = False,
    ) -> None:
        try:
            import mujoco
            import numpy as np
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError(
                "The simulation demo needs MuJoCo, NumPy, and Pillow. Run it from .venv-training."
            ) from exc
        self.mujoco = mujoco
        self.np = np
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.model_path = Path(model_path).resolve()
        self.width = _positive_int(width, "width")
        self.height = _positive_int(height, "height")
        self.fps = _positive_int(fps, "fps")
        self.live_preview = bool(live_preview)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._renderers = {
            name: mujoco.Renderer(self.model, height=self.height, width=self.width)
            for name in ("overview", "arm_pov")
        }
        self._lock = RLock()
        self._frames: list[Any] = []
        self._queued_frames: list[tuple[Any, Any, str]] = []
        self._closed = False
        self._status = "SIMULATION READY"
        self._cube_mocap_id = self._mocap_id("red_cube")
        self._cube_position = self.data.mocap_pos[self._cube_mocap_id].copy()
        self._platform_position = self._body_position("green_platform")
        self._joint_addresses = {
            name: self._joint_qpos_address(name)
            for name in (*JOINT_NAMES, GRIPPER_JOINT, GRIPPER_MIRROR_JOINT)
        }
        self._gripper_width_m = 0.042

    def _named_id(self, object_type: Any, name: str) -> int:
        value = self.mujoco.mj_name2id(self.model, object_type, name)
        if value < 0:
            raise ValueError(f"MuJoCo demo model is missing {name}")
        return int(value)

    def _joint_qpos_address(self, name: str) -> int:
        joint_id = self._named_id(self.mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[joint_id])

    def _mocap_id(self, body_name: str) -> int:
        body_id = self._named_id(self.mujoco.mjtObj.mjOBJ_BODY, body_name)
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"MuJoCo body {body_name} is not a mocap body")
        return mocap_id

    def _body_position(self, name: str) -> Any:
        body_id = self._named_id(self.mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[body_id].copy()

    @property
    def cube_position(self) -> tuple[float, float, float]:
        with self._lock:
            return tuple(float(value) for value in self._cube_position)

    @property
    def platform_position(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._platform_position)

    def camera_calibration(self, camera_name: str = "overview") -> dict[str, Any]:
        """Return OpenCV-style intrinsics/extrinsics for the rendered camera."""

        with self._lock:
            self.mujoco.mj_forward(self.model, self.data)
            camera_id = self._named_id(self.mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            position = self.data.cam_xpos[camera_id].copy()
            mujoco_rotation = self.data.cam_xmat[camera_id].reshape(3, 3).copy()
            # MuJoCo camera axes are +X right, +Y up, -Z forward. OpenCV uses
            # +X right, +Y down, +Z forward.
            camera_to_base_rotation = mujoco_rotation @ self.np.diag([1.0, -1.0, -1.0])
            transform = self.np.eye(4)
            transform[:3, :3] = camera_to_base_rotation
            transform[:3, 3] = position
            focal = self.height / (
                2.0 * math.tan(math.radians(float(self.model.cam_fovy[camera_id])) / 2.0)
            )
            return {
                "resolution_px": [self.width, self.height],
                "camera_matrix": [
                    [focal, 0.0, self.width / 2.0],
                    [0.0, focal, self.height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                "camera_to_base_matrix": transform.tolist(),
            }

    def _render(self, camera_name: str) -> Any:
        renderer = self._renderers[camera_name]
        renderer.update_scene(self.data, camera=camera_name)
        return renderer.render().copy()

    def render(self, camera_name: str = "overview") -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("MuJoCo demo scene is closed")
            self.mujoco.mj_forward(self.model, self.data)
            return self._render(camera_name)

    def capture_jpeg(self, camera_name: str = "overview") -> bytes:
        image = self.Image.fromarray(self.render(camera_name))
        payload = io.BytesIO()
        image.save(payload, format="JPEG", quality=92, optimize=True)
        return payload.getvalue()

    def _composite(self, status: str) -> Any:
        overview = self.Image.fromarray(self._render("overview"))
        arm_pov = self.Image.fromarray(self._render("arm_pov"))
        header_height = 54
        canvas = self.Image.new("RGB", (self.width * 2, self.height + header_height), (6, 10, 18))
        canvas.paste(overview, (0, header_height))
        canvas.paste(arm_pov, (self.width, header_height))
        draw = self.ImageDraw.Draw(canvas)
        draw.text((16, 8), "MExT // MUJOCO DIGITAL TWIN", fill=(90, 220, 255))
        draw.text((16, 29), status, fill=(245, 245, 245))
        draw.text((self.width + 16, 29), "ARM POV", fill=(85, 255, 145))
        return canvas

    def record_frame(self, status: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._status = status
            self.mujoco.mj_forward(self.model, self.data)
            frame = self._composite(status)
            self._frames.append(frame)
            self._show_live_frame(frame)

    def _show_live_frame(self, frame: Any, *, delay_ms: int = 1) -> None:
        if not self.live_preview:
            return
        try:
            import cv2

            cv2.imshow(
                "MExT MuJoCo digital-twin demo",
                cv2.cvtColor(self.np.asarray(frame), cv2.COLOR_RGB2BGR),
            )
            cv2.waitKey(max(1, delay_ms))
        except Exception:
            self.live_preview = False

    def queue_frame(self, status: str) -> None:
        """Snapshot control-thread state for later owner-thread rendering."""

        with self._lock:
            if self._closed:
                return
            self._queued_frames.append((self.data.qpos.copy(), self.data.mocap_pos.copy(), status))

    def set_pose(
        self,
        joints: tuple[float, ...],
        *,
        gripper_angle: float,
        cube_position: Any | None = None,
    ) -> None:
        if len(joints) != len(JOINT_NAMES):
            raise ValueError("MuJoCo demo expects base-yaw and shoulder positions")
        with self._lock:
            for name, value in zip(JOINT_NAMES, joints, strict=True):
                self.data.qpos[self._joint_addresses[name]] = float(value)
            self.data.qpos[self._joint_addresses[GRIPPER_JOINT]] = float(gripper_angle)
            self.data.qpos[self._joint_addresses[GRIPPER_MIRROR_JOINT]] = -float(gripper_angle)
            if cube_position is not None:
                value = self.np.asarray(cube_position, dtype=float)
                if value.shape != (3,) or not self.np.isfinite(value).all():
                    raise ValueError("Cube position must contain three finite values")
                self._cube_position = value.copy()
                self.data.mocap_pos[self._cube_mocap_id] = value
            self.mujoco.mj_forward(self.model, self.data)

    def robot_state(self) -> RobotState:
        with self._lock:
            joints = tuple(
                float(self.data.qpos[self._joint_addresses[name]]) for name in JOINT_NAMES
            )
            return RobotState(
                joint_positions={"left": joints},
                gripper_widths_m={"left": self._gripper_width_m},
                observed_at=time.time(),
                source="observed",
            )

    def save_artifacts(self, output_dir: Path, response_audio: bytes | None) -> SimulationArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        initial_path = output_dir / "initial_overview.jpg"
        final_path = output_dir / "final_overview.jpg"
        arm_path = output_dir / "final_arm_pov.jpg"
        animation_path = output_dir / "mext_simulation.gif"
        audio_path = output_dir / "response.wav"
        response_path = audio_path if response_audio else None
        with self._lock:
            initial = self._frames[0] if self._frames else self._composite("INITIAL SCENE")
            initial.crop((0, 54, self.width, self.height + 54)).save(
                initial_path, format="JPEG", quality=94
            )
            final_qpos = self.data.qpos.copy()
            final_mocap = self.data.mocap_pos.copy()
            rendered_motion = []
            for qpos, mocap, status in self._queued_frames:
                self.data.qpos[:] = qpos
                self.data.mocap_pos[:] = mocap
                self.mujoco.mj_forward(self.model, self.data)
                frame = self._composite(status)
                rendered_motion.append(frame)
                self._show_live_frame(frame, delay_ms=round(1000 / self.fps))
            self.data.qpos[:] = final_qpos
            self.data.mocap_pos[:] = final_mocap
            self.mujoco.mj_forward(self.model, self.data)
            self.Image.fromarray(self._render("overview")).save(
                final_path, format="JPEG", quality=94
            )
            self.Image.fromarray(self._render("arm_pov")).save(arm_path, format="JPEG", quality=94)
            frames = [initial, *rendered_motion, *self._frames[1:]]
            frames[0].save(
                animation_path,
                save_all=True,
                append_images=frames[1:],
                duration=max(20, round(1000 / self.fps)),
                loop=0,
                optimize=False,
            )
        if response_path is not None and response_audio is not None:
            response_path.write_bytes(response_audio)
        else:
            audio_path.unlink(missing_ok=True)
        return SimulationArtifacts(
            animation_path,
            initial_path,
            final_path,
            arm_path,
            response_path,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for renderer in self._renderers.values():
                renderer.close()
            if self.live_preview:
                try:
                    import cv2

                    cv2.destroyWindow("MExT MuJoCo digital-twin demo")
                except Exception:
                    pass


class MujocoCameraSource:
    """Production CameraSource adapter backed by an actual MuJoCo render."""

    def __init__(self, scene: MujocoDemoScene, camera_id: str = "sim-overview") -> None:
        self.scene = scene
        self.camera_id = camera_id

    def capture(self) -> CameraFrame:
        return CameraFrame(
            self.camera_id,
            self.scene.capture_jpeg("overview"),
            time.time(),
            "image/jpeg",
        )


class RunScopedComponentCache:
    """Stabilize confirmation reruns while retaining real model inference."""

    def __init__(self, component: Any, key: Callable[[Any], str]) -> None:
        if not callable(getattr(component, "run", None)) or not callable(key):
            raise TypeError("Cached components need run() and a key function")
        self.component = component
        self.key = key
        self._values: dict[str, Any] = {}
        self._lock = RLock()

    def run(self, request: Any) -> Any:
        cache_key = self.key(request)
        with self._lock:
            if cache_key in self._values:
                return self._values[cache_key]
        result = self.component.run(request)
        with self._lock:
            self._values.setdefault(cache_key, result)
            return self._values[cache_key]


class MujocoPerceptionFusion:
    """Use image detections for identity and the digital twin for stable poses."""

    def __init__(self, component: Any, scene: MujocoDemoScene) -> None:
        if not callable(getattr(component, "run", None)):
            raise TypeError("Perception fusion needs a component with run()")
        self.component = component
        self.scene = scene

    def run(self, request: Any) -> WorldState:
        world = self.component.run(request)
        if not isinstance(world, WorldState):
            raise TypeError("Perception component returned an invalid world state")
        positions = {
            "red cube": self.scene.cube_position,
            "green platform": self.scene.platform_position,
        }
        objects = tuple(
            replace(item, position_m=positions.get(item.label, item.position_m))
            for item in world.objects
        )
        return replace(world, objects=objects)


def _perception_cache_key(request: Any) -> str:
    digest = hashlib.sha256()
    for frame in request.frames:
        digest.update(frame.camera_id.encode("utf-8"))
        data = frame.data if isinstance(frame.data, bytes) else repr(frame.data).encode("utf-8")
        digest.update(data)
    return digest.hexdigest()


def _grounding_cache_key(request: Any) -> str:
    scene = tuple((item.id, item.label, item.position_m) for item in request.world.objects)
    stable_personal = (
        tuple(
            sorted((str(key), repr(value)) for key, value in request.personal.preferences.items())
        ),
        request.personal.accommodations,
    )
    value = (request.transcript, scene, stable_personal)
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _step_action(step_id: str) -> str:
    normalized = step_id.casefold().replace("_", "-")
    for action in ("approach", "grasp", "lift", "transfer", "release", "place"):
        if normalized.endswith(f"-{action}") or f"-{action}-" in normalized:
            return action
    return "move"


class MujocoArmDriver:
    """Execute validated production action chunks against the MuJoCo twin."""

    def __init__(
        self,
        scene: MujocoDemoScene,
        *,
        arm: str = "left",
        measured_mass_kg: float = 0.02,
        playback_speed: float = 1.0,
    ) -> None:
        if arm not in ("left", "right"):
            raise ValueError("MuJoCo arm must be left or right")
        self.scene = scene
        self.arm = arm
        self.measured_mass_kg = _positive_number(measured_mass_kg, "measured_mass_kg")
        self.playback_speed = _positive_number(playback_speed, "playback_speed")
        self.steps: list[str] = []
        self.stopped = False
        self._holding_cube = False
        self._grasp_offset = self.scene.np.zeros(3, dtype=float)

    def _telemetry(self, step_id: str, success: bool, issue: str | None = None) -> ArmTelemetry:
        return ArmTelemetry(
            self.arm,
            step_id,
            success,
            self.measured_mass_kg,
            issue,
            normal_force_n=self.measured_mass_kg * 9.81,
            slip_probability=0.02 if success else None,
        )

    def execute(self, step: Any, world: WorldState) -> ArmTelemetry:
        del world
        if self.stopped:
            return self._telemetry(step.id, False, "MuJoCo emergency stop is latched")
        self.steps.append(step.id)
        self.scene.queue_frame(f"SIM CONTROL // {str(step.action).upper()}")
        return self._telemetry(step.id, True)

    def execute_chunk(
        self,
        chunk: ActionChunk,
        trajectory: MotionTrajectory,
        world: WorldState,
    ) -> ArmTelemetry:
        del world
        if self.arm not in chunk.arms:
            raise ValueError(f"Action chunk {chunk.id} does not command {self.arm}")
        if self.stopped:
            return self._telemetry(chunk.step_id, False, "MuJoCo emergency stop is latched")
        action = _step_action(chunk.step_id)
        start_cube = self.scene.np.asarray(self.scene.cube_position, dtype=float)
        target_cube = start_cube.copy()
        target_gripper = self.scene.np.asarray(chunk.target_pose[:3], dtype=float)
        if action == "grasp":
            self._grasp_offset = target_gripper - start_cube
        elif action in ("lift", "transfer", "release", "place"):
            target_cube = target_gripper - self._grasp_offset

        points = trajectory.points
        previous_time = 0.0
        for index, point in enumerate(points, start=1):
            if self.stopped:
                return self._telemetry(
                    chunk.step_id, False, "MuJoCo emergency stop interrupted the trajectory"
                )
            joints = point.joint_positions.get(self.arm)
            if joints is None:
                raise ValueError(f"Trajectory point does not command {self.arm}")
            fraction = index / len(points)
            moving_cube = self._holding_cube or action in ("lift", "transfer", "release", "place")
            cube_position = (
                start_cube + fraction * (target_cube - start_cube) if moving_cube else start_cube
            )
            gripper_angle = 0.46 if action in ("grasp", "lift", "transfer") else 0.0
            self.scene.set_pose(
                tuple(float(value) for value in joints),
                gripper_angle=gripper_angle,
                cube_position=cube_position,
            )
            self.scene.queue_frame(f"ROUTED {chunk.skill.upper()} // EXECUTING {action.upper()}")
            delay = max(0.0, float(point.time_s) - previous_time) / self.playback_speed
            previous_time = float(point.time_s)
            if delay:
                time.sleep(delay)

        if action == "grasp":
            self._holding_cube = True
        elif action in ("release", "place"):
            self._holding_cube = False
        self.steps.append(chunk.step_id)
        return self._telemetry(chunk.step_id, True)

    def stop(self) -> None:
        self.stopped = True


def _simulation_pipeline_config(
    config: SimulationDemoConfig, scene: MujocoDemoScene
) -> SoftwareIntegrationConfig:
    workspace = copy.deepcopy(config.pipeline.workspace)
    perception = workspace.get("perception")
    if not isinstance(perception, dict):
        raise ValueError("Simulation workspace needs perception configuration")
    perception["camera_calibrations"] = {"sim-overview": scene.camera_calibration("overview")}
    return replace(config.pipeline, workspace=workspace)


def run_simulation_demo(
    config_path: str | Path,
    *,
    instruction: str = "Move the red cube onto the green platform.",
    audio_path: Path | None = None,
    event_sink: Callable[[PipelineEvent], None] | None = None,
    live_preview: bool = False,
    speak: bool = True,
) -> SimulationDemoRun:
    """Run voice-to-verified-action using MuJoCo in place of physical I/O."""

    config = load_simulation_demo_config(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    scene = MujocoDemoScene(
        config.mujoco_model,
        width=config.width,
        height=config.height,
        fps=config.fps,
        live_preview=live_preview,
    )
    pipeline_config = _simulation_pipeline_config(config, scene)
    camera = MujocoCameraSource(scene)
    driver = MujocoArmDriver(
        scene,
        measured_mass_kg=pipeline_config.assumptions["per_arm_payload_kg"],
        playback_speed=config.playback_speed,
    )
    session = None
    user_id = f"simulation-{uuid4().hex}"
    try:
        scene.record_frame("SIMULATION READY // CAMERA ONLINE")
        session, _memory, _driver, _learned_models = build_software_integration_session(
            pipeline_config,
            event_sink=event_sink,
            camera_source=camera,
            arm_driver=driver,
            control_component_id="mujoco-sim-control",
            control_model="mujoco-cad-digital-twin-v1",
            control_runtime="local",
            perception_wrapper=lambda component: MujocoPerceptionFusion(
                RunScopedComponentCache(component, _perception_cache_key), scene
            ),
            voice_grounder_wrapper=lambda component: RunScopedComponentCache(
                component, _grounding_cache_key
            ),
            profile_user_id=user_id,
        )
        interaction = HumanAwarePhysicalSession(
            session,
            user_id=user_id,
            workspace=pipeline_config.workspace,
        )
        command_audio = (
            audio_path.read_bytes()
            if audio_path is not None
            else session.system.synthesize_speech(instruction, user_id)
        )
        proposal = interaction.plan(
            audio=command_audio,
            robot_state=scene.robot_state(),
            speak=speak,
        )
        if isinstance(proposal, ClarificationResult):
            raise RuntimeError(f"Simulation command needs clarification: {proposal.question}")
        if isinstance(proposal, EmergencyStopResult):
            raise RuntimeError("Simulation command triggered the emergency stop")
        if not isinstance(proposal, PlanProposal):
            raise TypeError("Simulation planning returned an unknown result")

        confirmation_audio = session.system.synthesize_speech("yes", user_id)
        executed = interaction.respond(
            confirmation_id=proposal.confirmation_id,
            audio=confirmation_audio,
            robot_state=scene.robot_state(),
            speak=speak,
        )
        if not isinstance(executed, PhysicalAIResult):
            raise RuntimeError("Simulation confirmation did not execute a plan")
        if not executed.control.executed or not executed.control.success:
            raise RuntimeError("MuJoCo control failed: " + "; ".join(executed.control.issues))
        if executed.outcome is None or executed.outcome.status != "succeeded":
            raise RuntimeError("Simulation outcome was not verified as succeeded")
        scene.record_frame("SIMULATION COMPLETE // CAMERA VERIFIED")
        artifacts = scene.save_artifacts(config.output_dir, executed.response_audio)
        return SimulationDemoRun(
            result=executed,
            artifacts=artifacts,
            routed_components=tuple(item.component_id for item in executed.routing),
            world_models=tuple(
                dict.fromkeys(item.model_kind for item in executed.world_predictions)
            ),
        )
    finally:
        if session is not None:
            session.close()
        scene.close()
