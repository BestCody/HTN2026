"""Configuration-driven Raspberry Pi and Baseten physical runtime."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from .cloud import (
    BasetenComponent,
    BasetenEndpoint,
    BasetenModelAPI,
    JsonComponent,
    JsonEndpoint,
    JsonHttpClient,
    RemoteComponentRouter,
    load_runtime_environment,
)
from .brand import DEFAULT_WAKE_WORD, DISPLAY_NAME
from .components import ComponentDecision, ComponentRegistry, Layer, ModelComponent
from .contracts import decode_physical_response
from .edge_components import LoadFeedbackComponent, SQLitePersonalMemory
from .episode_dataset import validate_camera_calibration
from .learned_specialists import (
    AcceptedSimulationModels,
    LearnedForwardDynamics,
    LearnedWaypointKinematics,
)
from .model_api_components import ModelAPIScenePerception, ModelAPIVoiceGrounder
from .pca9685 import PCA9685PulseDevice, pca9685_installed_arm_drivers
from .physical import (
    BimanualControlComponent,
    ControlInput,
    ControlReport,
    PhysicalAI,
    PipelineEvent,
)
from .pi import PiRuntimeProfile
from .robot_config import RobotModel, load_robot_model
from .session import JsonlRunJournal, PhysicalSession
from .simulation_learning import load_simulation_learning_config
from .specialists import (
    AnalyticGraspPlanner,
    BoundedTrajectoryPlanner,
    ConservativeCollisionChecker,
    HardSafetyRiskModel,
    ModelBasedTaskReward,
    PlanarBimanualIK,
    SpecializedManipulationPolicy,
    StateSpaceWorldModel,
    TactileSignalModel,
    TelemetryFailureClassifier,
    TelemetryLoadEstimator,
    TelemetryOutcomeVerifier,
    WorldModelErrorTracker,
)

SINGLE_ARM_PICK_PLACE_COMPONENTS = frozenset(
    {
        "baseten-vision-scene",
        "baseten-grasp-pose",
        "baseten-stt",
        "baseten-voice-nlp",
        "baseten-tts",
        "baseten-task-planner",
        "baseten-waypoint-policy",
        "baseten-forward-dynamics",
        "baseten-rigid-world",
        "baseten-grasp-contact-world",
        "baseten-task-reward",
        "baseten-outcome-verifier",
    }
)


@dataclass(frozen=True)
class RemoteEndpointConfig:
    transport: Literal["baseten_deployment", "baseten_model_api", "json_http"]
    id_env: str | None = None
    entity: Literal["model", "chain"] | None = None
    environment_env: str = "BASETEN_ENVIRONMENT"
    default_environment: str = "development"
    model_env: str | None = None
    default_model: str | None = None
    url_env: str | None = None
    token_env: str | None = None

    def __post_init__(self) -> None:
        if self.transport not in (
            "baseten_deployment",
            "baseten_model_api",
            "json_http",
        ):
            raise ValueError("unsupported remote endpoint transport")
        if self.transport == "baseten_deployment":
            if not self.id_env or self.entity not in ("model", "chain"):
                raise ValueError("Baseten deployments need id_env and model/chain entity")
            if not self.environment_env.strip() or not self.default_environment.strip():
                raise ValueError("Baseten deployment environments must be non-empty")
        elif self.transport == "baseten_model_api":
            if not self.default_model and not self.model_env:
                raise ValueError("Baseten Model API endpoints need a model setting")
            if self.entity is not None or self.id_env is not None or self.url_env is not None:
                raise ValueError("Model API endpoints cannot contain deployment or URL settings")
        elif not self.url_env:
            raise ValueError("JSON HTTP endpoints need url_env")

    @property
    def required_environment_variable(self) -> str | None:
        if self.transport == "baseten_deployment":
            return self.id_env
        if self.transport == "json_http":
            return self.url_env
        return None


@dataclass(frozen=True)
class RouterConfig:
    id_env: str
    environment_env: str = "BASETEN_ROUTER_ENVIRONMENT"
    default_environment: str = "development"

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.id_env, self.environment_env, self.default_environment)
        ):
            raise ValueError("Router environment settings must be non-empty strings")


@dataclass(frozen=True)
class CameraRuntimeConfig:
    transport: Literal["opencv", "lan_http"]
    camera_id: str
    device_env: str | None = None
    url_env: str | None = None
    token_env: str | None = None
    rotation_degrees: Literal[0, 90, 180, 270] = 0

    def __post_init__(self) -> None:
        if self.transport not in ("opencv", "lan_http"):
            raise ValueError("unsupported camera transport")
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        if self.rotation_degrees not in (0, 90, 180, 270):
            raise ValueError("camera rotation_degrees must be 0, 90, 180, or 270")
        if self.transport == "lan_http":
            if (
                not isinstance(self.url_env, str)
                or not self.url_env.strip()
                or not isinstance(self.token_env, str)
                or not self.token_env.strip()
            ):
                raise ValueError("LAN camera transport needs url_env and token_env")
            if self.device_env is not None:
                raise ValueError("LAN camera transport cannot contain device_env")
        if self.transport == "opencv":
            if not isinstance(self.device_env, str) or not self.device_env.strip():
                raise ValueError("OpenCV camera transport needs device_env")
            if self.url_env is not None or self.token_env is not None:
                raise ValueError("OpenCV camera transport cannot contain LAN settings")


@dataclass(frozen=True)
class ControllerRuntimeConfig:
    transport: Literal["local_pca9685", "pi_http"]
    url_env: str | None = None
    token_env: str | None = None

    def __post_init__(self) -> None:
        if self.transport not in ("local_pca9685", "pi_http"):
            raise ValueError("unsupported robot-controller transport")
        if self.transport == "pi_http" and (
            not isinstance(self.url_env, str)
            or not self.url_env.strip()
            or not isinstance(self.token_env, str)
            or not self.token_env.strip()
        ):
            raise ValueError("Pi HTTP controller needs url_env and token_env")
        if self.transport == "local_pca9685" and (
            self.url_env is not None or self.token_env is not None
        ):
            raise ValueError("Local PCA9685 controller cannot contain HTTP settings")


@dataclass(frozen=True)
class AudioRuntimeConfig:
    transport: Literal["ffmpeg_dshow"]
    device_env: str
    sample_rate_hz: int = 16_000
    channels: int = 1
    command_duration_seconds: int = 5
    wake_word: str = DEFAULT_WAKE_WORD
    wake_window_seconds: int = 4

    def __post_init__(self) -> None:
        if self.transport != "ffmpeg_dshow":
            raise ValueError("unsupported laptop audio transport")
        if not isinstance(self.device_env, str) or not self.device_env.strip():
            raise ValueError("laptop audio transport needs device_env")
        if (
            not isinstance(self.sample_rate_hz, int)
            or isinstance(self.sample_rate_hz, bool)
            or not 8_000 <= self.sample_rate_hz <= 48_000
        ):
            raise ValueError("laptop audio sample rate must be in [8000, 48000]")
        if self.channels not in (1, 2):
            raise ValueError("laptop audio channels must be 1 or 2")
        if (
            not isinstance(self.command_duration_seconds, int)
            or isinstance(self.command_duration_seconds, bool)
            or not 1 <= self.command_duration_seconds <= 30
        ):
            raise ValueError("laptop command duration must be in [1, 30] seconds")
        if (
            not isinstance(self.wake_word, str)
            or len(self.wake_word.split()) != 1
            or not self.wake_word.isalnum()
        ):
            raise ValueError("laptop wake word must be one alphanumeric word")
        if (
            not isinstance(self.wake_window_seconds, int)
            or isinstance(self.wake_window_seconds, bool)
            or not 1 <= self.wake_window_seconds <= 10
        ):
            raise ValueError("laptop wake window must be in [1, 10] seconds")


@dataclass(frozen=True)
class LocalSpecialistRuntimeConfig:
    """Explicit laptop-hosted specialists used by the physical runtime."""

    simulation_learning_config: Path
    simulation_checkpoint: Path
    component_ids: frozenset[str]
    max_gripper_width_m: float
    fixed_link_reach_tolerance_m: float
    per_arm_payload_kg: float
    placement_tolerance_m: float

    def __post_init__(self) -> None:
        supported = {
            "baseten-grasp-pose",
            "baseten-waypoint-policy",
            "baseten-forward-dynamics",
            "baseten-rigid-world",
            "baseten-grasp-contact-world",
            "baseten-task-reward",
            "baseten-outcome-verifier",
        }
        if self.component_ids != supported:
            missing = sorted(supported - self.component_ids)
            extra = sorted(self.component_ids - supported)
            raise ValueError(
                "Local specialist profile must declare the complete single-arm set: "
                f"missing={missing}, extra={extra}"
            )
        for name, value in (
            ("max_gripper_width_m", self.max_gripper_width_m),
            ("fixed_link_reach_tolerance_m", self.fixed_link_reach_tolerance_m),
            ("per_arm_payload_kg", self.per_arm_payload_kg),
            ("placement_tolerance_m", self.placement_tolerance_m),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Local specialist {name} must be finite and positive")


@dataclass(frozen=True)
class PhysicalRuntimeConfig:
    path: Path
    robot_model: Path
    camera_calibration: Path
    component_manifest: Path
    memory_path: Path
    journal_path: Path
    camera: CameraRuntimeConfig
    audio: AudioRuntimeConfig
    controller: ControllerRuntimeConfig
    router: RouterConfig
    remote_components: Mapping[str, RemoteEndpointConfig]
    local_specialists: LocalSpecialistRuntimeConfig | None = None


def _resolved_relative(config_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Runtime {name} must be a non-empty path")
    return (config_path.parent / value).resolve()


def load_physical_runtime_config(path: str | Path) -> PhysicalRuntimeConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Physical runtime config schema_version must be 1")
    router_raw = raw.get("router")
    if not isinstance(router_raw, dict):
        raise TypeError("Physical runtime router must be an object")
    router = RouterConfig(
        router_raw.get("id_env", ""),
        router_raw.get("environment_env", "BASETEN_ROUTER_ENVIRONMENT"),
        router_raw.get("default_environment", "development"),
    )
    camera_raw = raw.get(
        "camera",
        {
            "transport": "opencv",
            "camera_id": "co6-usb",
            "device_env": "MOIRA_CAMERA_DEVICE",
        },
    )
    if not isinstance(camera_raw, dict):
        raise TypeError("Physical runtime camera must be an object")
    camera = CameraRuntimeConfig(
        transport=camera_raw.get("transport", "opencv"),
        camera_id=camera_raw.get("camera_id", "co6-usb"),
        device_env=camera_raw.get("device_env"),
        url_env=camera_raw.get("url_env"),
        token_env=camera_raw.get("token_env"),
        rotation_degrees=camera_raw.get("rotation_degrees", 0),
    )
    audio_raw = raw.get("audio")
    if not isinstance(audio_raw, dict):
        raise TypeError("Physical runtime audio must be an object")
    audio = AudioRuntimeConfig(
        transport=audio_raw.get("transport", "ffmpeg_dshow"),
        device_env=audio_raw.get("device_env", ""),
        sample_rate_hz=audio_raw.get("sample_rate_hz", 16_000),
        channels=audio_raw.get("channels", 1),
        command_duration_seconds=audio_raw.get("command_duration_seconds", 5),
        wake_word=audio_raw.get("wake_word", DEFAULT_WAKE_WORD),
        wake_window_seconds=audio_raw.get("wake_window_seconds", 4),
    )
    controller_raw = raw.get("controller", {"transport": "local_pca9685"})
    if not isinstance(controller_raw, dict):
        raise TypeError("Physical runtime controller must be an object")
    controller = ControllerRuntimeConfig(
        controller_raw.get("transport", "local_pca9685"),
        controller_raw.get("url_env"),
        controller_raw.get("token_env"),
    )
    remote_raw = raw.get("remote_components")
    if not isinstance(remote_raw, dict) or not remote_raw:
        raise ValueError("Physical runtime requires remote_components")
    remote_components: dict[str, RemoteEndpointConfig] = {}
    for component_id, value in remote_raw.items():
        if not isinstance(component_id, str) or not component_id.strip():
            raise ValueError("Remote component IDs must be non-empty strings")
        if not isinstance(value, dict):
            raise TypeError(f"Remote component {component_id} must be an object")
        transport = value.get("transport", "baseten_deployment")
        remote_components[component_id] = RemoteEndpointConfig(
            transport=transport,
            id_env=value.get("id_env"),
            entity=value.get("entity"),
            environment_env=value.get("environment_env", "BASETEN_ENVIRONMENT"),
            default_environment=value.get("default_environment", "development"),
            model_env=value.get("model_env"),
            default_model=value.get("default_model"),
            url_env=value.get("url_env"),
            token_env=value.get("token_env"),
        )
    local_raw = raw.get("local_specialists")
    local_specialists = None
    if local_raw is not None:
        if not isinstance(local_raw, dict):
            raise TypeError("Physical runtime local_specialists must be an object")
        component_ids = local_raw.get("component_ids")
        assumptions = local_raw.get("assumptions")
        if (
            not isinstance(component_ids, list)
            or any(not isinstance(value, str) or not value.strip() for value in component_ids)
            or len(component_ids) != len(set(component_ids))
        ):
            raise ValueError("Physical runtime local specialist IDs must be unique strings")
        if not isinstance(assumptions, dict):
            raise TypeError("Physical runtime local specialist assumptions must be an object")
        local_specialists = LocalSpecialistRuntimeConfig(
            _resolved_relative(
                config_path,
                local_raw.get("simulation_learning_config"),
                "local_specialists.simulation_learning_config",
            ),
            _resolved_relative(
                config_path,
                local_raw.get("simulation_checkpoint"),
                "local_specialists.simulation_checkpoint",
            ),
            frozenset(component_ids),
            assumptions.get("max_gripper_width_m"),
            assumptions.get("fixed_link_reach_tolerance_m"),
            assumptions.get("per_arm_payload_kg"),
            assumptions.get("placement_tolerance_m"),
        )
    config = PhysicalRuntimeConfig(
        config_path,
        _resolved_relative(config_path, raw.get("robot_model"), "robot_model"),
        _resolved_relative(
            config_path,
            raw.get("camera_calibration"),
            "camera_calibration",
        ),
        _resolved_relative(config_path, raw.get("component_manifest"), "component_manifest"),
        _resolved_relative(config_path, raw.get("memory_path"), "memory_path"),
        _resolved_relative(config_path, raw.get("journal_path"), "journal_path"),
        camera,
        audio,
        controller,
        router,
        remote_components,
        local_specialists,
    )
    manifest = json.loads(config.component_manifest.read_text(encoding="utf-8"))
    values = manifest.get("components") if isinstance(manifest, dict) else None
    if not isinstance(values, list):
        raise ValueError("Component manifest must contain a components list")
    manifest_remote = {
        value.get("id")
        for value in values
        if isinstance(value, dict) and value.get("runtime") == "remote"
    }
    configured_remote = set(config.remote_components)
    if manifest_remote != configured_remote:
        missing = sorted(manifest_remote - configured_remote)
        extra = sorted(configured_remote - manifest_remote)
        raise ValueError(
            "Runtime remote component map does not match the component manifest: "
            f"missing={missing}, extra={extra}"
        )
    return config


def prepare_physical_workspace(
    config: PhysicalRuntimeConfig,
    workspace: Mapping[str, Any],
    *,
    camera_id: str,
) -> dict[str, Any]:
    """Bind trusted robot/camera calibration to task-specific workspace data."""

    if not isinstance(workspace, Mapping):
        raise TypeError("workspace must be a mapping")
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("camera_id must be non-empty")
    model = load_robot_model(config.robot_model, verify_source=True)
    calibration_raw = json.loads(config.camera_calibration.read_text(encoding="utf-8"))
    if not isinstance(calibration_raw, dict):
        raise TypeError("Camera calibration file must contain a JSON object")
    calibration = validate_camera_calibration(calibration_raw, model)
    if calibration["camera_id"] != camera_id:
        raise ValueError(
            f"Camera ID {camera_id} does not match calibration {calibration['camera_id']}"
        )
    calibration_workspace = calibration.get("workspace")
    if not isinstance(calibration_workspace, dict):
        raise TypeError("Camera calibration workspace must be an object")
    table_height = calibration_workspace.get("table_height_m")
    if (
        not isinstance(table_height, (int, float))
        or isinstance(table_height, bool)
        or not math.isfinite(table_height)
    ):
        raise ValueError("Camera calibration workspace.table_height_m must be finite")
    if model.up_axis not in ("x", "y", "z"):
        raise ValueError("Robot model needs a calibrated up axis")

    result = dict(workspace)
    for name, expected in (
        ("coordinate_frame", "robot_base"),
        ("up_axis", model.up_axis),
    ):
        if name in result and result[name] != expected:
            raise ValueError(f"Workspace {name} contradicts calibrated value {expected}")
        result[name] = expected
    # The checkerboard defines the camera's robot-base reference plane. A task
    # may deliberately use a raised support surface, which is required by the
    # current fixed-elbow reach arc. Keep the calibrated plane as the default
    # while allowing an explicit finite task surface height.
    task_table_height = result.get("table_height_m", float(table_height))
    if (
        not isinstance(task_table_height, (int, float))
        or isinstance(task_table_height, bool)
        or not math.isfinite(task_table_height)
    ):
        raise ValueError("Workspace table_height_m must be finite")
    result["table_height_m"] = float(task_table_height)
    result["calibration_reference_plane_height_m"] = float(table_height)
    perception = result.get("perception")
    if not isinstance(perception, Mapping):
        raise ValueError("Workspace needs a perception object with demo labels and dimensions")
    perception = dict(perception)
    if "camera_calibrations" in perception:
        raise ValueError("Task workspace cannot override trusted camera calibration")
    perception["camera_calibrations"] = {
        camera_id: {
            "resolution_px": calibration["resolution_px"],
            "camera_matrix": calibration["camera_matrix"],
            "distortion_coefficients": calibration["distortion_coefficients"],
            "camera_to_base_matrix": calibration["camera_to_base_matrix"],
        }
    }
    result["perception"] = perception
    if calibration_workspace.get("bounds_base_frame_m") is not None:
        result["bounds_base_frame_m"] = calibration_workspace["bounds_base_frame_m"]
    return result


class ExactContractRouter:
    """Route exact contracts locally and use the MoIRA Chain for semantic policy choice."""

    def __init__(self, registry: ComponentRegistry, semantic: RemoteComponentRouter) -> None:
        self.registry = registry
        self.semantic = semantic

    def decide(
        self,
        layer: Layer,
        capability: str,
        context: Mapping[str, Any] | None = None,
    ) -> ComponentDecision:
        return self.registry.decide(layer, capability, context)

    def select_compatible(
        self,
        layer: Layer,
        capabilities: tuple[str, ...],
        routing_text: str,
        context: Mapping[str, Any] | None = None,
    ) -> ComponentDecision:
        return self.semantic.select_compatible(layer, capabilities, routing_text, context)


class LazyPCA9685Control:
    """Open the I2C device only for an authorized execute request."""

    def __init__(self, model: RobotModel) -> None:
        model.require_motion_ready()
        self.model = model
        self._controller: BimanualControlComponent | None = None
        self._device: PCA9685PulseDevice | None = None
        self._lock = Lock()
        self._stop_latched = False

    def _load(self) -> BimanualControlComponent:
        with self._lock:
            if self._stop_latched:
                raise RuntimeError(
                    "Emergency stop latch is active; check the workspace and restart runtime"
                )
            if self._controller is None:
                drivers, device = pca9685_installed_arm_drivers(self.model)
                self._device = device
                self._controller = BimanualControlComponent.from_installed_arms(
                    self.model,
                    left=drivers.get("left"),
                    right=drivers.get("right"),
                )
            return self._controller

    def run(self, request: ControlInput) -> ControlReport:
        if not isinstance(request, ControlInput):
            raise TypeError("PCA9685 control expects ControlInput")
        if not request.execute:
            return ControlReport(request.plan.candidate.id, False, True, ())
        with self._lock:
            stop_latched = self._stop_latched
        if stop_latched:
            return ControlReport(
                request.plan.candidate.id,
                False,
                False,
                (),
                ("Emergency stop latch is active; check the workspace and restart runtime",),
            )
        return self._load().run(request)

    def emergency_stop(self) -> tuple[str, ...]:
        with self._lock:
            self._stop_latched = True
            controller = self._controller
        return controller.emergency_stop() if controller is not None else ()

    def stop(self) -> None:
        issues = self.emergency_stop()
        if issues:
            raise RuntimeError("; ".join(issues))

    def close(self) -> None:
        with self._lock:
            if self._controller is not None:
                for driver in self._controller.drivers.values():
                    driver.stop()
            close = getattr(self._device, "close", None)
            if callable(close):
                close()
            self._controller = None
            self._device = None


def _remote_factory(
    component_id: str,
    config: RemoteEndpointConfig,
    *,
    timeout_seconds: float,
) -> Callable[[], ModelComponent]:
    def create() -> ModelComponent:
        http = JsonHttpClient(timeout_seconds=timeout_seconds, attempts=1)
        if config.transport == "baseten_model_api":
            model_name = (
                os.environ.get(config.model_env, "") if config.model_env else ""
            ) or config.default_model
            model = BasetenModelAPI(model_name, http=http)
            if component_id == "baseten-vision-scene":
                return ModelAPIScenePerception(model)
            if component_id == "baseten-voice-nlp":
                return ModelAPIVoiceGrounder(model)
            raise ValueError(f"No Model API adapter is defined for {component_id}")
        if config.transport == "json_http":
            url = os.environ.get(config.url_env or "")
            if not url:
                raise RuntimeError(f"Set {config.url_env} before invoking {component_id}")
            token = os.environ.get(config.token_env or "") or None
            return JsonComponent(
                JsonEndpoint(
                    url,
                    token=token,
                    http=JsonHttpClient(
                        timeout_seconds=timeout_seconds,
                        attempts=1,
                        allow_private_http=True,
                    ),
                ),
                decode=decode_physical_response,
            )
        entity_id = os.environ.get(config.id_env or "")
        if not entity_id:
            raise RuntimeError(f"Set {config.id_env} before invoking {component_id}")
        environment = os.environ.get(config.environment_env, config.default_environment)
        return BasetenComponent(
            BasetenEndpoint(
                entity_id,
                entity=config.entity,
                environment=environment,
                http=http,
            ),
            decode=decode_physical_response,
        )

    return create


def physical_runtime_factories(
    config: PhysicalRuntimeConfig,
    model: RobotModel,
    profile: PiRuntimeProfile,
    *,
    factory_overrides: Mapping[str, Callable[[], ModelComponent]] | None = None,
) -> Mapping[str, Callable[[], ModelComponent]]:
    model.require_motion_ready()
    factories: dict[str, Callable[[], ModelComponent]] = {
        component_id: _remote_factory(
            component_id,
            endpoint,
            timeout_seconds=profile.cloud_timeout_seconds,
        )
        for component_id, endpoint in config.remote_components.items()
    }
    if config.controller.transport == "pi_http":
        from .robot_link import PiRobotControlComponent

        controller_url = os.environ.get(config.controller.url_env or "")
        controller_token = os.environ.get(config.controller.token_env or "")
        if not controller_url:
            raise RuntimeError(
                f"Set {config.controller.url_env} before starting {DISPLAY_NAME}"
            )
        if not controller_token:
            raise RuntimeError(
                f"Set {config.controller.token_env} before starting {DISPLAY_NAME}"
            )

        def controller_factory() -> ModelComponent:
            return PiRobotControlComponent(controller_url, controller_token, model)

    else:

        def controller_factory() -> ModelComponent:
            return LazyPCA9685Control(model)

    factories.update(
        {
            "sqlite-personal-memory": lambda: SQLitePersonalMemory(config.memory_path),
            "local-bimanual-ik": lambda: PlanarBimanualIK.from_robot_model(model),
            "local-trajectory-planner": lambda: BoundedTrajectoryPlanner.from_robot_model(model),
            "local-collision-checker": lambda: ConservativeCollisionChecker.from_robot_model(model),
            "local-safety-risk": lambda: HardSafetyRiskModel.from_robot_model(model),
            "local-tactile-signal": lambda: TactileSignalModel.from_robot_model(model),
            "dual-arm-hardware": controller_factory,
            "local-failure-classifier": lambda: TelemetryFailureClassifier.from_robot_model(model),
            "local-load-estimator": TelemetryLoadEstimator,
            "local-prediction-error": WorldModelErrorTracker,
            "local-load-feedback": lambda: LoadFeedbackComponent.from_robot_model(model),
        }
    )
    local = config.local_specialists
    if local is not None:
        learned_models = AcceptedSimulationModels(
            local.simulation_checkpoint,
            load_simulation_learning_config(local.simulation_learning_config),
        )
        local_factories: dict[str, Callable[[], ModelComponent]] = {
            "baseten-grasp-pose": AnalyticGraspPlanner,
            "baseten-waypoint-policy": lambda: SpecializedManipulationPolicy("waypoint"),
            "baseten-forward-dynamics": lambda: LearnedForwardDynamics(
                learned_models,
                max_gripper_width_m=local.max_gripper_width_m,
            ),
            "baseten-rigid-world": lambda: StateSpaceWorldModel(
                "rigid-dynamics",
                per_arm_payload_kg=local.per_arm_payload_kg,
            ),
            "baseten-grasp-contact-world": lambda: StateSpaceWorldModel(
                "grasp-contact",
                per_arm_payload_kg=local.per_arm_payload_kg,
            ),
            "baseten-task-reward": ModelBasedTaskReward,
            "baseten-outcome-verifier": lambda: TelemetryOutcomeVerifier(
                placement_tolerance_m=local.placement_tolerance_m
            ),
            "local-bimanual-ik": lambda: LearnedWaypointKinematics(
                learned_models,
                max_gripper_width_m=local.max_gripper_width_m,
                position_tolerance_m=local.fixed_link_reach_tolerance_m,
            ),
        }
        undeclared = set(local_factories) - (
            set(local.component_ids) | {"local-bimanual-ik"}
        )
        if undeclared:
            raise ValueError(
                "Local specialist factories contain undeclared component IDs: "
                + ", ".join(sorted(undeclared))
            )
        factories.update(local_factories)
    if factory_overrides is not None:
        unknown = set(factory_overrides) - set(factories)
        if unknown:
            raise ValueError(
                "Physical factory overrides contain unknown component IDs: "
                + ", ".join(sorted(unknown))
            )
        if any(not callable(factory) for factory in factory_overrides.values()):
            raise TypeError("Physical factory overrides must contain callables")
        factories.update(factory_overrides)
    return factories


def build_physical_session(
    config: PhysicalRuntimeConfig,
    cameras: tuple[Any, ...],
    *,
    profile: PiRuntimeProfile | None = None,
    event_sink: Callable[[PipelineEvent], None] | None = None,
    factory_overrides: Mapping[str, Callable[[], ModelComponent]] | None = None,
) -> PhysicalSession:
    load_runtime_environment()
    model = load_robot_model(config.robot_model, verify_source=True)
    model.require_motion_ready()
    profile = profile or PiRuntimeProfile.for_pi4()
    factories = physical_runtime_factories(
        config,
        model,
        profile,
        factory_overrides=factory_overrides,
    )
    registry = ComponentRegistry.from_json(
        config.component_manifest,
        factories,
        ram_budget_mb=profile.component_ram_budget_mb,
        allow_remote=True,
    )
    router_id = os.environ.get(config.router.id_env)
    if not router_id:
        raise RuntimeError(
            f"Set {config.router.id_env} in .env before starting {DISPLAY_NAME}"
        )
    router_environment = os.environ.get(
        config.router.environment_env,
        config.router.default_environment,
    )
    semantic = RemoteComponentRouter.from_baseten_chain(
        registry,
        router_id,
        environment=router_environment,
    )
    system = PhysicalAI.from_robot_model(
        registry,
        model,
        router=ExactContractRouter(registry, semantic),
        profile=profile,
        require_camera_verification=True,
        event_sink=event_sink,
    )
    return PhysicalSession(system, cameras, JsonlRunJournal(config.journal_path))


def inspect_physical_runtime(
    config: PhysicalRuntimeConfig,
    *,
    local_component_ids: Collection[str] = (),
) -> dict[str, Any]:
    load_runtime_environment()
    model = load_robot_model(config.robot_model, verify_source=True)
    configured_local_ids = (
        frozenset()
        if config.local_specialists is None
        else config.local_specialists.component_ids
    )
    local_ids = frozenset(local_component_ids) | configured_local_ids
    if any(not isinstance(component_id, str) for component_id in local_ids):
        raise TypeError("local_component_ids must contain strings")
    unknown_local_ids = local_ids - set(config.remote_components)
    if unknown_local_ids:
        raise ValueError(
            "Local physical component overrides contain unknown component IDs: "
            + ", ".join(sorted(unknown_local_ids))
        )
    endpoint_variables = {
        component_id: endpoint.required_environment_variable
        for component_id, endpoint in config.remote_components.items()
        if component_id not in local_ids
    }
    missing_endpoint_variables = sorted(
        variable
        for variable in endpoint_variables.values()
        if variable is not None and not os.environ.get(variable)
    )
    core_variables = {
        endpoint_variables[component_id]
        for component_id in SINGLE_ARM_PICK_PLACE_COMPONENTS & set(endpoint_variables)
        if endpoint_variables[component_id] is not None
    }
    missing_core_variables = sorted(
        variable for variable in core_variables if not os.environ.get(variable)
    )
    missing_optional_variables = sorted(
        set(missing_endpoint_variables) - set(missing_core_variables)
    )
    base_key_present = bool(os.environ.get("BASETEN_API_KEY"))
    router_present = bool(os.environ.get(config.router.id_env))
    opencv_present = importlib.util.find_spec("cv2") is not None
    camera_url_present = bool(config.camera.url_env and os.environ.get(config.camera.url_env))
    camera_token_present = bool(config.camera.token_env and os.environ.get(config.camera.token_env))
    camera_device_present = bool(
        config.camera.device_env and os.environ.get(config.camera.device_env)
    )
    if config.camera.transport == "lan_http":
        camera_ready = camera_url_present and camera_token_present
        camera_blockers = []
        if not camera_url_present:
            camera_blockers.append(f"{config.camera.url_env} is not set")
        if not camera_token_present:
            camera_blockers.append(f"{config.camera.token_env} is not set")
    else:
        camera_ready = opencv_present and camera_device_present
        camera_blockers = []
        if not opencv_present:
            camera_blockers.append("OpenCV USB camera dependency is not installed")
        if not camera_device_present:
            camera_blockers.append(f"{config.camera.device_env} is not set")
    controller_url_present = bool(
        config.controller.url_env and os.environ.get(config.controller.url_env)
    )
    controller_token_present = bool(
        config.controller.token_env and os.environ.get(config.controller.token_env)
    )
    if config.controller.transport == "pi_http":
        controller_ready = controller_url_present and controller_token_present
        controller_blockers = []
        if not controller_url_present:
            controller_blockers.append(f"{config.controller.url_env} is not set")
        if not controller_token_present:
            controller_blockers.append(f"{config.controller.token_env} is not set")
    else:
        controller_ready = True
        controller_blockers = []
    local_specialist_issues: list[str] = []
    if config.local_specialists is not None:
        local = config.local_specialists
        for name, path in (
            ("simulation learning config", local.simulation_learning_config),
            ("simulation checkpoint", local.simulation_checkpoint),
        ):
            if not path.is_file():
                local_specialist_issues.append(f"local specialist {name} is missing: {path}")
    blockers = [*model.readiness_issues, *local_specialist_issues]
    if not base_key_present:
        blockers.append("BASETEN_API_KEY is not set")
    if not router_present:
        blockers.append(f"{config.router.id_env} is not set")
    if missing_core_variables:
        blockers.append(
            "missing Baseten IDs for the single-arm pick/place demo: "
            + ", ".join(missing_core_variables)
        )
    blockers.extend(camera_blockers)
    blockers.extend(controller_blockers)
    return {
        "ready": not blockers,
        "ready_for_single_arm_pick_place": not blockers,
        "catalog_complete": not missing_endpoint_variables,
        "robot_model_id": model.model_id,
        "motion_ready": model.motion_ready,
        "bimanual_motion_ready": model.bimanual_motion_ready,
        "robot_readiness_issues": model.readiness_issues,
        "baseten_api_key_present": base_key_present,
        "router_id_present": router_present,
        "component_endpoints": {
            component_id: {
                "environment_variable": endpoint.required_environment_variable,
                "transport": config.remote_components[component_id].transport,
                "configured": (
                    component_id in local_ids
                    or endpoint.required_environment_variable is None
                    or bool(os.environ.get(endpoint.required_environment_variable))
                ),
                "overridden_locally": component_id in local_ids,
            }
            for component_id, endpoint in config.remote_components.items()
        },
        "missing_core_endpoint_variables": missing_core_variables,
        "missing_optional_endpoint_variables": missing_optional_variables,
        "opencv_present": opencv_present,
        "camera_transport": config.camera.transport,
        "camera_id": config.camera.camera_id,
        "camera_ready": camera_ready,
        "camera_device_environment_variable": config.camera.device_env,
        "camera_device_configured": camera_device_present,
        "camera_url_present": camera_url_present,
        "camera_token_present": camera_token_present,
        "controller_transport": config.controller.transport,
        "controller_ready": controller_ready,
        "controller_url_present": controller_url_present,
        "controller_token_present": controller_token_present,
        "local_specialist_components": sorted(configured_local_ids),
        "local_specialist_issues": local_specialist_issues,
        "memory_path": str(config.memory_path),
        "journal_path": str(config.journal_path),
        "blockers": tuple(dict.fromkeys(blockers)),
    }
