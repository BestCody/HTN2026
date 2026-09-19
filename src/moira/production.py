"""Configuration-driven Raspberry Pi and Baseten physical runtime."""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from dotenv import load_dotenv

from .cloud import BasetenComponent, BasetenEndpoint, RemoteComponentRouter
from .components import ComponentDecision, ComponentRegistry, Layer, ModelComponent
from .contracts import decode_physical_response
from .edge_components import LoadFeedbackComponent, SQLitePersonalMemory
from .pca9685 import PCA9685PulseDevice, pca9685_installed_arm_drivers
from .physical import BimanualControlComponent, ControlInput, ControlReport, PhysicalAI
from .pi import PiRuntimeProfile
from .robot_config import RobotModel, load_robot_model
from .session import JsonlRunJournal, PhysicalSession
from .specialists import (
    BoundedTrajectoryPlanner,
    ConservativeCollisionChecker,
    HardSafetyRiskModel,
    PlanarBimanualIK,
    TactileSignalModel,
    TelemetryFailureClassifier,
    TelemetryLoadEstimator,
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
    id_env: str
    entity: Literal["model", "chain"]
    environment_env: str = "BASETEN_ENVIRONMENT"
    default_environment: str = "development"

    def __post_init__(self) -> None:
        for value, name in (
            (self.id_env, "endpoint id_env"),
            (self.environment_env, "endpoint environment_env"),
            (self.default_environment, "endpoint default_environment"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.entity not in ("model", "chain"):
            raise ValueError("endpoint entity must be model or chain")


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
class PhysicalRuntimeConfig:
    path: Path
    robot_model: Path
    component_manifest: Path
    memory_path: Path
    journal_path: Path
    router: RouterConfig
    remote_components: Mapping[str, RemoteEndpointConfig]


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
    remote_raw = raw.get("remote_components")
    if not isinstance(remote_raw, dict) or not remote_raw:
        raise ValueError("Physical runtime requires remote_components")
    remote_components: dict[str, RemoteEndpointConfig] = {}
    for component_id, value in remote_raw.items():
        if not isinstance(component_id, str) or not component_id.strip():
            raise ValueError("Remote component IDs must be non-empty strings")
        if not isinstance(value, dict):
            raise TypeError(f"Remote component {component_id} must be an object")
        remote_components[component_id] = RemoteEndpointConfig(
            value.get("id_env", ""),
            value.get("entity", "model"),
            value.get("environment_env", "BASETEN_ENVIRONMENT"),
            value.get("default_environment", "development"),
        )
    config = PhysicalRuntimeConfig(
        config_path,
        _resolved_relative(config_path, raw.get("robot_model"), "robot_model"),
        _resolved_relative(
            config_path, raw.get("component_manifest"), "component_manifest"
        ),
        _resolved_relative(config_path, raw.get("memory_path"), "memory_path"),
        _resolved_relative(config_path, raw.get("journal_path"), "journal_path"),
        router,
        remote_components,
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

    def _load(self) -> BimanualControlComponent:
        with self._lock:
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
        return self._load().run(request)

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
) -> Callable[[], ModelComponent]:
    def create() -> ModelComponent:
        entity_id = os.environ.get(config.id_env)
        if not entity_id:
            raise RuntimeError(
                f"Set {config.id_env} in .env before invoking {component_id}"
            )
        environment = os.environ.get(config.environment_env, config.default_environment)
        return BasetenComponent(
            BasetenEndpoint(
                entity_id,
                entity=config.entity,
                environment=environment,
            ),
            decode=decode_physical_response,
        )

    return create


def physical_runtime_factories(
    config: PhysicalRuntimeConfig,
    model: RobotModel,
) -> Mapping[str, Callable[[], ModelComponent]]:
    model.require_motion_ready()
    factories: dict[str, Callable[[], ModelComponent]] = {
        component_id: _remote_factory(component_id, endpoint)
        for component_id, endpoint in config.remote_components.items()
    }
    factories.update(
        {
            "sqlite-personal-memory": lambda: SQLitePersonalMemory(config.memory_path),
            "local-bimanual-ik": lambda: PlanarBimanualIK.from_robot_model(model),
            "local-trajectory-planner": lambda: BoundedTrajectoryPlanner.from_robot_model(model),
            "local-collision-checker": lambda: ConservativeCollisionChecker.from_robot_model(
                model
            ),
            "local-safety-risk": lambda: HardSafetyRiskModel.from_robot_model(model),
            "local-tactile-signal": lambda: TactileSignalModel.from_robot_model(model),
            "dual-arm-hardware": lambda: LazyPCA9685Control(model),
            "local-failure-classifier": lambda: TelemetryFailureClassifier.from_robot_model(
                model
            ),
            "local-load-estimator": TelemetryLoadEstimator,
            "local-prediction-error": WorldModelErrorTracker,
            "local-load-feedback": lambda: LoadFeedbackComponent.from_robot_model(model),
        }
    )
    return factories


def build_physical_session(
    config: PhysicalRuntimeConfig,
    cameras: tuple[Any, ...],
    *,
    profile: PiRuntimeProfile | None = None,
) -> PhysicalSession:
    load_dotenv(override=False)
    model = load_robot_model(config.robot_model, verify_source=True)
    model.require_motion_ready()
    profile = profile or PiRuntimeProfile.for_pi4()
    factories = physical_runtime_factories(config, model)
    registry = ComponentRegistry.from_json(
        config.component_manifest,
        factories,
        ram_budget_mb=profile.component_ram_budget_mb,
        allow_remote=True,
    )
    router_id = os.environ.get(config.router.id_env)
    if not router_id:
        raise RuntimeError(f"Set {config.router.id_env} in .env before starting MoIRA")
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
    )
    return PhysicalSession(system, cameras, JsonlRunJournal(config.journal_path))


def inspect_physical_runtime(config: PhysicalRuntimeConfig) -> dict[str, Any]:
    load_dotenv(override=False)
    model = load_robot_model(config.robot_model, verify_source=True)
    endpoint_variables = {
        component_id: endpoint.id_env
        for component_id, endpoint in config.remote_components.items()
    }
    missing_endpoint_variables = sorted(
        variable for variable in endpoint_variables.values() if not os.environ.get(variable)
    )
    core_variables = {
        endpoint_variables[component_id]
        for component_id in SINGLE_ARM_PICK_PLACE_COMPONENTS
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
    blockers = [*model.readiness_issues]
    if not base_key_present:
        blockers.append("BASETEN_API_KEY is not set")
    if not router_present:
        blockers.append(f"{config.router.id_env} is not set")
    if missing_core_variables:
        blockers.append(
            "missing Baseten IDs for the single-arm pick/place demo: "
            + ", ".join(missing_core_variables)
        )
    if not opencv_present:
        blockers.append("OpenCV USB camera dependency is not installed")
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
                "environment_variable": variable,
                "configured": bool(os.environ.get(variable)),
            }
            for component_id, variable in endpoint_variables.items()
        },
        "missing_core_endpoint_variables": missing_core_variables,
        "missing_optional_endpoint_variables": missing_optional_variables,
        "opencv_present": opencv_present,
        "memory_path": str(config.memory_path),
        "journal_path": str(config.journal_path),
        "blockers": tuple(dict.fromkeys(blockers)),
    }
