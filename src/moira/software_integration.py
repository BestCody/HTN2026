"""Non-actuating integration profile for the hackathon workflow.

This profile exercises real cloud/LAN boundaries and loads hash-bound, accepted
simulation checkpoints for fixed-elbow waypoint control and short-horizon
dynamics. Specialists without robot-specific checkpoints remain explicitly
identified by their implementation type. The profile can never authorize
physical execution.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

from dotenv import load_dotenv

from .cloud import (
    BasetenComponent,
    BasetenEndpoint,
    BasetenModelAPI,
    JsonComponent,
    JsonEndpoint,
    JsonHttpClient,
    RemoteComponentRouter,
)
from .components import ComponentRegistry, ComponentSpec, Layer
from .contracts import decode_physical_response
from .edge_components import (
    LoadFeedbackComponent,
    SimulatedArmDriver,
    SQLitePersonalMemory,
)
from .human_interaction import HumanAwarePhysicalSession, PlanProposal
from .learned_specialists import (
    AcceptedSimulationModels,
    LearnedForwardDynamics,
    LearnedWaypointKinematics,
)
from .model_api_components import ModelAPIScenePerception, ModelAPIVoiceGrounder
from .physical import (
    BimanualControlComponent,
    CameraFrame,
    ClarificationResult,
    PhysicalAI,
    PhysicalAIResult,
    PipelineEvent,
    RobotState,
)
from .pi import PiRuntimeProfile
from .production import ExactContractRouter
from .robot_config import load_robot_model
from .session import JsonlRunJournal, PhysicalSession
from .simulation_learning import load_simulation_learning_config
from .specialists import (
    AnalyticGraspPlanner,
    BoundedTrajectoryPlanner,
    ConservativeCollisionChecker,
    HardSafetyRiskModel,
    ModelBasedTaskReward,
    SpecializedManipulationPolicy,
    StateSpaceWorldModel,
    TactileSignalModel,
    TelemetryFailureClassifier,
    TelemetryLoadEstimator,
    TelemetryOutcomeVerifier,
    WorldModelErrorTracker,
)


@dataclass(frozen=True)
class SoftwareIntegrationConfig:
    path: Path
    robot_model: Path
    memory_path: Path
    journal_path: Path
    scene_image: Path
    command_audio: Path
    simulation_learning_config: Path
    simulation_checkpoint: Path
    workspace: dict[str, Any]
    profile: PiRuntimeProfile
    assumptions: dict[str, float]


@dataclass(frozen=True)
class SoftwareIntegrationRun:
    result: PhysicalAIResult
    transcript: str
    routed_components: tuple[str, ...]
    world_models: tuple[str, ...]


class ImageFileCamera:
    """Read the same encoded scene for deterministic pre-calibration validation."""

    def __init__(self, path: Path, camera_id: str = "software-camera") -> None:
        self.path = path
        self.camera_id = camera_id

    def capture(self) -> CameraFrame:
        data = self.path.read_bytes()
        if not data:
            raise ValueError(f"Software integration image is empty: {self.path}")
        return CameraFrame(self.camera_id, data, time())


def load_software_integration_config(path: str | Path) -> SoftwareIntegrationConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("software integration schema_version must be 1")
    if raw.get("validated_for_physical_use") is not False:
        raise ValueError("software integration profile must never authorize physical use")
    profile = raw.get("profile")
    assumptions = raw.get("integration_assumptions")
    workspace = raw.get("workspace")
    if not isinstance(profile, dict) or not isinstance(assumptions, dict):
        raise TypeError("software integration profile and assumptions must be objects")
    if not isinstance(workspace, dict):
        raise TypeError("software integration workspace must be an object")
    required_assumptions = {
        "per_arm_payload_kg",
        "max_gripper_width_m",
        "fixed_link_reach_tolerance_m",
        "required_clearance_m",
        "max_joint_velocity_rad_s",
        "max_gripper_velocity_m_s",
        "max_gripper_force_n",
        "max_slip_probability",
        "max_collision_probability",
        "min_grasp_stability_score",
    }
    if set(assumptions) != required_assumptions or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
        for value in assumptions.values()
    ):
        raise ValueError("software integration assumptions are incomplete or invalid")
    root = config_path.parent

    def resolved(name: str) -> Path:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"software integration {name} must be a path")
        return (root / value).resolve()

    return SoftwareIntegrationConfig(
        config_path,
        resolved("robot_model"),
        resolved("memory_path"),
        resolved("journal_path"),
        resolved("scene_image"),
        resolved("command_audio"),
        resolved("simulation_learning_config"),
        resolved("simulation_checkpoint"),
        workspace,
        PiRuntimeProfile(**profile),
        {key: float(value) for key, value in assumptions.items()},
    )


def _environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} before running software integration")
    return value


def _register(
    registry: ComponentRegistry,
    component_id: str,
    layer: Layer,
    capabilities: tuple[str, ...],
    model: str,
    component: Any,
    *,
    runtime: str = "local",
    priority: int = 100,
    concurrency: int = 1,
) -> None:
    registry.register(
        ComponentSpec(
            component_id,
            layer,
            capabilities,
            model,
            runtime=runtime,
            estimated_ram_mb=8,
            priority=priority,
            max_concurrency=concurrency,
        ),
        lambda component=component: component,
    )


def build_software_integration_session(
    config: SoftwareIntegrationConfig,
    *,
    event_sink: Callable[[PipelineEvent], None] | None = None,
    camera_source: Any | None = None,
    arm_driver: Any | None = None,
    control_component_id: str = "dual-arm-hardware",
    control_model: str = "non-actuating-integration-controller-v1",
    control_runtime: str = "hardware",
    perception_wrapper: Callable[[Any], Any] | None = None,
    voice_grounder_wrapper: Callable[[Any], Any] | None = None,
    profile_user_id: str = "demo-user",
) -> tuple[PhysicalSession, SQLitePersonalMemory, Any, AcceptedSimulationModels]:
    """Build a session that cannot move a physical motor under any request."""

    load_dotenv(config.path.parent.parent / ".env", override=False)
    load_dotenv(Path.cwd() / ".env", override=False)
    model = load_robot_model(config.robot_model, verify_source=True)
    assumptions = config.assumptions
    learned_models = AcceptedSimulationModels(
        config.simulation_checkpoint,
        load_simulation_learning_config(config.simulation_learning_config),
    )
    http = JsonHttpClient(
        timeout_seconds=config.profile.cloud_timeout_seconds,
        attempts=2,
    )
    model_name = os.environ.get("BASETEN_VISION_MODEL_API", "zai-org/GLM-5.3-Flash")
    voice_model_name = os.environ.get("BASETEN_VOICE_NLP_MODEL_API", "zai-org/GLM-5.3-Flash")
    stt = JsonComponent(
        JsonEndpoint(
            _environment("MOIRA_STT_URL"),
            token=os.environ.get("MOIRA_LAN_TOKEN"),
            http=JsonHttpClient(
                timeout_seconds=config.profile.cloud_timeout_seconds,
                attempts=1,
                allow_private_http=True,
            ),
        ),
        decode=decode_physical_response,
    )
    tts = JsonComponent(
        JsonEndpoint(
            _environment("MOIRA_TTS_URL"),
            token=os.environ.get("MOIRA_LAN_TOKEN"),
            http=JsonHttpClient(
                timeout_seconds=config.profile.cloud_timeout_seconds,
                attempts=1,
                allow_private_http=True,
            ),
        ),
        decode=decode_physical_response,
    )
    planner = BasetenComponent(
        BasetenEndpoint(
            _environment("BASETEN_PLANNER_CHAIN_ID"),
            entity="chain",
            environment=os.environ.get("BASETEN_PLANNER_ENVIRONMENT", "production"),
            http=http,
        ),
        decode=decode_physical_response,
    )
    memory = SQLitePersonalMemory(config.memory_path)
    memory.set_profile(
        profile_user_id,
        preferences={"motion_style": "slow", "delivery_side": "right"},
        accommodations=("avoid sudden motion near the user's left side",),
        workspace={"known_workspace": "hackathon demo table"},
    )
    left_driver = arm_driver or SimulatedArmDriver(
        "left", measured_mass_kg=assumptions["per_arm_payload_kg"]
    )
    registry = ComponentRegistry(
        ram_budget_mb=config.profile.component_ram_budget_mb,
        allow_remote=True,
    )
    _register(
        registry,
        "baseten-vision-scene",
        Layer.PERCEPTION,
        ("perception.scene",),
        model_name,
        (
            perception_wrapper(ModelAPIScenePerception(BasetenModelAPI(model_name, http=http)))
            if perception_wrapper is not None
            else ModelAPIScenePerception(BasetenModelAPI(model_name, http=http))
        ),
        runtime="remote",
    )
    _register(
        registry,
        "sqlite-personal-memory",
        Layer.PERSONAL,
        ("personal.recall", "personal.record"),
        "sqlite-personal-memory-v1",
        memory,
    )
    _register(
        registry,
        "baseten-grasp-pose",
        Layer.GRASP,
        ("grasp.pose_6d",),
        "integration-analytic-grasp-v1",
        AnalyticGraspPlanner(),
    )
    _register(
        registry,
        "baseten-stt",
        Layer.VOICE,
        ("voice.transcribe",),
        "faster-whisper-large-v3-turbo-rtx",
        stt,
        runtime="remote",
    )
    _register(
        registry,
        "baseten-voice-nlp",
        Layer.VOICE,
        ("voice.ground",),
        voice_model_name,
        (
            voice_grounder_wrapper(
                ModelAPIVoiceGrounder(BasetenModelAPI(voice_model_name, http=http))
            )
            if voice_grounder_wrapper is not None
            else ModelAPIVoiceGrounder(BasetenModelAPI(voice_model_name, http=http))
        ),
        runtime="remote",
    )
    _register(
        registry,
        "baseten-tts",
        Layer.VOICE,
        ("voice.synthesize",),
        "kokoro-82m-rtx",
        tts,
        runtime="remote",
    )
    _register(
        registry,
        "baseten-task-planner",
        Layer.PLANNING,
        ("planning.candidates", "planning.select"),
        "baseten-deterministic-planner-chain",
        planner,
        runtime="remote",
        concurrency=2,
    )
    for component_id, capability, policy_name in (
        ("baseten-waypoint-policy", "manipulation.skill.waypoint", "waypoint"),
        ("baseten-pour-policy", "manipulation.skill.pour", "pour"),
        ("baseten-insert-policy", "manipulation.skill.insert", "insert"),
        ("baseten-open-lid-policy", "manipulation.skill.open_lid", "open-lid"),
        ("baseten-handover-policy", "manipulation.skill.handover", "handover"),
    ):
        _register(
            registry,
            component_id,
            Layer.MANIPULATION,
            (capability,),
            f"semantic-{policy_name}-action-chunker-v1",
            SpecializedManipulationPolicy(policy_name),
            runtime="remote",
            concurrency=2,
        )
    _register(
        registry,
        "local-bimanual-ik",
        Layer.KINEMATICS,
        ("kinematics.inverse",),
        f"fixed-elbow-waypoint-policy-{learned_models.run_id}",
        LearnedWaypointKinematics(
            learned_models,
            max_gripper_width_m=assumptions["max_gripper_width_m"],
            position_tolerance_m=assumptions["fixed_link_reach_tolerance_m"],
        ),
        concurrency=2,
    )
    _register(
        registry,
        "local-trajectory-planner",
        Layer.MOTION,
        ("motion.trajectory",),
        "bounded-trajectory-integration-v1",
        BoundedTrajectoryPlanner(
            max_joint_velocity_rad_s=assumptions["max_joint_velocity_rad_s"],
            max_gripper_velocity_m_s=assumptions["max_gripper_velocity_m_s"],
            max_gripper_force_n=assumptions["max_gripper_force_n"],
        ),
        concurrency=2,
    )
    _register(
        registry,
        "local-collision-checker",
        Layer.MOTION,
        ("motion.collision_check",),
        "conservative-collision-integration-v1",
        ConservativeCollisionChecker(required_clearance_m=assumptions["required_clearance_m"]),
        concurrency=2,
    )
    _register(
        registry,
        "local-tactile-signal",
        Layer.TACTILE,
        (
            "tactile.contact",
            "tactile.slip",
            "tactile.force",
            "tactile.grasp_stability",
        ),
        "tactile-integration-v1",
        TactileSignalModel(min_stability_score=assumptions["min_grasp_stability_score"]),
    )
    for component_id, layer, capability, kind in (
        ("baseten-forward-dynamics", Layer.DYNAMICS, "dynamics.predict", "forward-dynamics"),
        ("baseten-rigid-world", Layer.WORLD, "world.rigid_dynamics", "rigid-dynamics"),
        ("baseten-grasp-contact-world", Layer.WORLD, "world.grasp_contact", "grasp-contact"),
        (
            "baseten-deformable-world",
            Layer.WORLD,
            "world.deformable_dynamics",
            "deformable-dynamics",
        ),
        ("baseten-human-motion-world", Layer.WORLD, "world.human_motion", "human-motion"),
    ):
        component = (
            LearnedForwardDynamics(
                learned_models,
                max_gripper_width_m=assumptions["max_gripper_width_m"],
            )
            if kind == "forward-dynamics"
            else StateSpaceWorldModel(
                kind,
                per_arm_payload_kg=assumptions["per_arm_payload_kg"],
            )
        )
        component_model = (
            f"fixed-elbow-dynamics-{learned_models.run_id}"
            if kind == "forward-dynamics"
            else f"state-space-{kind}-v1"
        )
        _register(
            registry,
            component_id,
            layer,
            (capability,),
            component_model,
            component,
            concurrency=2,
        )
    _register(
        registry,
        "baseten-task-reward",
        Layer.REWARD,
        ("reward.task_progress",),
        "model-based-task-reward-v1",
        ModelBasedTaskReward(),
        concurrency=2,
    )
    _register(
        registry,
        "local-safety-risk",
        Layer.SAFETY,
        ("safety.risk",),
        "hard-safety-integration-v1",
        HardSafetyRiskModel(
            per_arm_payload_kg=assumptions["per_arm_payload_kg"],
            max_slip_probability=assumptions["max_slip_probability"],
            max_collision_probability=assumptions["max_collision_probability"],
            min_grasp_stability_score=assumptions["min_grasp_stability_score"],
        ),
        concurrency=2,
    )
    _register(
        registry,
        control_component_id,
        Layer.CONTROL,
        ("control.single_arm",),
        control_model,
        BimanualControlComponent(left_driver),
        runtime=control_runtime,
    )
    _register(
        registry,
        "baseten-outcome-verifier",
        Layer.OUTCOME,
        ("outcome.verify",),
        "integration-telemetry-outcome-v1",
        TelemetryOutcomeVerifier(
            placement_tolerance_m=float(
                config.workspace["human_error_policy"]["max_pre_execution_target_drift_m"]
            )
        ),
    )
    _register(
        registry,
        "local-failure-classifier",
        Layer.FAILURE,
        ("failure.classify",),
        "integration-failure-v1",
        TelemetryFailureClassifier(max_slip_probability=assumptions["max_slip_probability"]),
    )
    _register(
        registry,
        "local-load-estimator",
        Layer.LOAD,
        ("load.estimate",),
        "integration-load-v1",
        TelemetryLoadEstimator(),
    )
    _register(
        registry,
        "local-prediction-error",
        Layer.FEEDBACK,
        ("feedback.prediction_error",),
        "integration-prediction-error-v1",
        WorldModelErrorTracker(),
    )
    _register(
        registry,
        "local-load-feedback",
        Layer.FEEDBACK,
        ("feedback.learn",),
        "integration-feedback-v1",
        LoadFeedbackComponent(single_arm_payload_kg=assumptions["per_arm_payload_kg"]),
    )
    semantic = RemoteComponentRouter.from_baseten_chain(
        registry,
        _environment("BASETEN_ROUTER_CHAIN_ID"),
        environment=os.environ.get("BASETEN_ROUTER_ENVIRONMENT", "development"),
        http=http,
    )
    system = PhysicalAI(
        registry,
        router=ExactContractRouter(registry, semantic),
        profile=config.profile,
        robot_model_id=model.model_id,
        gripper_geometry={
            "type": "parallel-jaw",
            "max_width_m": assumptions["max_gripper_width_m"],
        },
        available_arms=("left",),
        require_camera_verification=True,
        event_sink=event_sink,
    )
    session = PhysicalSession(
        system,
        (camera_source or ImageFileCamera(config.scene_image),),
        JsonlRunJournal(config.journal_path),
    )
    return session, memory, left_driver, learned_models


def commanded_home_state(
    models: AcceptedSimulationModels,
    *,
    max_gripper_width_m: float,
) -> RobotState:
    """Create the explicit motor-locked state used before physical telemetry exists."""

    commands = (
        models.config.base_commands[0],
        models.config.shoulder_commands[0],
        models.config.gripper_commands[1],
    )
    qpos = models.commands_to_qpos(commands)
    return RobotState(
        joint_positions={"left": (qpos[0], qpos[1])},
        gripper_widths_m={"left": max_gripper_width_m},
        observed_at=time(),
        source="commanded_home",
    )


def run_software_integration(
    config: SoftwareIntegrationConfig,
    *,
    event_sink: Callable[[PipelineEvent], None] | None = None,
) -> SoftwareIntegrationRun:
    if not config.scene_image.is_file() or not config.command_audio.is_file():
        raise FileNotFoundError("software integration scene image and command audio are required")
    session, _memory, driver, learned_models = build_software_integration_session(
        config,
        event_sink=event_sink,
    )
    try:
        result = session.run(
            user_id="demo-user",
            audio=config.command_audio.read_bytes(),
            workspace=config.workspace,
            robot_state=commanded_home_state(
                learned_models,
                max_gripper_width_m=config.assumptions["max_gripper_width_m"],
            ),
            execute=False,
            speak=True,
        )
        if isinstance(result, ClarificationResult):
            raise RuntimeError(f"software integration unexpectedly requested: {result.question}")
        if not isinstance(result, PhysicalAIResult):
            raise TypeError("software integration returned an unknown result")
        return validate_software_integration_result(config, result, driver)
    finally:
        session.close()


def validate_software_integration_result(
    config: SoftwareIntegrationConfig,
    result: PhysicalAIResult,
    driver: SimulatedArmDriver,
    *,
    required_transcript_terms: tuple[str, ...] = ("red block", "blue tray"),
    generic_pick_place: bool = False,
) -> SoftwareIntegrationRun:
    """Apply the same strict evidence checks to normal and recovery demos."""

    transcript = result.intent.transcript
    object_ids = {item.id for item in result.world.objects}
    routed = tuple(item.component_id for item in result.routing)
    world_models = tuple(dict.fromkeys(item.model_kind for item in result.world_predictions))
    grounded_roles = result.intent.object_roles
    if generic_pick_place:
        visual_grounding = bool(object_ids)
        grounded_intent = any(
            object_id in object_ids and role == "manipulated"
            for object_id, role in grounded_roles.items()
        ) and all(object_id in object_ids for object_id in grounded_roles)
    else:
        visual_grounding = {"red-block-1", "blue-tray-1"} <= object_ids
        grounded_intent = grounded_roles.get("red-block-1") == "manipulated" and (
            grounded_roles.get("blue-tray-1") == "destination"
        )
    checks = {
        "speech transcript": all(
            term.casefold() in transcript.casefold() for term in required_transcript_terms
        )
        and bool(transcript.strip()),
        "visual grounding": visual_grounding,
        "grounded intent": grounded_intent,
        "personal memory": bool(result.personal.accommodations),
        "router chain": any(item.router == "remote_semantic" for item in result.routing),
        "planner chain": routed.count("baseten-task-planner") == 2,
        "candidate count": len(result.candidates) == config.profile.max_candidate_plans,
        "parallel predictions": len(result.simulations) == len(result.candidates),
        "prediction horizon": all(
            item.horizon_seconds == config.profile.simulation_horizon_seconds
            for item in result.simulations
        ),
        "safe selection": result.plan.simulation.safe,
        "motor lock": not result.control.executed and not driver.steps,
        "planned outcome": result.outcome is not None
        and result.outcome.status == "planned"
        and result.world_after is None,
        "prediction logging": result.prediction_error is not None
        and result.prediction_error.sample_count == 0,
        "spoken response": bool(result.response_audio)
        and result.response_audio.startswith(b"RIFF"),
        "journal": config.journal_path.is_file(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("software integration checks failed: " + ", ".join(failed))
    return SoftwareIntegrationRun(result, transcript, routed, world_models)


def run_human_error_integration(
    config: SoftwareIntegrationConfig,
    *,
    event_sink: Callable[[PipelineEvent], None] | None = None,
    correction_sink: Callable[[str, str], None] | None = None,
) -> SoftwareIntegrationRun:
    """Demonstrate spoken ambiguity, clarification, correction, and safe replanning."""

    if not config.scene_image.is_file():
        raise FileNotFoundError("software integration scene image is required")
    session, _memory, driver, learned_models = build_software_integration_session(
        config,
        event_sink=event_sink,
    )
    try:
        conversation = HumanAwarePhysicalSession(
            session,
            user_id="demo-user",
            workspace=config.workspace,
        )
        robot_state = commanded_home_state(
            learned_models,
            max_gripper_width_m=config.assumptions["max_gripper_width_m"],
        )
        ambiguous_text = "Move the red block."
        ambiguous_audio = session.system.synthesize_speech(ambiguous_text, "demo-user")
        first = conversation.plan(
            audio=ambiguous_audio,
            robot_state=robot_state,
            speak=True,
        )
        if not isinstance(first, ClarificationResult):
            raise RuntimeError("human-error demo did not request a missing destination")
        correction_text = "Put it in the blue tray."
        if correction_sink is not None:
            correction_sink(first.question, correction_text)
        correction_audio = session.system.synthesize_speech(correction_text, "demo-user")
        second = conversation.plan(
            audio=correction_audio,
            robot_state=robot_state,
            speak=True,
        )
        if not isinstance(second, PlanProposal):
            raise RuntimeError("human-error correction did not produce a confirmable plan")
        return validate_software_integration_result(
            config,
            second.result,
            driver,
            required_transcript_terms=("blue tray",),
        )
    finally:
        session.close()
