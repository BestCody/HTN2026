"""Offline layered physical-AI demo for Raspberry Pi orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time

from .components import ComponentRegistry, ComponentSpec, Layer
from .edge_components import (
    LayeredRulePlanner,
    LoadFeedbackComponent,
    SceneGroundedVoiceNLP,
    SimulatedArmDriver,
    SQLitePersonalMemory,
    StructuredFramePerception,
    Utf8SpeechFixture,
    Utf8SpeechSynthesisFixture,
)
from .physical import (
    BimanualControlComponent,
    CameraFrame,
    PhysicalAI,
    RobotState,
    TactileSample,
    TaskRequest,
)
from .pi import PiRuntimeProfile
from .robot_config import load_bundled_current_arm_model
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


@dataclass(frozen=True)
class EdgeDemoResult:
    selected_plan: str
    routed_components: tuple[str, ...]
    executed: bool
    success: bool
    learned_facts: tuple[str, ...]
    response: str
    world_models: tuple[str, ...]


def _register(
    registry: ComponentRegistry,
    spec: ComponentSpec,
    component,
) -> None:
    registry.register(spec, lambda: component)


def run_edge_demo(memory_path: str | Path | None = None) -> EdgeDemoResult:
    temporary = TemporaryDirectory() if memory_path is None else None
    try:
        arm_model = load_bundled_current_arm_model()
        path = Path(memory_path) if memory_path is not None else Path(temporary.name) / "memory.db"
        memory = SQLitePersonalMemory(path)
        memory.set_profile(
            "sam",
            preferences={"delivery_side": "right"},
            accommodations=("left arm is broken",),
            workspace={"counter_height_m": 0.9},
        )
        planner = LayeredRulePlanner()
        left, right = (
            SimulatedArmDriver("left", measured_mass_kg=0.08),
            SimulatedArmDriver("right", measured_mass_kg=0.08),
        )
        registry = ComponentRegistry(ram_budget_mb=256, allow_remote=True)
        specs = [
            (
                ComponentSpec(
                    "camera-object-detector",
                    Layer.PERCEPTION,
                    ("perception.scene",),
                    "structured-frame-fixture",
                    estimated_ram_mb=16,
                ),
                StructuredFramePerception(),
            ),
            (
                ComponentSpec(
                    "personal-sqlite",
                    Layer.PERSONAL,
                    ("personal.recall", "personal.record"),
                    "sqlite-personal-memory-v1",
                    estimated_ram_mb=4,
                ),
                memory,
            ),
            (
                ComponentSpec(
                    "analytic-grasp-planner",
                    Layer.GRASP,
                    ("grasp.pose_6d",),
                    "analytic-6d-grasp-v1",
                    estimated_ram_mb=2,
                ),
                AnalyticGraspPlanner(),
            ),
            (
                ComponentSpec(
                    "scene-grounded-voice-nlp",
                    Layer.VOICE,
                    ("voice.ground",),
                    "scene-grounded-voice-nlp-fixture",
                    estimated_ram_mb=2,
                ),
                SceneGroundedVoiceNLP(),
            ),
            (
                ComponentSpec(
                    "speech-fixture",
                    Layer.VOICE,
                    ("voice.transcribe",),
                    "utf8-stt-fixture",
                    estimated_ram_mb=1,
                ),
                Utf8SpeechFixture(),
            ),
            (
                ComponentSpec(
                    "tts-fixture",
                    Layer.VOICE,
                    ("voice.synthesize",),
                    "utf8-tts-fixture",
                    estimated_ram_mb=1,
                ),
                Utf8SpeechSynthesisFixture(),
            ),
            (
                ComponentSpec(
                    "layered-planner",
                    Layer.PLANNING,
                    ("planning.candidates", "planning.select"),
                    "layered-rule-planner-v1",
                    estimated_ram_mb=8,
                ),
                planner,
            ),
            *(
                (
                    ComponentSpec(
                        f"policy-{name}",
                        Layer.MANIPULATION,
                        (capability,),
                        f"{name}-policy-fixture-v1",
                        estimated_ram_mb=2,
                        max_concurrency=2,
                    ),
                    SpecializedManipulationPolicy(name),
                )
                for name, capability in (
                    ("waypoint", "manipulation.skill.waypoint"),
                    ("bimanual", "manipulation.bimanual"),
                    ("pour", "manipulation.skill.pour"),
                    ("insert", "manipulation.skill.insert"),
                    ("open-lid", "manipulation.skill.open_lid"),
                    ("handover", "manipulation.skill.handover"),
                )
            ),
            (
                ComponentSpec(
                    "local-analytic-ik",
                    Layer.KINEMATICS,
                    ("kinematics.inverse",),
                    "offline-generic-planar-3dof-fixture-v1",
                    estimated_ram_mb=2,
                    max_concurrency=2,
                ),
                PlanarBimanualIK(include_wrist_joint=False),
            ),
            (
                ComponentSpec(
                    "local-trajectory-planner",
                    Layer.MOTION,
                    ("motion.trajectory",),
                    "bounded-trajectory-v1",
                    estimated_ram_mb=2,
                    max_concurrency=2,
                ),
                BoundedTrajectoryPlanner(),
            ),
            (
                ComponentSpec(
                    "local-collision-checker",
                    Layer.MOTION,
                    ("motion.collision_check",),
                    "conservative-collision-v1",
                    estimated_ram_mb=2,
                    max_concurrency=2,
                ),
                ConservativeCollisionChecker(),
            ),
            (
                ComponentSpec(
                    "local-tactile-model",
                    Layer.TACTILE,
                    (
                        "tactile.contact",
                        "tactile.slip",
                        "tactile.force",
                        "tactile.grasp_stability",
                    ),
                    "shared-tactile-encoder-v1",
                    estimated_ram_mb=4,
                ),
                TactileSignalModel(),
            ),
            *(
                (
                    ComponentSpec(
                        component_id,
                        layer,
                        (capability,),
                        f"state-space-{kind}-v1",
                        estimated_ram_mb=3,
                        max_concurrency=2,
                    ),
                    StateSpaceWorldModel(kind, per_arm_payload_kg=arm_model.payload_limit_kg),
                )
                for component_id, layer, capability, kind in (
                    ("forward-dynamics", Layer.DYNAMICS, "dynamics.predict", "forward-dynamics"),
                    ("rigid-world", Layer.WORLD, "world.rigid_dynamics", "rigid-dynamics"),
                    ("contact-world", Layer.WORLD, "world.grasp_contact", "grasp-contact"),
                    (
                        "bimanual-world",
                        Layer.WORLD,
                        "world.bimanual_coordination",
                        "bimanual-coordination",
                    ),
                    (
                        "deformable-world",
                        Layer.WORLD,
                        "world.deformable_dynamics",
                        "deformable-dynamics",
                    ),
                    ("human-world", Layer.WORLD, "world.human_motion", "human-motion"),
                )
            ),
            (
                ComponentSpec(
                    "task-reward",
                    Layer.REWARD,
                    ("reward.task_progress",),
                    "model-based-task-reward-v1",
                    estimated_ram_mb=2,
                    max_concurrency=2,
                ),
                ModelBasedTaskReward(),
            ),
            (
                ComponentSpec(
                    "local-safety-risk",
                    Layer.SAFETY,
                    ("safety.risk",),
                    "hard-safety-risk-v1",
                    estimated_ram_mb=2,
                    max_concurrency=2,
                ),
                HardSafetyRiskModel(per_arm_payload_kg=arm_model.payload_limit_kg),
            ),
            (
                ComponentSpec(
                    "dual-arm-controller",
                    Layer.CONTROL,
                    ("control.single_arm", "control.bimanual"),
                    "dual-arm-hardware-driver-v1",
                    runtime="hardware",
                    estimated_ram_mb=2,
                ),
                BimanualControlComponent(left, right),
            ),
            (
                ComponentSpec(
                    "outcome-verifier",
                    Layer.OUTCOME,
                    ("outcome.verify",),
                    "telemetry-outcome-v1",
                    estimated_ram_mb=2,
                ),
                TelemetryOutcomeVerifier(),
            ),
            (
                ComponentSpec(
                    "failure-classifier",
                    Layer.FAILURE,
                    ("failure.classify",),
                    "telemetry-failure-v1",
                    estimated_ram_mb=2,
                ),
                TelemetryFailureClassifier(),
            ),
            (
                ComponentSpec(
                    "load-estimator",
                    Layer.LOAD,
                    ("load.estimate",),
                    "telemetry-load-v1",
                    estimated_ram_mb=2,
                ),
                TelemetryLoadEstimator(),
            ),
            (
                ComponentSpec(
                    "load-feedback",
                    Layer.FEEDBACK,
                    ("feedback.learn",),
                    "online-load-feedback-v1",
                    estimated_ram_mb=2,
                ),
                LoadFeedbackComponent(single_arm_payload_kg=arm_model.payload_limit_kg),
            ),
            (
                ComponentSpec(
                    "prediction-error-tracker",
                    Layer.FEEDBACK,
                    ("feedback.prediction_error",),
                    "world-model-error-tracker-v1",
                    estimated_ram_mb=2,
                ),
                WorldModelErrorTracker(),
            ),
        ]
        for spec, component in specs:
            _register(registry, spec, component)
        profile = PiRuntimeProfile(component_ram_budget_mb=256)
        system = PhysicalAI(
            registry,
            profile=profile,
            robot_model_id="offline-generic-dual-arm-fixture",
            gripper_geometry={"type": "parallel-jaw", "max_width_m": 0.08},
        )
        frame = CameraFrame(
            "csi0",
            {
                "objects": [
                    {
                        "id": "mug-1",
                        "label": "mug",
                        "confidence": 0.99,
                        "position_m": (0.4, 0.1, 0.9),
                        "estimated_mass_kg": 0.08,
                        "attributes": {"width_m": 0.07, "height_m": 0.10},
                    }
                ]
            },
            time(),
        )
        result = system.run(
            TaskRequest(
                "sam",
                (frame,),
                instruction="Bring me the mug",
                workspace={"room": "kitchen", "minimum_clearance_m": 0.10},
                tactile_samples=(
                    TactileSample("left", time(), 0.45, 0.04),
                    TactileSample("right", time(), 0.47, 0.04),
                ),
                friction_coefficient=0.60,
                robot_state=RobotState(
                    {
                        "left": (0.05, -0.10, 0.20),
                        "right": (-0.05, -0.10, 0.20),
                    },
                    {"left": 0.08, "right": 0.08},
                    time(),
                ),
                execute=True,
                speak=True,
            )
        )
        return EdgeDemoResult(
            result.plan.candidate.id,
            tuple(decision.component_id for decision in result.routing),
            result.control.executed,
            result.control.success,
            result.feedback.learned_facts,
            result.response_text,
            tuple(dict.fromkeys(item.model_kind for item in result.world_predictions)),
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
