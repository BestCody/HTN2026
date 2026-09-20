"""Layered physical-AI orchestration with cloud models and local control."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, fields, is_dataclass
from threading import Event, Lock
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .robot_config import RobotModel

from .components import ComponentDecision, ComponentRegistry, ComponentRouter, Layer
from .pi import PiRuntimeProfile

MANIPULATION_POLICY_CAPABILITIES = (
    "manipulation.skill.waypoint",
    "manipulation.bimanual",
    "manipulation.skill.pour",
    "manipulation.skill.insert",
    "manipulation.skill.open_lid",
    "manipulation.skill.handover",
)

_STOP_WORDS = frozenset(("stop", "cancel", "abort", "freeze", "halt", "pause", "wait"))
_STOP_PHRASES = frozenset(
    ("do not move", "dont move", "hold position", "hold on", "never mind", "nevermind")
)


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def is_emergency_stop_command(text: str) -> bool:
    """Recognize a small local stop vocabulary without consulting a cloud model."""

    _nonempty(text, "Stop command text")
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    words = set(normalized.split())
    return normalized in _STOP_PHRASES or bool(words & _STOP_WORDS)


def _wire_equivalent(value: Any) -> Any:
    """Canonicalize typed values after a JSON request/response round trip."""

    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (item.name, _wire_equivalent(getattr(value, item.name))) for item in fields(value)
        )
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _wire_equivalent(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_wire_equivalent(item) for item in value)
    return value


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    data: Any
    captured_at: float
    media_type: str = "image/jpeg"

    def __post_init__(self) -> None:
        _nonempty(self.camera_id, "camera_id")
        _nonempty(self.media_type, "media_type")
        if self.data is None:
            raise ValueError("Camera frame data is required")
        if not isinstance(self.captured_at, (int, float)) or not math.isfinite(self.captured_at):
            raise ValueError("captured_at must be finite")


class CameraSource(Protocol):
    def capture(self) -> CameraFrame: ...


@dataclass(frozen=True)
class DetectedObject:
    id: str
    label: str
    confidence: float
    position_m: tuple[float, float, float]
    estimated_mass_kg: float | None = None
    attributes: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.id, "Detected object id")
        _nonempty(self.label, "Detected object label")
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("Object confidence must be in [0, 1]")
        if len(self.position_m) != 3 or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in self.position_m
        ):
            raise ValueError("Object position must contain three finite coordinates")
        if self.estimated_mass_kg is not None and (
            not isinstance(self.estimated_mass_kg, (int, float))
            or not math.isfinite(self.estimated_mass_kg)
            or self.estimated_mass_kg <= 0
        ):
            raise ValueError("Object mass must be finite and positive")
        if self.attributes is not None and not isinstance(self.attributes, Mapping):
            raise TypeError("Object attributes must be a mapping or null")


@dataclass(frozen=True)
class WorldState:
    objects: tuple[DetectedObject, ...]
    workspace: Mapping[str, Any]
    hazards: tuple[str, ...] = ()
    observed_at: float = 0.0
    geometry: Mapping[str, Any] = field(default_factory=dict)
    coordinate_frame: str = "robot_base"
    up_axis: str = "z"

    def __post_init__(self) -> None:
        if any(not isinstance(item, DetectedObject) for item in self.objects):
            raise TypeError("World objects must be DetectedObject instances")
        ids = [item.id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("World object IDs must be unique")
        if not isinstance(self.workspace, Mapping):
            raise TypeError("World workspace must be a mapping")
        if not isinstance(self.geometry, Mapping):
            raise TypeError("World geometry must be a mapping")
        _nonempty(self.coordinate_frame, "World coordinate frame")
        if self.up_axis not in ("x", "y", "z"):
            raise ValueError("World up_axis must be x, y, or z")
        if not isinstance(self.observed_at, (int, float)) or not math.isfinite(self.observed_at):
            raise ValueError("World observed_at must be finite")
        if not isinstance(self.hazards, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.hazards
        ):
            raise ValueError("World hazards must be strings")


class SceneChangedError(RuntimeError):
    """The observed task scene changed after planning and before motor authority."""


class ConfirmedPlanChangedError(RuntimeError):
    """Fresh planning no longer matches the intent and plan the user confirmed."""


@dataclass(frozen=True)
class SceneRevalidationReport:
    target_drift_m: Mapping[str, float]
    maximum_allowed_drift_m: float


def validate_pre_execution_scene(
    before: WorldState,
    after: WorldState,
    target_ids: tuple[str, ...],
    maximum_drift_m: float,
) -> SceneRevalidationReport:
    """Reject missing, moved, or newly hazardous scenes before motor execution."""

    if not isinstance(before, WorldState) or not isinstance(after, WorldState):
        raise TypeError("Scene revalidation requires two WorldState values")
    if before.coordinate_frame != after.coordinate_frame or before.up_axis != after.up_axis:
        raise SceneChangedError("Scene coordinate frame changed after planning")
    if (
        not isinstance(target_ids, tuple)
        or len(set(target_ids)) != len(target_ids)
        or any(not isinstance(target_id, str) or not target_id.strip() for target_id in target_ids)
    ):
        raise ValueError("Scene revalidation requires unique target IDs")
    if (
        not isinstance(maximum_drift_m, (int, float))
        or isinstance(maximum_drift_m, bool)
        or not math.isfinite(maximum_drift_m)
        or maximum_drift_m <= 0
    ):
        raise ValueError("maximum_drift_m must be finite and positive")
    before_by_id = {item.id: item for item in before.objects}
    after_by_id = {item.id: item for item in after.objects}
    missing_targets = sorted(set(target_ids) - set(after_by_id))
    drift_by_target = {
        target_id: math.dist(
            before_by_id[target_id].position_m,
            after_by_id[target_id].position_m,
        )
        for target_id in set(target_ids) & set(before_by_id) & set(after_by_id)
    }
    moved_targets = {
        target_id: drift
        for target_id, drift in drift_by_target.items()
        if drift > maximum_drift_m
    }
    new_hazards = tuple(sorted(set(after.hazards) - set(before.hazards)))
    if missing_targets or moved_targets or new_hazards:
        reasons = []
        if missing_targets:
            reasons.append("missing targets: " + ", ".join(missing_targets))
        if moved_targets:
            reasons.append(
                "moved targets: "
                + ", ".join(
                    f"{target_id}={drift:.3f}m"
                    for target_id, drift in sorted(moved_targets.items())
                )
            )
        if new_hazards:
            reasons.append("new hazards: " + ", ".join(new_hazards))
        raise SceneChangedError(
            "Scene changed after planning; execution blocked: " + "; ".join(reasons)
        )
    return SceneRevalidationReport(drift_by_target, float(maximum_drift_m))


@dataclass(frozen=True)
class PersonalContext:
    user_id: str
    preferences: Mapping[str, Any]
    accommodations: tuple[str, ...]
    recent_comments: tuple[str, ...]
    workspace: Mapping[str, Any]
    learned_object_masses: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.user_id, "user_id")
        if not isinstance(self.preferences, Mapping) or not isinstance(self.workspace, Mapping):
            raise TypeError("Personal preferences and workspace must be mappings")
        if not isinstance(self.accommodations, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.accommodations
        ):
            raise ValueError("Accommodations must be non-empty strings")
        if not isinstance(self.recent_comments, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.recent_comments
        ):
            raise ValueError("Recent comments must be non-empty strings")
        if any(
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(mass, (int, float))
            or not math.isfinite(mass)
            or mass <= 0
            for label, mass in self.learned_object_masses.items()
        ):
            raise ValueError("Learned object masses must be positive finite values")


@dataclass(frozen=True)
class PlanStep:
    id: str
    action: str
    arms: tuple[str, ...]
    duration_seconds: float
    target_object_id: str | None = None
    parameters: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.id, "Plan step id")
        _nonempty(self.action, "Plan step action")
        if (
            not self.arms
            or len(set(self.arms)) != len(self.arms)
            or any(arm not in ("left", "right") for arm in self.arms)
        ):
            raise ValueError("Plan step arms must contain left, right, or both")
        if (
            not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise ValueError("Plan step duration must be finite and positive")
        if self.target_object_id is not None:
            _nonempty(self.target_object_id, "Plan step target object id")
        if self.parameters is not None and not isinstance(self.parameters, Mapping):
            raise TypeError("Plan step parameters must be a mapping or null")


@dataclass(frozen=True)
class CandidatePlan:
    id: str
    steps: tuple[PlanStep, ...]
    rationale: str

    def __post_init__(self) -> None:
        _nonempty(self.id, "Candidate plan id")
        _nonempty(self.rationale, "Candidate rationale")
        if not self.steps or any(not isinstance(step, PlanStep) for step in self.steps):
            raise ValueError("Candidate plans need typed steps")
        if len({step.id for step in self.steps}) != len(self.steps):
            raise ValueError("Candidate plan step IDs must be unique")


def policy_routing_text(transcript: str, candidate: CandidatePlan) -> str:
    """Build the semantic routing query from grounded, planner-produced data."""
    _nonempty(transcript, "Routing transcript")
    if not isinstance(candidate, CandidatePlan):
        raise TypeError("candidate must be a CandidatePlan")
    steps = "; ".join(
        (
            f"action={step.action}, arms={'+'.join(step.arms)}, "
            f"target={step.target_object_id or 'workspace'}"
        )
        for step in candidate.steps
    )
    return (
        f"User goal: {transcript}\n"
        f"Candidate rationale: {candidate.rationale}\n"
        f"Candidate steps: {steps}"
    )


@dataclass(frozen=True)
class SimulationOutcome:
    plan_id: str
    score: float
    safe: bool
    horizon_seconds: float
    risks: tuple[str, ...] = ()
    predicted: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "Simulation plan id")
        if not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ValueError("Simulation score must be finite")
        if not 0 <= self.score <= 1:
            raise ValueError("Simulation score must be in [0, 1]")
        if not isinstance(self.safe, bool):
            raise TypeError("Simulation safe must be boolean")
        if not 2 <= self.horizon_seconds <= 3:
            raise ValueError("Simulation outcome horizon must be between 2 and 3 seconds")


@dataclass(frozen=True)
class FinalPlan:
    candidate: CandidatePlan
    simulation: SimulationOutcome
    summary: str

    def __post_init__(self) -> None:
        if self.candidate.id != self.simulation.plan_id:
            raise ValueError("Final plan and simulation IDs must match")
        if not self.simulation.safe:
            raise ValueError("Final plan must have a safe simulation outcome")
        _nonempty(self.summary, "Final plan summary")


@dataclass(frozen=True)
class ArmTelemetry:
    arm: str
    step_id: str
    success: bool
    measured_mass_kg: float | None = None
    issue: str | None = None
    normal_force_n: float | None = None
    slip_probability: float | None = None

    def __post_init__(self) -> None:
        if self.arm not in ("left", "right"):
            raise ValueError("Telemetry arm must be left or right")
        _nonempty(self.step_id, "Telemetry step id")
        if not isinstance(self.success, bool):
            raise TypeError("Telemetry success must be boolean")
        if self.measured_mass_kg is not None and (
            not isinstance(self.measured_mass_kg, (int, float))
            or not math.isfinite(self.measured_mass_kg)
            or self.measured_mass_kg <= 0
        ):
            raise ValueError("Measured mass must be finite and positive")
        if self.normal_force_n is not None and (
            not isinstance(self.normal_force_n, (int, float))
            or not math.isfinite(self.normal_force_n)
            or self.normal_force_n < 0
        ):
            raise ValueError("Telemetry force must be finite and nonnegative")
        if self.slip_probability is not None and (
            not isinstance(self.slip_probability, (int, float))
            or not math.isfinite(self.slip_probability)
            or not 0 <= self.slip_probability <= 1
        ):
            raise ValueError("Telemetry slip probability must be in [0, 1]")


@dataclass(frozen=True)
class ControlReport:
    plan_id: str
    executed: bool
    success: bool
    telemetry: tuple[ArmTelemetry, ...]
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "Control plan id")
        if not isinstance(self.executed, bool) or not isinstance(self.success, bool):
            raise TypeError("Control status must be boolean")
        if any(not isinstance(value, ArmTelemetry) for value in self.telemetry):
            raise TypeError("Control telemetry must contain ArmTelemetry")
        if any(not isinstance(value, str) or not value.strip() for value in self.issues):
            raise ValueError("Control issues must be non-empty strings")
        if self.success and self.issues:
            raise ValueError("A successful control report cannot contain issues")
        if not self.executed and self.telemetry:
            raise ValueError("An unexecuted control report cannot contain telemetry")
        if self.success and self.executed and not self.telemetry:
            raise ValueError("Successful physical execution requires telemetry")


@dataclass(frozen=True)
class FeedbackReport:
    learned_facts: tuple[str, ...]
    next_time_adjustments: tuple[str, ...]
    object_masses_kg: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strings = (*self.learned_facts, *self.next_time_adjustments)
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise ValueError("Feedback entries must be non-empty strings")
        if any(
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(mass, (int, float))
            or not math.isfinite(mass)
            or mass <= 0
            for label, mass in self.object_masses_kg.items()
        ):
            raise ValueError("Feedback object masses must be positive finite values")


@dataclass(frozen=True)
class DialogueTurn:
    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError("Dialogue role must be user or assistant")
        _nonempty(self.text, "Dialogue text")


@dataclass(frozen=True)
class GroundedIntent:
    transcript: str
    action: str
    target_object_ids: tuple[str, ...]
    object_roles: Mapping[str, str]
    constraints: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarification_question: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.transcript, "Intent transcript")
        _nonempty(self.action, "Intent action")
        if not isinstance(self.target_object_ids, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.target_object_ids
        ):
            raise ValueError("Intent target_object_ids must be a tuple of IDs")
        if len(set(self.target_object_ids)) != len(self.target_object_ids):
            raise ValueError("Intent target_object_ids must be unique")
        allowed_roles = {
            "manipulated",
            "destination",
            "tool",
            "recipient",
            "support",
            "context",
        }
        if (
            not isinstance(self.object_roles, Mapping)
            or set(self.object_roles) != set(self.target_object_ids)
            or any(role not in allowed_roles for role in self.object_roles.values())
        ):
            raise ValueError(
                "Intent object_roles must assign every target a supported physical role"
            )
        if not isinstance(self.constraints, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.constraints
        ):
            raise ValueError("Intent constraints must be a tuple of non-empty strings")
        if not isinstance(self.needs_clarification, bool):
            raise TypeError("needs_clarification must be boolean")
        if self.needs_clarification:
            _nonempty(self.clarification_question, "Clarification question")
        elif self.clarification_question is not None:
            raise ValueError("A resolved intent cannot contain a clarification question")

    @property
    def manipulated_object_ids(self) -> tuple[str, ...]:
        return tuple(
            object_id
            for object_id in self.target_object_ids
            if self.object_roles[object_id] in ("manipulated", "tool")
        )

    @property
    def destination_object_ids(self) -> tuple[str, ...]:
        return tuple(
            object_id
            for object_id in self.target_object_ids
            if self.object_roles[object_id] in ("destination", "recipient", "support")
        )


def _finite_tuple(value: tuple[float, ...], length: int, name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != length
        or any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{name} must contain {length} finite numbers")


@dataclass(frozen=True)
class GraspPose:
    """One collision-scored six-degree-of-freedom gripper pose."""

    id: str
    target_object_id: str
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    width_m: float
    score: float
    collision_probability: float

    def __post_init__(self) -> None:
        _nonempty(self.id, "Grasp pose id")
        _nonempty(self.target_object_id, "Grasp target object id")
        _finite_tuple(self.position_m, 3, "Grasp position")
        _finite_tuple(self.orientation_xyzw, 4, "Grasp orientation")
        norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("Grasp orientation quaternion must be normalized")
        if not isinstance(self.width_m, (int, float)) or not 0 < self.width_m <= 0.5:
            raise ValueError("Grasp width must be in (0, 0.5] metres")
        for name, value in (
            ("Grasp score", self.score),
            ("Grasp collision probability", self.collision_probability),
        ):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class GraspPlanningInput:
    world: WorldState
    target_object_ids: tuple[str, ...]
    gripper_geometry: Mapping[str, Any]
    excluded_regions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        known = {item.id for item in self.world.objects}
        if not self.target_object_ids or any(item not in known for item in self.target_object_ids):
            raise ValueError("Grasp targets must be present in the world state")
        if not isinstance(self.gripper_geometry, Mapping):
            raise TypeError("Gripper geometry must be a mapping")
        max_width = self.gripper_geometry.get("max_width_m")
        if (
            not isinstance(max_width, (int, float))
            or isinstance(max_width, bool)
            or not math.isfinite(max_width)
            or max_width <= 0
        ):
            raise ValueError("Gripper geometry requires a finite positive max_width_m")


@dataclass(frozen=True)
class GraspPlan:
    target_object_id: str
    grasps: tuple[GraspPose, ...]

    def __post_init__(self) -> None:
        _nonempty(self.target_object_id, "Grasp-plan target")
        if not self.grasps or any(
            not isinstance(item, GraspPose) or item.target_object_id != self.target_object_id
            for item in self.grasps
        ):
            raise ValueError("Grasp plans need poses for exactly one target")
        if len({item.id for item in self.grasps}) != len(self.grasps):
            raise ValueError("Grasp pose IDs must be unique")


@dataclass(frozen=True)
class ActionChunk:
    """A bounded, policy-generated action segment; never direct motor authority."""

    id: str
    step_id: str
    skill: str
    arms: tuple[str, ...]
    duration_seconds: float
    target_object_id: str | None
    target_pose: tuple[float, float, float, float, float, float, float]
    gripper_width_m: float
    force_limit_n: float

    def __post_init__(self) -> None:
        _nonempty(self.id, "Action chunk id")
        _nonempty(self.step_id, "Action chunk step id")
        _nonempty(self.skill, "Action chunk skill")
        if self.target_object_id is not None:
            _nonempty(self.target_object_id, "Action chunk target object id")
        if not self.arms or any(arm not in ("left", "right") for arm in self.arms):
            raise ValueError("Action chunk arms must contain left, right, or both")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("Action chunk arms must be unique")
        if (
            not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise ValueError("Action chunk duration must be finite and positive")
        _finite_tuple(self.target_pose, 7, "Action target pose")
        quaternion = self.target_pose[3:]
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("Action target quaternion must be normalized")
        if (
            not isinstance(self.gripper_width_m, (int, float))
            or not 0 <= self.gripper_width_m <= 0.5
        ):
            raise ValueError("Action gripper width must be in [0, 0.5] metres")
        if (
            not isinstance(self.force_limit_n, (int, float))
            or not math.isfinite(self.force_limit_n)
            or self.force_limit_n <= 0
        ):
            raise ValueError("Action force limit must be finite and positive")


@dataclass(frozen=True)
class PolicyInput:
    candidate: CandidatePlan
    world: WorldState
    personal: PersonalContext
    grasps: tuple[GraspPlan, ...]
    frames: tuple[CameraFrame, ...] = ()
    robot_state: RobotState | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate, CandidatePlan)
            or not isinstance(self.world, WorldState)
            or not isinstance(self.personal, PersonalContext)
        ):
            raise TypeError("Policy input requires a candidate, world, and personal context")
        if any(not isinstance(item, GraspPlan) for item in self.grasps):
            raise TypeError("Policy grasps must contain GraspPlan objects")
        if any(not isinstance(item, CameraFrame) for item in self.frames):
            raise TypeError("Policy frames must contain CameraFrame objects")
        if self.robot_state is not None and not isinstance(self.robot_state, RobotState):
            raise TypeError("Policy robot_state must be a RobotState or null")


@dataclass(frozen=True)
class PolicyPlan:
    plan_id: str
    policy: str
    chunks: tuple[ActionChunk, ...]

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "Policy plan id")
        _nonempty(self.policy, "Policy name")
        if not self.chunks or any(not isinstance(item, ActionChunk) for item in self.chunks):
            raise ValueError("A policy plan needs typed action chunks")
        if len({item.id for item in self.chunks}) != len(self.chunks):
            raise ValueError("Action chunk IDs must be unique within a policy plan")


def validate_policy_for_candidate(candidate: CandidatePlan, policy: PolicyPlan) -> None:
    """Reject semantic drift between a task plan and a model-generated motor policy."""

    if not isinstance(candidate, CandidatePlan) or not isinstance(policy, PolicyPlan):
        raise TypeError("Policy validation requires a CandidatePlan and PolicyPlan")
    if policy.plan_id != candidate.id:
        raise ValueError("Manipulation policy returned the wrong plan ID")
    steps = {step.id: step for step in candidate.steps}
    chunk_step_ids = [chunk.step_id for chunk in policy.chunks]
    if set(chunk_step_ids) != set(steps):
        raise ValueError("Manipulation policy does not cover every selected plan step")
    for chunk in policy.chunks:
        step = steps[chunk.step_id]
        if chunk.arms != step.arms:
            raise ValueError(f"Policy chunk {chunk.id} changes the commanded arms for {step.id}")
        if chunk.target_object_id != step.target_object_id:
            raise ValueError(f"Policy chunk {chunk.id} changes the target object for {step.id}")
    for step in candidate.steps:
        policy_duration = sum(
            chunk.duration_seconds for chunk in policy.chunks if chunk.step_id == step.id
        )
        if not math.isclose(
            policy_duration,
            step.duration_seconds,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Policy chunks change the total duration for {step.id}")


@dataclass(frozen=True)
class KinematicsInput:
    policy: PolicyPlan
    world: WorldState
    robot_state: RobotState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, PolicyPlan) or not isinstance(self.world, WorldState):
            raise TypeError("Kinematics requires a typed policy and world state")
        if self.robot_state is not None and not isinstance(self.robot_state, RobotState):
            raise TypeError("Kinematics robot_state must be a RobotState or null")


@dataclass(frozen=True)
class KinematicsSolution:
    plan_id: str
    joint_targets: Mapping[str, Mapping[str, tuple[float, ...]]]
    feasible: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "Kinematics plan id")
        if not isinstance(self.joint_targets, Mapping):
            raise TypeError("Kinematics joint targets must be a mapping")
        if not isinstance(self.feasible, bool):
            raise TypeError("Kinematics feasibility must be boolean")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, str) or not reason.strip() for reason in self.reasons
        ):
            raise ValueError("Kinematics reasons must be non-empty strings")
        for chunk_id, targets in self.joint_targets.items():
            _nonempty(chunk_id, "Kinematics chunk id")
            if not isinstance(targets, Mapping):
                raise TypeError("Per-chunk joint targets must be mappings")
            for arm, joints in targets.items():
                if arm not in ("left", "right"):
                    raise ValueError("Kinematics target arm must be left or right")
                if (
                    not isinstance(joints, tuple)
                    or not joints
                    or any(
                        not isinstance(value, (int, float)) or not math.isfinite(value)
                        for value in joints
                    )
                ):
                    raise ValueError("Joint targets must be non-empty finite tuples")


@dataclass(frozen=True)
class TrajectoryPoint:
    time_s: float
    joint_positions: Mapping[str, tuple[float, ...]]
    chunk_id: str | None = None
    gripper_widths_m: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.time_s, (int, float))
            or not math.isfinite(self.time_s)
            or self.time_s < 0
        ):
            raise ValueError("Trajectory time must be finite and nonnegative")
        if not self.joint_positions or any(
            arm not in ("left", "right") for arm in self.joint_positions
        ):
            raise ValueError("Trajectory points need left and/or right joint positions")
        for joints in self.joint_positions.values():
            if (
                not isinstance(joints, tuple)
                or not joints
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in joints
                )
            ):
                raise ValueError("Trajectory joint positions must be non-empty finite tuples")
        if self.chunk_id is not None:
            _nonempty(self.chunk_id, "Trajectory chunk id")
        if any(
            arm not in ("left", "right")
            or not isinstance(width, (int, float))
            or not math.isfinite(width)
            or not 0 <= width <= 0.5
            for arm, width in self.gripper_widths_m.items()
        ):
            raise ValueError("Trajectory gripper widths must be in [0, 0.5] metres")


@dataclass(frozen=True)
class TrajectoryPlanningInput:
    policy: PolicyPlan
    kinematics: KinematicsSolution
    robot_state: RobotState | None = None

    def __post_init__(self) -> None:
        if self.policy.plan_id != self.kinematics.plan_id:
            raise ValueError("Trajectory policy and kinematics IDs must match")
        if not self.kinematics.feasible:
            detail = "; ".join(self.kinematics.reasons) or "inverse kinematics failed"
            raise ValueError(f"Cannot plan a trajectory for infeasible kinematics: {detail}")
        chunks = {chunk.id: chunk for chunk in self.policy.chunks}
        if set(self.kinematics.joint_targets) != set(chunks):
            raise ValueError("Kinematics targets must cover exactly the policy chunks")
        arm_joint_counts: dict[str, int] = {}
        for chunk_id, targets in self.kinematics.joint_targets.items():
            if set(targets) != set(chunks[chunk_id].arms):
                raise ValueError(f"Kinematics targets use the wrong arms for chunk {chunk_id}")
            for arm, joints in targets.items():
                count = len(joints)
                if arm in arm_joint_counts and arm_joint_counts[arm] != count:
                    raise ValueError(f"Kinematics changed the joint count for {arm}")
                arm_joint_counts[arm] = count
        if self.robot_state is not None and not isinstance(self.robot_state, RobotState):
            raise TypeError("Trajectory robot_state must be a RobotState or null")
        if self.robot_state is not None:
            missing_arms = set(arm_joint_counts) - set(self.robot_state.joint_positions)
            if missing_arms:
                raise ValueError(
                    "Robot state is missing commanded arms: " + ", ".join(sorted(missing_arms))
                )
            missing_grippers = set(arm_joint_counts) - set(self.robot_state.gripper_widths_m)
            if missing_grippers:
                raise ValueError(
                    "Robot state is missing commanded gripper widths: "
                    + ", ".join(sorted(missing_grippers))
                )
            for arm, joints in self.robot_state.joint_positions.items():
                if arm in arm_joint_counts and len(joints) != arm_joint_counts[arm]:
                    raise ValueError(f"Robot state joint count does not match kinematics for {arm}")


@dataclass(frozen=True)
class MotionTrajectory:
    plan_id: str
    points: tuple[TrajectoryPoint, ...]
    duration_seconds: float

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "Motion trajectory plan id")
        if not self.points or any(not isinstance(item, TrajectoryPoint) for item in self.points):
            raise ValueError("Motion trajectory needs typed points")
        times = [item.time_s for item in self.points]
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("Trajectory times must be strictly increasing")
        if (
            not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
            or times[-1] > self.duration_seconds + 1e-6
        ):
            raise ValueError("Trajectory duration must contain every point")
        labels = [item.chunk_id for item in self.points]
        if any(label is not None for label in labels) and any(label is None for label in labels):
            raise ValueError("Trajectory chunk labels must be present on every point or none")
        seen: set[str] = set()
        previous: str | None = None
        for label in labels:
            if label is not None and label != previous:
                if label in seen:
                    raise ValueError("Trajectory points for each chunk must be contiguous")
                seen.add(label)
            previous = label

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(point.chunk_id for point in self.points if point.chunk_id is not None)
        )

    def for_chunk(self, chunk_id: str) -> MotionTrajectory:
        """Return one chunk's time-local trajectory for a low-level driver."""

        _nonempty(chunk_id, "Trajectory chunk id")
        indexed = [
            (index, point)
            for index, point in enumerate(self.points)
            if point.chunk_id == chunk_id
        ]
        if not indexed:
            raise ValueError(f"Trajectory has no points for action chunk {chunk_id}")
        start_index = indexed[0][0]
        offset = self.points[start_index - 1].time_s if start_index else 0.0
        points = tuple(
            TrajectoryPoint(
                round(point.time_s - offset, 6),
                point.joint_positions,
                point.chunk_id,
                point.gripper_widths_m,
            )
            for _, point in indexed
        )
        return MotionTrajectory(self.plan_id, points, points[-1].time_s)


@dataclass(frozen=True)
class CollisionCheckInput:
    trajectory: MotionTrajectory
    world: WorldState

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, MotionTrajectory) or not isinstance(
            self.world, WorldState
        ):
            raise TypeError("Collision checks require a typed trajectory and world state")


@dataclass(frozen=True)
class CollisionReport:
    plan_id: str
    safe: bool
    minimum_clearance_m: float
    collisions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "Collision report plan id")
        if not isinstance(self.safe, bool):
            raise TypeError("Collision status must be boolean")
        if (
            not isinstance(self.minimum_clearance_m, (int, float))
            or not math.isfinite(self.minimum_clearance_m)
            or self.minimum_clearance_m < 0
        ):
            raise ValueError("Minimum clearance must be finite and nonnegative")
        if not isinstance(self.collisions, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.collisions
        ):
            raise ValueError("Collision entries must be non-empty strings")
        if self.safe and self.collisions:
            raise ValueError("A safe collision report cannot list collisions")


@dataclass(frozen=True)
class TactileSample:
    arm: str
    captured_at: float
    normal_force_n: float
    shear_force_n: float
    contact: bool = True

    def __post_init__(self) -> None:
        if self.arm not in ("left", "right"):
            raise ValueError("Tactile sample arm must be left or right")
        for name, value in (
            ("captured_at", self.captured_at),
            ("normal_force_n", self.normal_force_n),
            ("shear_force_n", self.shear_force_n),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Tactile {name} must be finite and nonnegative")
        if not isinstance(self.contact, bool):
            raise TypeError("Tactile contact must be boolean")


@dataclass(frozen=True)
class TactileInput:
    samples: tuple[TactileSample, ...]
    friction_coefficient: float | None = None
    capability: str = "slip"

    def __post_init__(self) -> None:
        if any(not isinstance(item, TactileSample) for item in self.samples):
            raise TypeError("Tactile input must contain typed samples")
        if self.friction_coefficient is not None and (
            not isinstance(self.friction_coefficient, (int, float))
            or isinstance(self.friction_coefficient, bool)
            or not math.isfinite(self.friction_coefficient)
            or self.friction_coefficient <= 0
        ):
            raise ValueError("Friction coefficient must be finite and positive when present")
        if self.samples and self.friction_coefficient is None:
            raise ValueError("Tactile samples require a measured friction coefficient")
        if self.capability not in ("contact", "slip", "force", "stability"):
            raise ValueError("Unsupported tactile capability")


@dataclass(frozen=True)
class ContactEstimate:
    contact: bool
    per_arm: Mapping[str, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.contact, bool) or any(
            arm not in ("left", "right") or not isinstance(value, bool)
            for arm, value in self.per_arm.items()
        ):
            raise TypeError("Contact estimates must contain per-arm booleans")


@dataclass(frozen=True)
class SlipEstimate:
    probability: float
    per_arm: Mapping[str, float]

    def __post_init__(self) -> None:
        values = (self.probability, *self.per_arm.values())
        if any(arm not in ("left", "right") for arm in self.per_arm) or any(
            not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in values
        ):
            raise ValueError("Slip probabilities must be in [0, 1]")


@dataclass(frozen=True)
class ForceEstimate:
    total_normal_force_n: float
    per_arm_n: Mapping[str, float]

    def __post_init__(self) -> None:
        values = (self.total_normal_force_n, *self.per_arm_n.values())
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in values
        ):
            raise ValueError("Force estimates must be finite and nonnegative")


@dataclass(frozen=True)
class GraspStability:
    stable: bool
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.stable, bool):
            raise TypeError("Grasp stability must be boolean")
        if not isinstance(self.score, (int, float)) or not 0 <= self.score <= 1:
            raise ValueError("Grasp stability score must be in [0, 1]")


@dataclass(frozen=True)
class PredictedState:
    time_s: float
    object_poses: Mapping[str, tuple[float, float, float, float, float, float, float]]
    joint_positions: Mapping[str, tuple[float, ...]]
    contact_forces_n: Mapping[str, float]
    slip_probability: float
    collision_probability: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.time_s, (int, float))
            or not math.isfinite(self.time_s)
            or self.time_s < 0
        ):
            raise ValueError("Predicted-state time must be finite and nonnegative")
        for pose in self.object_poses.values():
            _finite_tuple(pose, 7, "Predicted object pose")
        for arm, joints in self.joint_positions.items():
            if (
                arm not in ("left", "right")
                or not isinstance(joints, tuple)
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in joints
                )
            ):
                raise ValueError("Predicted joints must be finite tuples for known arms")
        if any(
            arm not in ("left", "right")
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for arm, value in self.contact_forces_n.items()
        ):
            raise ValueError("Predicted contact forces must be finite and nonnegative")
        for name, value in (
            ("slip_probability", self.slip_probability),
            ("collision_probability", self.collision_probability),
        ):
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class WorldModelInput:
    policy: PolicyPlan
    trajectory: MotionTrajectory
    world: WorldState
    personal: PersonalContext
    horizon_seconds: float
    step_seconds: float

    def __post_init__(self) -> None:
        if self.policy.plan_id != self.trajectory.plan_id:
            raise ValueError("World-model policy and trajectory IDs must match")
        if not 2 <= self.horizon_seconds <= 3:
            raise ValueError("World-model horizon must be between 2 and 3 seconds")
        if not 0 < self.step_seconds <= self.horizon_seconds:
            raise ValueError("World-model step must be within its horizon")


@dataclass(frozen=True)
class WorldModelPrediction:
    plan_id: str
    model_kind: str
    states: tuple[PredictedState, ...]
    success_probability: float
    uncertainty: float
    risks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "World-model plan id")
        _nonempty(self.model_kind, "World-model kind")
        if not self.states or any(not isinstance(item, PredictedState) for item in self.states):
            raise ValueError("World-model predictions need typed future states")
        times = [item.time_s for item in self.states]
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("Predicted-state times must be strictly increasing")
        for name, value in (
            ("success_probability", self.success_probability),
            ("uncertainty", self.uncertainty),
        ):
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"World-model {name} must be in [0, 1]")
        if any(not isinstance(value, str) or not value.strip() for value in self.risks):
            raise ValueError("World-model risks must be non-empty strings")


@dataclass(frozen=True)
class RewardInput:
    candidate: CandidatePlan
    predictions: tuple[WorldModelPrediction, ...]
    collision: CollisionReport

    def __post_init__(self) -> None:
        if not self.predictions or any(
            item.plan_id != self.candidate.id for item in self.predictions
        ):
            raise ValueError("Reward input needs predictions for its candidate")
        if self.collision.plan_id != self.candidate.id:
            raise ValueError("Reward collision report does not match its candidate")


@dataclass(frozen=True)
class RewardScore:
    plan_id: str
    score: float
    components: Mapping[str, float]

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "Reward plan id")
        if not isinstance(self.score, (int, float)) or not 0 <= self.score <= 1:
            raise ValueError("Reward score must be in [0, 1]")
        if not isinstance(self.components, Mapping) or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
            for value in self.components.values()
        ):
            raise ValueError("Reward components must be finite values in [0, 1]")


@dataclass(frozen=True)
class SafetyInput:
    candidate: CandidatePlan
    predictions: tuple[WorldModelPrediction, ...]
    collision: CollisionReport
    slip: SlipEstimate
    stability: GraspStability
    world: WorldState | None = None
    policy: PolicyPlan | None = None
    personal: PersonalContext | None = None

    def __post_init__(self) -> None:
        if not self.predictions or any(
            item.plan_id != self.candidate.id for item in self.predictions
        ):
            raise ValueError("Safety input needs predictions for its candidate")
        if self.collision.plan_id != self.candidate.id:
            raise ValueError("Safety collision report does not match its candidate")
        supplied = (self.world is not None, self.policy is not None, self.personal is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("Safety world, policy, and personal context must be supplied together")
        if self.policy is not None:
            validate_policy_for_candidate(self.candidate, self.policy)


@dataclass(frozen=True)
class SafetyAssessment:
    plan_id: str
    safe: bool
    risk_score: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "Safety plan id")
        if not isinstance(self.safe, bool):
            raise TypeError("Safety status must be boolean")
        if not isinstance(self.risk_score, (int, float)) or not 0 <= self.risk_score <= 1:
            raise ValueError("Safety risk score must be in [0, 1]")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.reasons
        ):
            raise ValueError("Safety reasons must be non-empty strings")
        if self.safe and self.reasons:
            raise ValueError("A safe assessment cannot list blocking reasons")


@dataclass(frozen=True)
class OutcomeInput:
    plan: FinalPlan
    control: ControlReport
    world_before: WorldState
    world_after: WorldState | None = None

    def __post_init__(self) -> None:
        if self.plan.candidate.id != self.control.plan_id:
            raise ValueError("Outcome plan and control IDs must match")
        if not isinstance(self.world_before, WorldState):
            raise TypeError("Outcome verification needs a typed world state")
        if self.world_after is not None and not isinstance(self.world_after, WorldState):
            raise TypeError("Outcome verification after-state must be a typed world state or null")
        if not self.control.executed and self.world_after is not None:
            raise ValueError("An unexecuted plan cannot have a post-action world state")


@dataclass(frozen=True)
class OutcomeReport:
    plan_id: str
    status: str
    confidence: float
    observations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "Outcome plan id")
        if self.status not in ("planned", "succeeded", "failed", "uncertain"):
            raise ValueError("Outcome status is unsupported")
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("Outcome confidence must be in [0, 1]")
        if not isinstance(self.observations, Mapping):
            raise TypeError("Outcome observations must be a mapping")


@dataclass(frozen=True)
class FailureClassificationInput:
    outcome: OutcomeReport
    control: ControlReport
    slip: SlipEstimate

    def __post_init__(self) -> None:
        if self.outcome.plan_id != self.control.plan_id:
            raise ValueError("Failure outcome and control IDs must match")


@dataclass(frozen=True)
class FailureReport:
    failure: str | None
    confidence: float
    recommended_adjustments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.failure is not None:
            _nonempty(self.failure, "Failure class")
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("Failure confidence must be in [0, 1]")
        if not isinstance(self.recommended_adjustments, Mapping):
            raise TypeError("Failure recommended_adjustments must be a mapping")


@dataclass(frozen=True)
class LoadEstimationInput:
    control: ControlReport
    world: WorldState
    plan: FinalPlan | None = None

    def __post_init__(self) -> None:
        if self.plan is not None and self.plan.candidate.id != self.control.plan_id:
            raise ValueError("Load-estimation plan and control IDs must match")


@dataclass(frozen=True)
class LoadEstimate:
    object_masses_kg: Mapping[str, float]
    confidence: float

    def __post_init__(self) -> None:
        if any(
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for label, value in self.object_masses_kg.items()
        ):
            raise ValueError("Load estimates must be finite and positive")
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("Load confidence must be in [0, 1]")


@dataclass(frozen=True)
class PredictionErrorInput:
    predictions: tuple[WorldModelPrediction, ...]
    outcome: OutcomeReport
    control: ControlReport

    def __post_init__(self) -> None:
        if not self.predictions:
            raise ValueError("Prediction-error feedback needs at least one world model")
        if any(item.plan_id != self.outcome.plan_id for item in self.predictions):
            raise ValueError("Prediction-error inputs must describe the observed plan")
        if self.control.plan_id != self.outcome.plan_id:
            raise ValueError("Prediction-error control and outcome IDs must match")


@dataclass(frozen=True)
class PredictionErrorReport:
    observed_success: float | None
    per_model_absolute_error: Mapping[str, float]
    mean_absolute_error: float
    sample_count: int

    def __post_init__(self) -> None:
        if self.observed_success is not None and not 0 <= self.observed_success <= 1:
            raise ValueError("Observed success must be null or in [0, 1]")
        values = (*self.per_model_absolute_error.values(), self.mean_absolute_error)
        if any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in values):
            raise ValueError("Prediction errors must be in [0, 1]")
        if not isinstance(self.sample_count, int) or isinstance(self.sample_count, bool):
            raise TypeError("Prediction-error sample_count must be an integer")
        if self.sample_count < 0 or self.sample_count != len(self.per_model_absolute_error):
            raise ValueError("Prediction-error sample_count must match the model errors")


@dataclass(frozen=True)
class RobotState:
    joint_positions: Mapping[str, tuple[float, ...]]
    gripper_widths_m: Mapping[str, float] = field(default_factory=dict)
    observed_at: float = 0.0
    source: str = "observed"

    def __post_init__(self) -> None:
        if any(arm not in ("left", "right") for arm in self.joint_positions):
            raise ValueError("Robot joint state arms must be left or right")
        for joints in self.joint_positions.values():
            if (
                not isinstance(joints, tuple)
                or not joints
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in joints
                )
            ):
                raise ValueError("Robot joint positions must be non-empty finite tuples")
        if any(
            arm not in ("left", "right")
            or not isinstance(width, (int, float))
            or not math.isfinite(width)
            or not 0 <= width <= 0.5
            for arm, width in self.gripper_widths_m.items()
        ):
            raise ValueError("Robot gripper widths must be in [0, 0.5] metres")
        if not isinstance(self.observed_at, (int, float)) or not math.isfinite(self.observed_at):
            raise ValueError("Robot-state timestamp must be finite")
        if self.source not in ("observed", "camera_estimate", "commanded_home", "commanded"):
            raise ValueError("Robot-state source is unsupported")


@dataclass(frozen=True)
class TaskRequest:
    user_id: str
    frames: tuple[CameraFrame, ...]
    instruction: str | None = None
    audio: bytes | None = None
    workspace: Mapping[str, Any] | None = None
    dialogue: tuple[DialogueTurn, ...] = ()
    tactile_samples: tuple[TactileSample, ...] = ()
    friction_coefficient: float | None = None
    robot_state: RobotState | None = None
    execute: bool = False
    execution_confirmed: bool = False
    speak: bool = True
    confirmed_intent: GroundedIntent | None = None
    confirmed_candidate: CandidatePlan | None = None

    def __post_init__(self) -> None:
        _nonempty(self.user_id, "user_id")
        if (self.instruction is None) == (self.audio is None):
            raise ValueError("Provide exactly one of instruction or audio")
        if self.instruction is not None:
            _nonempty(self.instruction, "instruction")
        if self.audio is not None and not isinstance(self.audio, bytes):
            raise TypeError("audio must be bytes")
        if not self.frames or any(not isinstance(frame, CameraFrame) for frame in self.frames):
            raise ValueError("At least one camera frame is required")
        if self.workspace is not None and not isinstance(self.workspace, Mapping):
            raise TypeError("workspace must be a mapping or null")
        if not isinstance(self.dialogue, tuple) or any(
            not isinstance(turn, DialogueTurn) for turn in self.dialogue
        ):
            raise TypeError("dialogue must contain DialogueTurn objects")
        if not isinstance(self.tactile_samples, tuple) or any(
            not isinstance(sample, TactileSample) for sample in self.tactile_samples
        ):
            raise TypeError("tactile_samples must contain TactileSample objects")
        if self.friction_coefficient is not None and (
            not isinstance(self.friction_coefficient, (int, float))
            or isinstance(self.friction_coefficient, bool)
            or not math.isfinite(self.friction_coefficient)
            or self.friction_coefficient <= 0
        ):
            raise ValueError("friction_coefficient must be finite and positive when present")
        if self.tactile_samples and self.friction_coefficient is None:
            raise ValueError("Tactile samples require a measured friction_coefficient")
        if self.robot_state is not None and not isinstance(self.robot_state, RobotState):
            raise TypeError("robot_state must be a RobotState or null")
        if (
            not isinstance(self.execute, bool)
            or not isinstance(self.execution_confirmed, bool)
            or not isinstance(self.speak, bool)
        ):
            raise TypeError("execute, execution_confirmed, and speak must be boolean")
        if self.execute and self.robot_state is None:
            raise ValueError("Physical execution requires a robot_state")
        if self.execute and self.robot_state is not None and self.robot_state.observed_at <= 0:
            raise ValueError("Physical execution requires a timestamped robot_state")
        if self.execute and not self.execution_confirmed:
            raise ValueError("Physical execution requires explicit plan confirmation")
        if (self.confirmed_intent is None) != (self.confirmed_candidate is None):
            raise ValueError("Confirmed intent and candidate must be supplied together")
        if self.confirmed_intent is not None and (
            not self.execute or not self.execution_confirmed
        ):
            raise ValueError("Plan bindings are only valid for confirmed physical execution")
        if self.confirmed_intent is not None and not isinstance(
            self.confirmed_intent, GroundedIntent
        ):
            raise TypeError("confirmed_intent must be a GroundedIntent")
        if self.confirmed_candidate is not None and not isinstance(
            self.confirmed_candidate, CandidatePlan
        ):
            raise TypeError("confirmed_candidate must be a CandidatePlan")


@dataclass(frozen=True)
class SpeechInput:
    audio: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.audio, bytes) or not self.audio:
            raise ValueError("Speech audio must be non-empty bytes")


@dataclass(frozen=True)
class VoiceGroundingInput:
    transcript: str
    world: WorldState
    personal: PersonalContext
    dialogue: tuple[DialogueTurn, ...]

    def __post_init__(self) -> None:
        _nonempty(self.transcript, "Voice transcript")
        if not isinstance(self.world, WorldState) or not isinstance(self.personal, PersonalContext):
            raise TypeError("Voice grounding needs typed world and personal context")


@dataclass(frozen=True)
class PerceptionInput:
    frames: tuple[CameraFrame, ...]
    workspace: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.frames or any(not isinstance(frame, CameraFrame) for frame in self.frames):
            raise ValueError("Perception requires at least one typed camera frame")
        if not isinstance(self.workspace, Mapping):
            raise TypeError("Perception workspace must be a mapping")


@dataclass(frozen=True)
class MemoryQuery:
    user_id: str
    instruction: str
    workspace: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.user_id, "Memory user id")
        _nonempty(self.instruction, "Memory instruction")
        if not isinstance(self.workspace, Mapping):
            raise TypeError("Memory workspace must be a mapping")


@dataclass(frozen=True)
class CandidatePlanningInput:
    intent: GroundedIntent
    world: WorldState
    personal: PersonalContext
    limit: int
    grasps: tuple[GraspPlan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.intent, GroundedIntent):
            raise TypeError("Planning requires a GroundedIntent")
        if not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit < 1:
            raise ValueError("Planning candidate limit must be positive")
        if any(not isinstance(item, GraspPlan) for item in self.grasps):
            raise TypeError("Planning grasps must contain GraspPlan objects")


@dataclass(frozen=True)
class SimulationInput:
    candidate: CandidatePlan
    world: WorldState
    personal: PersonalContext
    horizon_seconds: float
    step_seconds: float

    def __post_init__(self) -> None:
        if not 2 <= self.horizon_seconds <= 3:
            raise ValueError("Simulation horizon must be between 2 and 3 seconds")
        if not 0 < self.step_seconds <= self.horizon_seconds:
            raise ValueError("Simulation step must be positive and within the horizon")


@dataclass(frozen=True)
class PlanSelectionInput:
    intent: GroundedIntent
    world: WorldState
    personal: PersonalContext
    candidates: tuple[CandidatePlan, ...]
    simulations: tuple[SimulationOutcome, ...]

    def __post_init__(self) -> None:
        if {item.id for item in self.candidates} != {item.plan_id for item in self.simulations}:
            raise ValueError("Plan selection requires one simulation per candidate")


@dataclass(frozen=True)
class ControlInput:
    plan: FinalPlan
    world: WorldState
    execute: bool
    policy: PolicyPlan | None = None
    trajectory: MotionTrajectory | None = None
    tactile: GraspStability | None = None
    execution_guard: Callable[[PlanStep, tuple[str, ...]], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, FinalPlan) or not isinstance(self.world, WorldState):
            raise TypeError("Control requires a typed final plan and world state")
        if not isinstance(self.execute, bool):
            raise TypeError("Control execute must be boolean")
        if self.execution_guard is not None and not callable(self.execution_guard):
            raise TypeError("Control execution_guard must be callable or null")
        if (self.policy is None) != (self.trajectory is None):
            raise ValueError("Control policy and trajectory must be supplied together")
        if self.policy is not None:
            validate_policy_for_candidate(self.plan.candidate, self.policy)
        if self.trajectory is not None and self.trajectory.plan_id != self.plan.candidate.id:
            raise ValueError("Control trajectory does not match the selected plan")
        if self.policy is not None and self.trajectory is not None:
            if set(self.trajectory.chunk_ids) != {chunk.id for chunk in self.policy.chunks}:
                raise ValueError("Control trajectory must label every policy chunk exactly once")
        if self.tactile is not None and not isinstance(self.tactile, GraspStability):
            raise TypeError("Control tactile guard must be a GraspStability or null")


@dataclass(frozen=True)
class FeedbackInput:
    user_id: str
    instruction: str
    world: WorldState
    plan: FinalPlan
    control: ControlReport
    outcome: OutcomeReport | None = None
    failure: FailureReport | None = None
    load: LoadEstimate | None = None
    prediction_error: PredictionErrorReport | None = None

    def __post_init__(self) -> None:
        _nonempty(self.user_id, "Feedback user id")
        _nonempty(self.instruction, "Feedback instruction")
        if self.plan.candidate.id != self.control.plan_id:
            raise ValueError("Feedback plan and control IDs must match")
        if self.outcome is not None and self.outcome.plan_id != self.control.plan_id:
            raise ValueError("Feedback outcome and control IDs must match")


@dataclass(frozen=True)
class MemoryRecord:
    user_id: str
    instruction: str
    plan_summary: str
    feedback: FeedbackReport

    def __post_init__(self) -> None:
        _nonempty(self.user_id, "Memory user id")
        _nonempty(self.instruction, "Memory instruction")
        _nonempty(self.plan_summary, "Memory plan summary")
        if not isinstance(self.feedback, FeedbackReport):
            raise TypeError("Memory feedback must be a FeedbackReport")


@dataclass(frozen=True)
class SpeechSynthesisInput:
    text: str
    user_id: str

    def __post_init__(self) -> None:
        _nonempty(self.text, "Speech synthesis text")
        _nonempty(self.user_id, "Speech synthesis user id")


@dataclass(frozen=True)
class PhysicalAIResult:
    intent: GroundedIntent
    world: WorldState
    personal: PersonalContext
    candidates: tuple[CandidatePlan, ...]
    simulations: tuple[SimulationOutcome, ...]
    plan: FinalPlan
    control: ControlReport
    feedback: FeedbackReport
    response_text: str
    response_audio: bytes | None
    routing: tuple[ComponentDecision, ...]
    grasps: tuple[GraspPlan, ...] = ()
    policies: tuple[PolicyPlan, ...] = ()
    trajectories: tuple[MotionTrajectory, ...] = ()
    world_predictions: tuple[WorldModelPrediction, ...] = ()
    rewards: tuple[RewardScore, ...] = ()
    safety: tuple[SafetyAssessment, ...] = ()
    contact: ContactEstimate | None = None
    slip: SlipEstimate | None = None
    force: ForceEstimate | None = None
    stability: GraspStability | None = None
    outcome: OutcomeReport | None = None
    failure: FailureReport | None = None
    load: LoadEstimate | None = None
    prediction_error: PredictionErrorReport | None = None
    world_pre_execute: WorldState | None = None
    world_after: WorldState | None = None


@dataclass(frozen=True)
class ClarificationResult:
    transcript: str
    question: str
    world: WorldState
    personal: PersonalContext
    response_audio: bytes | None
    routing: tuple[ComponentDecision, ...]


@dataclass(frozen=True)
class EmergencyStopResult:
    transcript: str
    response_text: str
    response_audio: bytes | None
    stop_issues: tuple[str, ...]
    routing: tuple[ComponentDecision, ...]

    def __post_init__(self) -> None:
        _nonempty(self.transcript, "Emergency-stop transcript")
        _nonempty(self.response_text, "Emergency-stop response")
        if self.response_audio is not None and not isinstance(self.response_audio, bytes):
            raise TypeError("Emergency-stop response_audio must be bytes or null")
        if not isinstance(self.stop_issues, tuple) or any(
            not isinstance(issue, str) or not issue.strip() for issue in self.stop_issues
        ):
            raise TypeError("Emergency-stop issues must contain non-empty strings")
        if not isinstance(self.routing, tuple) or any(
            not isinstance(item, ComponentDecision) for item in self.routing
        ):
            raise TypeError("Emergency-stop routing must contain ComponentDecision values")


@dataclass(frozen=True)
class PipelineEvent:
    """One bounded observability event from the typed physical-AI workflow."""

    kind: str
    elapsed_seconds: float
    layer: Layer | None = None
    capability: str | None = None
    component_id: str | None = None
    model: str | None = None
    runtime: str | None = None
    router: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.kind, "Pipeline event kind")
        if (
            not isinstance(self.elapsed_seconds, (int, float))
            or isinstance(self.elapsed_seconds, bool)
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("Pipeline event elapsed_seconds must be finite and nonnegative")
        if self.layer is not None and not isinstance(self.layer, Layer):
            raise TypeError("Pipeline event layer must be a Layer or null")
        if not isinstance(self.details, Mapping):
            raise TypeError("Pipeline event details must be a mapping")


class PhysicalAI:
    """Coordinate specialized components while keeping motor authority local."""

    def __init__(
        self,
        registry: ComponentRegistry,
        *,
        router: ComponentRouter | None = None,
        profile: PiRuntimeProfile | None = None,
        robot_model_id: str,
        gripper_geometry: Mapping[str, Any],
        available_arms: tuple[str, ...] = ("left", "right"),
        require_camera_verification: bool = False,
        event_sink: Callable[[PipelineEvent], None] | None = None,
    ) -> None:
        if not isinstance(registry, ComponentRegistry):
            raise TypeError("registry must be a ComponentRegistry")
        self.registry = registry
        self.router = router or registry
        if not callable(getattr(self.router, "decide", None)):
            raise TypeError("router must implement decide(layer, capability, context)")
        self.profile = profile or PiRuntimeProfile.for_pi4()
        _nonempty(robot_model_id, "Robot model ID")
        max_gripper_width = gripper_geometry.get("max_width_m")
        if (
            not isinstance(max_gripper_width, (int, float))
            or isinstance(max_gripper_width, bool)
            or not math.isfinite(max_gripper_width)
            or max_gripper_width <= 0
        ):
            raise ValueError("gripper_geometry requires a finite positive max_width_m")
        self.robot_model_id = robot_model_id
        self.gripper_geometry = dict(gripper_geometry)
        if (
            not available_arms
            or len(set(available_arms)) != len(available_arms)
            or any(arm not in ("left", "right") for arm in available_arms)
        ):
            raise ValueError("available_arms must contain unique left/right control slots")
        self.available_arms = available_arms
        if not isinstance(require_camera_verification, bool):
            raise TypeError("require_camera_verification must be boolean")
        self.require_camera_verification = require_camera_verification
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or null")
        self.event_sink = event_sink

    @classmethod
    def from_robot_model(
        cls,
        registry: ComponentRegistry,
        robot_model: RobotModel,
        *,
        router: ComponentRouter | None = None,
        profile: PiRuntimeProfile | None = None,
        require_camera_verification: bool = False,
        event_sink: Callable[[PipelineEvent], None] | None = None,
    ) -> PhysicalAI:
        robot_model.require_motion_ready()
        return cls(
            registry,
            router=router,
            profile=profile,
            robot_model_id=robot_model.model_id,
            gripper_geometry={
                "type": "parallel-jaw",
                "max_width_m": robot_model.max_gripper_width_m,
            },
            available_arms=robot_model.servo_controller.installed_arms,
            require_camera_verification=require_camera_verification,
            event_sink=event_sink,
        )

    def emergency_stop(self) -> tuple[str, ...]:
        """Interrupt every currently loaded local control component."""

        return self.registry.emergency_stop()

    def transcribe_audio(self, audio: bytes, *, allow_empty: bool = False) -> str:
        """Transcribe a conversational response without starting a robot task."""

        decision = self.router.decide(
            Layer.VOICE,
            "voice.transcribe",
            {"device": "raspberry-pi-4b", "routing_text": "Transcribe user response."},
        )
        result = self.registry.invoke(decision, SpeechInput(audio))
        if not isinstance(result, str):
            raise TypeError("Speech specialist returned a non-text conversational transcript")
        if not result.strip() and not allow_empty:
            raise ValueError("Speech specialist returned an empty conversational transcript")
        return " ".join(result.strip().split())

    def synthesize_speech(self, text: str, user_id: str) -> bytes:
        """Synthesize an interaction prompt through the configured voice specialist."""

        decision = self.router.decide(
            Layer.VOICE,
            "voice.synthesize",
            {"device": "raspberry-pi-4b", "routing_text": "Speak interaction prompt."},
        )
        result = self.registry.invoke(decision, SpeechSynthesisInput(text, user_id))
        if not isinstance(result, bytes) or not result:
            raise ValueError("Voice synthesizer returned empty interaction audio")
        return result

    def run(
        self,
        request: TaskRequest,
        *,
        pre_action_capture: Callable[[], tuple[CameraFrame, ...]] | None = None,
        post_action_capture: Callable[[], tuple[CameraFrame, ...]] | None = None,
    ) -> PhysicalAIResult | ClarificationResult | EmergencyStopResult:
        if not isinstance(request, TaskRequest):
            raise TypeError("request must be a TaskRequest")
        if request.execute and self.require_camera_verification and (
            pre_action_capture is None or post_action_capture is None
        ):
            raise RuntimeError(
                "Physical execution requires a pre-action camera capture and "
                "post-action camera capture"
            )
        run_started = monotonic()
        decisions: list[ComponentDecision] = []
        decisions_lock = Lock()
        route_context = {
            "device": "raspberry-pi-4b",
            "robot_model_id": self.robot_model_id,
            "has_audio": request.audio is not None,
            "camera_count": len(request.frames),
            "execute": request.execute,
            "available_arms": self.available_arms,
            "component_ram_budget_mb": self.profile.component_ram_budget_mb,
        }

        def emit(
            kind: str,
            decision: ComponentDecision | None = None,
            details: Mapping[str, Any] | None = None,
        ) -> None:
            if self.event_sink is None:
                return
            self.event_sink(
                PipelineEvent(
                    kind,
                    monotonic() - run_started,
                    layer=decision.layer if decision is not None else None,
                    capability=decision.capability if decision is not None else None,
                    component_id=decision.component_id if decision is not None else None,
                    model=decision.model if decision is not None else None,
                    runtime=decision.runtime if decision is not None else None,
                    router=decision.router if decision is not None else None,
                    details=dict(details or {}),
                )
            )

        emit(
            "pipeline.started",
            details={
                "execute": request.execute,
                "audio": request.audio is not None,
                "camera_count": len(request.frames),
                "robot_model_id": self.robot_model_id,
            },
        )

        def invoke(
            layer: Layer,
            capability: str,
            payload: Any,
            expected: type,
            *,
            local_authority: bool = False,
            context: Mapping[str, Any] | None = None,
        ) -> Any:
            current_context = {**route_context, **dict(context or {})}
            decision = self.router.decide(layer, capability, current_context)

            return invoke_decision(
                decision,
                payload,
                expected,
                local_authority=local_authority,
            )

        def invoke_decision(
            decision: ComponentDecision,
            payload: Any,
            expected: type,
            *,
            local_authority: bool = False,
        ) -> Any:
            if not isinstance(decision, ComponentDecision):
                raise TypeError("Router must return a ComponentDecision")
            if local_authority and decision.runtime == "remote":
                raise RuntimeError(
                    f"{decision.capability} must remain on the robot-side safety boundary"
                )
            with decisions_lock:
                decisions.append(decision)
            emit(
                "route.selected",
                decision,
                {"local_authority": local_authority},
            )
            component_started = monotonic()
            emit("component.started", decision)
            try:
                result = self.registry.invoke(decision, payload)
            except Exception as exc:
                emit(
                    "component.failed",
                    decision,
                    {
                        "latency_seconds": monotonic() - component_started,
                        "error_type": type(exc).__name__,
                    },
                )
                raise
            if not isinstance(result, expected):
                raise TypeError(
                    f"Component {decision.component_id} returned {type(result).__name__}; "
                    f"expected {expected.__name__}"
                )
            emit(
                "component.completed",
                decision,
                {
                    "latency_seconds": monotonic() - component_started,
                    "output_type": type(result).__name__,
                },
            )
            return result

        if request.audio is not None:
            transcript = invoke(
                Layer.VOICE,
                "voice.transcribe",
                SpeechInput(request.audio),
                str,
                context={"routing_text": "Transcribe the spoken robot command."},
            )
        else:
            transcript = request.instruction
        _nonempty(transcript, "Transcript")
        emit("speech.transcribed", details={"text": transcript})
        if is_emergency_stop_command(transcript):
            stop_issues = self.emergency_stop()
            response_text = (
                "Emergency stop requested. Motor control is stopped and remains latched."
                if not stop_issues
                else "Emergency stop requested, but one or more stop hooks reported an error."
            )
            response_audio = None
            if request.speak:
                response_audio = invoke(
                    Layer.VOICE,
                    "voice.synthesize",
                    SpeechSynthesisInput(response_text, request.user_id),
                    bytes,
                )
                if not response_audio:
                    raise ValueError("Voice synthesizer returned empty emergency-stop audio")
            emit(
                "safety.emergency_stop",
                details={"issues": stop_issues, "response": response_text},
            )
            return EmergencyStopResult(
                transcript,
                response_text,
                response_audio,
                stop_issues,
                tuple(decisions),
            )
        # The frozen MoIRA router compares this task text with descriptions of
        # every contract-compatible specialist. The edge allow-list remains the
        # authority over which components can ever be selected.
        route_context["routing_text"] = transcript
        workspace = dict(request.workspace or {})
        world = invoke(
            Layer.PERCEPTION,
            "perception.scene",
            PerceptionInput(request.frames, workspace),
            WorldState,
        )
        emit(
            "scene.perceived",
            details={
                "objects": tuple(
                    {"id": item.id, "label": item.label, "confidence": item.confidence}
                    for item in world.objects
                ),
                "hazards": world.hazards,
            },
        )
        personal = invoke(
            Layer.PERSONAL,
            "personal.recall",
            MemoryQuery(request.user_id, transcript, workspace),
            PersonalContext,
        )
        if personal.user_id != request.user_id:
            raise ValueError("Personal-memory component returned context for the wrong user")
        emit(
            "memory.recalled",
            details={
                "preferences": dict(personal.preferences),
                "accommodations": personal.accommodations,
                "recent_comment_count": len(personal.recent_comments),
            },
        )
        intent = invoke(
            Layer.VOICE,
            "voice.ground",
            VoiceGroundingInput(transcript, world, personal, request.dialogue),
            GroundedIntent,
        )
        emit(
            "intent.grounded",
            details={
                "action": intent.action,
                "object_roles": dict(intent.object_roles),
                "constraints": intent.constraints,
                "needs_clarification": intent.needs_clarification,
            },
        )
        known_objects = {item.id for item in world.objects}
        if any(object_id not in known_objects for object_id in intent.target_object_ids):
            raise ValueError("Voice NLP grounded the task to an object absent from perception")
        if intent.needs_clarification:
            response_audio = None
            if request.speak:
                response_audio = invoke(
                    Layer.VOICE,
                    "voice.synthesize",
                    SpeechSynthesisInput(intent.clarification_question, request.user_id),
                    bytes,
                )
                if not response_audio:
                    raise ValueError("Voice synthesizer returned empty audio")
            recorded = invoke(
                Layer.PERSONAL,
                "personal.record",
                MemoryRecord(
                    request.user_id,
                    transcript,
                    f"Clarification requested: {intent.clarification_question}",
                    FeedbackReport((), ()),
                ),
                PersonalContext,
            )
            if recorded.user_id != request.user_id:
                raise ValueError("Personal-memory component recorded the wrong user")
            emit(
                "clarification.required",
                details={
                    "question": intent.clarification_question,
                    "transcript": transcript,
                },
            )
            return ClarificationResult(
                transcript,
                intent.clarification_question,
                world,
                personal,
                response_audio,
                tuple(decisions),
            )

        grasp_targets = intent.manipulated_object_ids
        grasps: tuple[GraspPlan, ...] = ()
        if grasp_targets:
            grasp_result = invoke(
                Layer.GRASP,
                "grasp.pose_6d",
                GraspPlanningInput(
                    world,
                    grasp_targets,
                    self.gripper_geometry,
                    tuple(str(value) for value in workspace.get("excluded_grasp_regions", ())),
                ),
                tuple,
            )
            if (
                len(grasp_result) != len(grasp_targets)
                or any(not isinstance(item, GraspPlan) for item in grasp_result)
                or {item.target_object_id for item in grasp_result} != set(grasp_targets)
            ):
                raise ValueError("Grasp component did not cover every grounded target")
            grasps = grasp_result

        contact = invoke(
            Layer.TACTILE,
            "tactile.contact",
            TactileInput(
                request.tactile_samples,
                request.friction_coefficient,
                capability="contact",
            ),
            ContactEstimate,
            local_authority=True,
        )
        slip = invoke(
            Layer.TACTILE,
            "tactile.slip",
            TactileInput(
                request.tactile_samples,
                request.friction_coefficient,
                capability="slip",
            ),
            SlipEstimate,
            local_authority=True,
        )
        force = invoke(
            Layer.TACTILE,
            "tactile.force",
            TactileInput(
                request.tactile_samples,
                request.friction_coefficient,
                capability="force",
            ),
            ForceEstimate,
            local_authority=True,
        )
        stability = invoke(
            Layer.TACTILE,
            "tactile.grasp_stability",
            TactileInput(
                request.tactile_samples,
                request.friction_coefficient,
                capability="stability",
            ),
            GraspStability,
            local_authority=True,
        )
        candidates = invoke(
            Layer.PLANNING,
            "planning.candidates",
            CandidatePlanningInput(
                intent,
                world,
                personal,
                self.profile.max_candidate_plans,
                grasps,
            ),
            tuple,
        )
        if (
            not candidates
            or len(candidates) > self.profile.max_candidate_plans
            or any(not isinstance(candidate, CandidatePlan) for candidate in candidates)
            or len({candidate.id for candidate in candidates}) != len(candidates)
        ):
            raise ValueError("Planner returned invalid candidate plans")
        available_arms = set(self.available_arms)
        candidates = tuple(
            candidate
            for candidate in candidates
            if all(set(step.arms) <= available_arms for step in candidate.steps)
        )
        if not candidates:
            raise RuntimeError("Planner returned no plan supported by the installed physical arms")
        emit(
            "plans.generated",
            details={
                "candidate_ids": tuple(candidate.id for candidate in candidates),
                "candidate_count": len(candidates),
            },
        )
        grounded_targets = set(intent.target_object_ids)
        manipulated_targets = set(intent.manipulated_object_ids)
        for candidate in candidates:
            candidate_targets = {
                step.target_object_id
                for step in candidate.steps
                if step.target_object_id is not None
            }
            if not candidate_targets <= known_objects:
                raise ValueError("Planner targeted an object absent from perception")
            referenced_objects = set(candidate_targets)
            for step in candidate.steps:
                parameters = step.parameters or {}
                referenced_objects.update(
                    value
                    for key, value in parameters.items()
                    if key.endswith("_object_id") and isinstance(value, str)
                )
            if not manipulated_targets <= candidate_targets:
                raise ValueError("Planner omitted a manipulated object grounded by voice NLP")
            if not grounded_targets <= referenced_objects:
                raise ValueError("Planner omitted an object role grounded by voice NLP")
            if not referenced_objects <= known_objects:
                raise ValueError("Planner referenced an object absent from perception")

        def fixture_policy_capability(candidate: CandidatePlan) -> str:
            """Dispatch the dependency-free offline fixture without an ML router."""
            actions = {step.action.lower().replace(" ", "_") for step in candidate.steps}
            for action, capability in (
                ("pour", "manipulation.skill.pour"),
                ("insert", "manipulation.skill.insert"),
                ("open_lid", "manipulation.skill.open_lid"),
                ("handover", "manipulation.skill.handover"),
            ):
                if action in actions:
                    return capability
            if any(len(step.arms) == 2 for step in candidate.steps):
                return "manipulation.bimanual"
            return "manipulation.skill.waypoint"

        def world_capabilities(candidate: CandidatePlan) -> tuple[tuple[Layer, str], ...]:
            actions = {step.action.lower().replace(" ", "_") for step in candidate.steps}
            capabilities: list[tuple[Layer, str]] = [
                (Layer.DYNAMICS, "dynamics.predict"),
                (Layer.WORLD, "world.rigid_dynamics"),
            ]
            if actions & {"grasp", "lift", "hold", "place", "handover"}:
                capabilities.append((Layer.WORLD, "world.grasp_contact"))
            if any(len(step.arms) == 2 for step in candidate.steps):
                capabilities.append((Layer.WORLD, "world.bimanual_coordination"))
            if actions & {"pour", "wipe", "fold", "open_lid"}:
                capabilities.append((Layer.WORLD, "world.deformable_dynamics"))
            if world.workspace.get("people_present"):
                capabilities.append((Layer.WORLD, "world.human_motion"))
            return tuple(capabilities)

        def evaluate_candidate(
            candidate: CandidatePlan,
        ) -> tuple[
            PolicyPlan,
            MotionTrajectory | None,
            tuple[WorldModelPrediction, ...],
            RewardScore,
            SafetyAssessment,
            SimulationOutcome,
        ]:
            policy_input = PolicyInput(
                candidate,
                world,
                personal,
                grasps,
                request.frames,
                request.robot_state,
            )
            routing_text = policy_routing_text(transcript, candidate)
            semantic_select = getattr(self.router, "select_compatible", None)
            if callable(semantic_select):
                emit(
                    "semantic_router.started",
                    details={
                        "candidate_id": candidate.id,
                        "compatible_capabilities": MANIPULATION_POLICY_CAPABILITIES,
                    },
                )
                router_started = monotonic()
                decision = semantic_select(
                    Layer.MANIPULATION,
                    MANIPULATION_POLICY_CAPABILITIES,
                    routing_text,
                    {**route_context, "plan_id": candidate.id},
                )
                emit(
                    "semantic_router.completed",
                    decision,
                    {
                        "candidate_id": candidate.id,
                        "latency_seconds": monotonic() - router_started,
                    },
                )
                policy = invoke_decision(decision, policy_input, PolicyPlan)
            else:
                policy = invoke(
                    Layer.MANIPULATION,
                    fixture_policy_capability(candidate),
                    policy_input,
                    PolicyPlan,
                    context={"plan_id": candidate.id, "routing_text": routing_text},
                )
            validate_policy_for_candidate(candidate, policy)
            kinematics = invoke(
                Layer.KINEMATICS,
                "kinematics.inverse",
                KinematicsInput(policy, world, request.robot_state),
                KinematicsSolution,
                local_authority=True,
                context={"plan_id": candidate.id},
            )
            if kinematics.plan_id != candidate.id:
                raise ValueError("Kinematics component returned the wrong plan ID")
            if not kinematics.feasible:
                reasons = kinematics.reasons or ("inverse kinematics found no feasible solution",)
                reward = RewardScore(candidate.id, 0.0, {"kinematics": 0.0})
                safety = SafetyAssessment(candidate.id, False, 1.0, reasons)
                simulation = SimulationOutcome(
                    candidate.id,
                    0.0,
                    False,
                    self.profile.simulation_horizon_seconds,
                    reasons,
                    {"kinematics_feasible": False},
                )
                return policy, None, (), reward, safety, simulation
            trajectory = invoke(
                Layer.MOTION,
                "motion.trajectory",
                TrajectoryPlanningInput(policy, kinematics, request.robot_state),
                MotionTrajectory,
                local_authority=True,
                context={"plan_id": candidate.id},
            )
            if trajectory.plan_id != candidate.id:
                raise ValueError("Motion planner returned the wrong plan ID")
            collision = invoke(
                Layer.MOTION,
                "motion.collision_check",
                CollisionCheckInput(trajectory, world),
                CollisionReport,
                local_authority=True,
                context={"plan_id": candidate.id},
            )
            if collision.plan_id != candidate.id:
                raise ValueError("Collision checker returned the wrong plan ID")
            world_input = WorldModelInput(
                policy,
                trajectory,
                world,
                personal,
                self.profile.simulation_horizon_seconds,
                self.profile.simulation_step_seconds,
            )

            def predict(route: tuple[Layer, str]) -> WorldModelPrediction:
                expected_model_kind = {
                    "dynamics.predict": "forward-dynamics",
                    "world.rigid_dynamics": "rigid-dynamics",
                    "world.grasp_contact": "grasp-contact",
                    "world.bimanual_coordination": "bimanual-coordination",
                    "world.deformable_dynamics": "deformable-dynamics",
                    "world.human_motion": "human-motion",
                }[route[1]]
                prediction = invoke(
                    route[0],
                    route[1],
                    world_input,
                    WorldModelPrediction,
                    context={"plan_id": candidate.id, "policy": policy.policy},
                )
                if prediction.plan_id != candidate.id:
                    raise ValueError("World model returned the wrong plan ID")
                if prediction.model_kind != expected_model_kind:
                    raise ValueError(
                        f"{route[1]} returned {prediction.model_kind}; "
                        f"expected {expected_model_kind}"
                    )
                if not math.isclose(
                    prediction.states[-1].time_s,
                    self.profile.simulation_horizon_seconds,
                    abs_tol=max(1e-6, self.profile.simulation_step_seconds / 2),
                ):
                    raise ValueError("World model did not cover the configured prediction horizon")
                return prediction

            routes = world_capabilities(candidate)
            with ThreadPoolExecutor(
                max_workers=min(len(routes), 4), thread_name_prefix="moira-world"
            ) as world_pool:
                predictions = tuple(world_pool.map(predict, routes))
            reward = invoke(
                Layer.REWARD,
                "reward.task_progress",
                RewardInput(candidate, predictions, collision),
                RewardScore,
                context={"plan_id": candidate.id},
            )
            safety = invoke(
                Layer.SAFETY,
                "safety.risk",
                SafetyInput(
                    candidate,
                    predictions,
                    collision,
                    slip,
                    stability,
                    world,
                    policy,
                    personal,
                ),
                SafetyAssessment,
                local_authority=True,
                context={"plan_id": candidate.id},
            )
            if reward.plan_id != candidate.id or safety.plan_id != candidate.id:
                raise ValueError("Reward or safety model returned the wrong plan ID")
            prediction_risks = tuple(
                dict.fromkeys(
                    (
                        *collision.collisions,
                        *(risk for item in predictions for risk in item.risks),
                        *safety.reasons,
                        *(() if kinematics.feasible else kinematics.reasons),
                    )
                )
            )
            safe = safety.safe and collision.safe and kinematics.feasible
            simulation = SimulationOutcome(
                candidate.id,
                reward.score,
                safe,
                self.profile.simulation_horizon_seconds,
                prediction_risks,
                {
                    "world_models": tuple(item.model_kind for item in predictions),
                    "predicted_state_count": sum(len(item.states) for item in predictions),
                    "success_probability": min(item.success_probability for item in predictions),
                    "uncertainty": max(item.uncertainty for item in predictions),
                    "reward_components": dict(reward.components),
                    "risk_score": safety.risk_score,
                    "minimum_clearance_m": collision.minimum_clearance_m,
                },
            )
            emit(
                "simulation.completed",
                details={
                    "candidate_id": candidate.id,
                    "score": simulation.score,
                    "safe": simulation.safe,
                    "horizon_seconds": simulation.horizon_seconds,
                    "risks": simulation.risks,
                    "world_models": simulation.predicted["world_models"],
                },
            )
            return policy, trajectory, predictions, reward, safety, simulation

        workers = min(self.profile.simulation_workers, len(candidates))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="moira-sim") as pool:
            evaluations = tuple(pool.map(evaluate_candidate, candidates))
        policies = tuple(item[0] for item in evaluations)
        trajectories = tuple(item[1] for item in evaluations if item[1] is not None)
        world_predictions = tuple(prediction for item in evaluations for prediction in item[2])
        rewards = tuple(item[3] for item in evaluations)
        safety_assessments = tuple(item[4] for item in evaluations)
        simulations = tuple(item[5] for item in evaluations)
        if not any(outcome.safe for outcome in simulations):
            reasons = tuple(dict.fromkeys(risk for item in simulations for risk in item.risks))
            detail = "; ".join(reasons) if reasons else "risk threshold exceeded"
            raise RuntimeError(
                f"Every model-predicted plan was unsafe; physical execution is blocked: {detail}"
            )
        plan = invoke(
            Layer.PLANNING,
            "planning.select",
            PlanSelectionInput(intent, world, personal, candidates, simulations),
            FinalPlan,
        )
        emit(
            "plan.selected",
            details={
                "candidate_id": plan.candidate.id,
                "score": plan.simulation.score,
                "safe": plan.simulation.safe,
                "summary": plan.summary,
            },
        )
        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        simulation_by_id = {outcome.plan_id: outcome for outcome in simulations}
        if (
            _wire_equivalent(candidate_by_id.get(plan.candidate.id))
            != _wire_equivalent(plan.candidate)
            or _wire_equivalent(simulation_by_id.get(plan.simulation.plan_id))
            != _wire_equivalent(plan.simulation)
        ):
            raise ValueError("Planner selected a plan that was not supplied and simulated")
        if request.confirmed_intent is not None and (
            _wire_equivalent(intent) != _wire_equivalent(request.confirmed_intent)
            or _wire_equivalent(plan.candidate)
            != _wire_equivalent(request.confirmed_candidate)
        ):
            emit(
                "plan.confirmation_invalidated",
                details={
                    "reason": "fresh intent or selected plan changed",
                },
            )
            raise ConfirmedPlanChangedError(
                "Fresh intent or selected plan changed after confirmation"
            )
        policy_by_id = {item.plan_id: item for item in policies}
        trajectory_by_id = {item.plan_id: item for item in trajectories}
        selected_trajectory = trajectory_by_id.get(plan.candidate.id)
        if selected_trajectory is None:
            raise RuntimeError("Selected plan has no validated motion trajectory")
        world_pre_execute = None
        execution_guard = None
        if request.execute:
            human_error_policy = workspace.get("human_error_policy")
            if not isinstance(human_error_policy, Mapping):
                raise RuntimeError(
                    "Physical execution requires workspace.human_error_policy"
                )
            max_drift_m = human_error_policy.get("max_pre_execution_target_drift_m")
            monitor_between_steps = human_error_policy.get("monitor_scene_between_steps")
            if (
                not isinstance(max_drift_m, (int, float))
                or isinstance(max_drift_m, bool)
                or not math.isfinite(max_drift_m)
                or max_drift_m <= 0
            ):
                raise RuntimeError(
                    "Physical execution requires a finite positive "
                    "human_error_policy.max_pre_execution_target_drift_m"
                )
            if not isinstance(monitor_between_steps, bool):
                raise RuntimeError(
                    "Physical execution requires boolean "
                    "human_error_policy.monitor_scene_between_steps"
                )
            if pre_action_capture is None:
                raise RuntimeError("Physical execution requires pre-action scene capture")
            revalidation_frames = pre_action_capture()
            if (
                not isinstance(revalidation_frames, tuple)
                or not revalidation_frames
                or any(not isinstance(frame, CameraFrame) for frame in revalidation_frames)
            ):
                raise TypeError(
                    "pre_action_capture must return a non-empty tuple of CameraFrame objects"
                )
            world_pre_execute = invoke(
                Layer.PERCEPTION,
                "perception.scene",
                PerceptionInput(revalidation_frames, workspace),
                WorldState,
                context={"observation_phase": "pre_execution_revalidation"},
            )
            try:
                revalidation = validate_pre_execution_scene(
                    world,
                    world_pre_execute,
                    intent.target_object_ids,
                    max_drift_m,
                )
            except SceneChangedError as exc:
                stop_issues = self.emergency_stop()
                emit(
                    "scene.changed",
                    details={
                        "reason": str(exc),
                        "stop_issues": stop_issues,
                    },
                )
                raise
            emit(
                "scene.revalidated",
                details={
                    "target_drift_m": dict(revalidation.target_drift_m),
                    "maximum_allowed_drift_m": revalidation.maximum_allowed_drift_m,
                },
            )
            if monitor_between_steps:
                reference_world = world_pre_execute
                destination_ids = intent.destination_object_ids
                manipulated_ids = intent.manipulated_object_ids

                def execution_guard(
                    next_step: PlanStep,
                    completed_actions: tuple[str, ...],
                ) -> None:
                    del next_step
                    moving_actions = {
                        "lift",
                        "transfer",
                        "move",
                        "place",
                        "release",
                        "handover",
                        "insert",
                        "pour",
                    }
                    manipulated_still_stationary = not any(
                        action.casefold() in moving_actions for action in completed_actions
                    )
                    guarded_ids = tuple(
                        dict.fromkeys(
                            [
                                *destination_ids,
                                *(manipulated_ids if manipulated_still_stationary else ()),
                            ]
                        )
                    )
                    live_frames = pre_action_capture()
                    if (
                        not isinstance(live_frames, tuple)
                        or not live_frames
                        or any(not isinstance(frame, CameraFrame) for frame in live_frames)
                    ):
                        raise TypeError(
                            "execution scene guard capture must return CameraFrame objects"
                        )
                    live_world = invoke(
                        Layer.PERCEPTION,
                        "perception.scene",
                        PerceptionInput(live_frames, workspace),
                        WorldState,
                        context={"observation_phase": "during_execution_guard"},
                    )
                    try:
                        report = validate_pre_execution_scene(
                            reference_world,
                            live_world,
                            guarded_ids,
                            float(max_drift_m),
                        )
                    except SceneChangedError as exc:
                        emit("scene.changed", details={"reason": str(exc)})
                        raise
                    emit(
                        "scene.execution_revalidated",
                        details={
                            "completed_actions": completed_actions,
                            "guarded_target_ids": guarded_ids,
                            "target_drift_m": dict(report.target_drift_m),
                        },
                    )
        selected_arms = {
            arm for step in plan.candidate.steps for arm in step.arms
        }
        control_capability = (
            "control.bimanual" if len(selected_arms) > 1 else "control.single_arm"
        )
        control_decision = self.router.decide(
            Layer.CONTROL,
            control_capability,
            route_context,
        )
        if control_decision.runtime == "remote":
            raise RuntimeError("Remote components cannot hold motor authority")
        control = invoke_decision(
            control_decision,
            ControlInput(
                plan,
                world,
                request.execute,
                policy_by_id[plan.candidate.id],
                selected_trajectory,
                stability,
                execution_guard,
            ),
            ControlReport,
            local_authority=True,
        )
        if not isinstance(control, ControlReport) or control.plan_id != plan.candidate.id:
            raise TypeError("Bimanual controller returned an invalid report")
        if not request.execute and control.executed:
            raise RuntimeError("Controller executed motion during a plan-only request")
        emit(
            "control.finished",
            details={
                "executed": control.executed,
                "success": control.success,
                "telemetry_count": len(control.telemetry),
                "issues": control.issues,
            },
        )
        world_after = None
        if control.executed and post_action_capture is not None:
            verification_frames = post_action_capture()
            if (
                not isinstance(verification_frames, tuple)
                or not verification_frames
                or any(not isinstance(frame, CameraFrame) for frame in verification_frames)
            ):
                raise TypeError(
                    "post_action_capture must return a non-empty tuple of CameraFrame objects"
                )
            world_after = invoke(
                Layer.PERCEPTION,
                "perception.scene",
                PerceptionInput(verification_frames, workspace),
                WorldState,
                context={"observation_phase": "after_execution"},
            )
        outcome = invoke(
            Layer.OUTCOME,
            "outcome.verify",
            OutcomeInput(plan, control, world, world_after),
            OutcomeReport,
        )
        if outcome.plan_id != plan.candidate.id:
            raise ValueError("Outcome verifier returned the wrong plan ID")
        if not control.executed and outcome.status != "planned":
            raise ValueError("Outcome verifier contradicted an unexecuted control report")
        if control.executed and not control.success and outcome.status == "succeeded":
            raise ValueError("Outcome verifier contradicted failed control telemetry")
        emit(
            "outcome.verified",
            details={
                "status": outcome.status,
                "confidence": outcome.confidence,
                "camera_verified": world_after is not None,
            },
        )
        failure = invoke(
            Layer.FAILURE,
            "failure.classify",
            FailureClassificationInput(outcome, control, slip),
            FailureReport,
        )
        if outcome.status != "failed" and failure.failure is not None:
            raise ValueError("Failure classifier reported a failure for a non-failed outcome")
        load = invoke(
            Layer.LOAD,
            "load.estimate",
            LoadEstimationInput(control, world, plan),
            LoadEstimate,
            local_authority=True,
        )
        selected_predictions = tuple(
            item for item in world_predictions if item.plan_id == plan.candidate.id
        )
        prediction_error = invoke(
            Layer.FEEDBACK,
            "feedback.prediction_error",
            PredictionErrorInput(selected_predictions, outcome, control),
            PredictionErrorReport,
            local_authority=True,
        )
        feedback = invoke(
            Layer.FEEDBACK,
            "feedback.learn",
            FeedbackInput(
                request.user_id,
                transcript,
                world,
                plan,
                control,
                outcome,
                failure,
                load,
                prediction_error,
            ),
            FeedbackReport,
        )
        recorded = invoke(
            Layer.PERSONAL,
            "personal.record",
            MemoryRecord(request.user_id, transcript, plan.summary, feedback),
            PersonalContext,
        )
        if recorded.user_id != request.user_id:
            raise ValueError("Personal-memory component recorded the wrong user")
        emit(
            "feedback.saved",
            details={
                "learned_facts": feedback.learned_facts,
                "adjustments": feedback.next_time_adjustments,
                "prediction_error_samples": prediction_error.sample_count,
            },
        )
        status = "Completed" if outcome.status == "succeeded" else "Failed"
        if outcome.status == "uncertain":
            status = "Uncertain"
        elif not control.executed:
            status = "Planned"
        response_text = f"{status}: {plan.summary}"
        response_audio = None
        if request.speak:
            response_audio = invoke(
                Layer.VOICE,
                "voice.synthesize",
                SpeechSynthesisInput(response_text, request.user_id),
                bytes,
            )
            if not response_audio:
                raise ValueError("Voice synthesizer returned empty audio")
        emit(
            "pipeline.completed",
            details={
                "selected_plan": plan.candidate.id,
                "executed": control.executed,
                "outcome": outcome.status,
                "response_audio_bytes": len(response_audio or b""),
            },
        )
        return PhysicalAIResult(
            intent=intent,
            world=world,
            personal=personal,
            candidates=candidates,
            simulations=simulations,
            plan=plan,
            control=control,
            feedback=feedback,
            response_text=response_text,
            response_audio=response_audio,
            routing=tuple(decisions),
            grasps=grasps,
            policies=policies,
            trajectories=trajectories,
            world_predictions=world_predictions,
            rewards=rewards,
            safety=safety_assessments,
            contact=contact,
            slip=slip,
            force=force,
            stability=stability,
            outcome=outcome,
            failure=failure,
            load=load,
            prediction_error=prediction_error,
            world_pre_execute=world_pre_execute,
            world_after=world_after,
        )


def capture_frames(sources: Sequence[CameraSource]) -> tuple[CameraFrame, ...]:
    if not sources:
        raise ValueError("At least one camera source is required")
    with ThreadPoolExecutor(
        max_workers=min(2, len(sources)), thread_name_prefix="moira-camera"
    ) as pool:
        frames = tuple(pool.map(lambda source: source.capture(), sources))
    if any(not isinstance(frame, CameraFrame) for frame in frames):
        raise TypeError("Camera sources must return CameraFrame objects")
    return frames


class ArmDriver(Protocol):
    def execute(self, step: PlanStep, world: WorldState) -> ArmTelemetry: ...
    def execute_chunk(
        self,
        chunk: ActionChunk,
        trajectory: MotionTrajectory,
        world: WorldState,
    ) -> ArmTelemetry: ...
    def stop(self) -> None: ...


class BimanualControlComponent:
    """Run one- or two-arm steps and stop every installed driver on an exception."""

    def __init__(
        self,
        left: ArmDriver,
        right: ArmDriver | None = None,
        *,
        timeout_margin_seconds: float = 1.0,
        max_slip_probability: float = 0.35,
    ) -> None:
        drivers = {"left": left}
        if right is not None:
            drivers["right"] = right
        for name, driver in drivers.items():
            if not callable(getattr(driver, "execute", None)) or not callable(
                getattr(driver, "stop", None)
            ):
                raise TypeError(f"{name} arm driver must implement execute() and stop()")
        if (
            not isinstance(timeout_margin_seconds, (int, float))
            or not math.isfinite(timeout_margin_seconds)
            or timeout_margin_seconds < 0
        ):
            raise ValueError("Controller timeout margin must be finite and nonnegative")
        self.drivers = drivers
        self.timeout_margin_seconds = timeout_margin_seconds
        if (
            not isinstance(max_slip_probability, (int, float))
            or isinstance(max_slip_probability, bool)
            or not math.isfinite(max_slip_probability)
            or not 0 < max_slip_probability <= 1
        ):
            raise ValueError("Controller max_slip_probability must be in (0, 1]")
        self.max_slip_probability = float(max_slip_probability)
        self._stop_latch = Event()

    @classmethod
    def from_robot_model(
        cls,
        left: ArmDriver,
        right: ArmDriver,
        robot_model: RobotModel,
    ) -> BimanualControlComponent:
        robot_model.require_bimanual_motion_ready()
        return cls(
            left,
            right,
            timeout_margin_seconds=robot_model.controller_timeout_margin_s,
            max_slip_probability=robot_model.max_slip_probability,
        )

    @classmethod
    def from_installed_arms(
        cls,
        robot_model: RobotModel,
        *,
        left: ArmDriver | None = None,
        right: ArmDriver | None = None,
    ) -> BimanualControlComponent:
        robot_model.require_motion_ready()
        drivers = {
            arm for arm, driver in (("left", left), ("right", right)) if driver is not None
        }
        expected = set(robot_model.servo_controller.installed_arms)
        if drivers != expected:
            raise ValueError("Arm drivers must exactly match the installed physical arms")
        if left is None:
            raise ValueError("The primary left control slot requires an installed driver")
        return cls(
            left,
            right,
            timeout_margin_seconds=robot_model.controller_timeout_margin_s,
            max_slip_probability=robot_model.max_slip_probability,
        )

    def emergency_stop(self) -> tuple[str, ...]:
        """Stop every installed arm and report hardware stop-hook failures."""

        self._stop_latch.set()
        issues: list[str] = []
        for arm, driver in self.drivers.items():
            try:
                driver.stop()
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                issues.append(f"{arm}: {detail}")
        return tuple(issues)

    def stop(self) -> None:
        issues = self.emergency_stop()
        if issues:
            raise RuntimeError("; ".join(issues))

    def run(self, request: ControlInput) -> ControlReport:
        if not isinstance(request, ControlInput):
            raise TypeError("Bimanual controller expects ControlInput")
        if not request.execute:
            return ControlReport(request.plan.candidate.id, False, True, ())
        if self._stop_latch.is_set():
            return ControlReport(
                request.plan.candidate.id,
                False,
                False,
                (),
                ("Emergency stop latch is active; check the workspace and restart runtime",),
            )
        if request.tactile is not None and not request.tactile.stable:
            return ControlReport(
                request.plan.candidate.id,
                False,
                False,
                (),
                ("Local tactile guard blocked an unstable grasp",),
            )
        required_arms = {arm for step in request.plan.candidate.steps for arm in step.arms}
        unavailable = sorted(required_arms - set(self.drivers))
        if unavailable:
            return ControlReport(
                request.plan.candidate.id,
                False,
                False,
                (),
                (f"Plan requires unavailable arm control slots: {', '.join(unavailable)}",),
            )
        chunks_by_step: dict[str, list[ActionChunk]] = {}
        if request.policy is not None:
            for chunk in request.policy.chunks:
                chunks_by_step.setdefault(chunk.step_id, []).append(chunk)
        trajectories_by_chunk = (
            {chunk.id: request.trajectory.for_chunk(chunk.id) for chunk in request.policy.chunks}
            if request.policy is not None and request.trajectory is not None
            else {}
        )

        def execute(
            arm: str,
            step: PlanStep,
            chunk: ActionChunk | None,
        ) -> ArmTelemetry:
            driver = self.drivers[arm]
            if chunk is None:
                result = driver.execute(step, request.world)
            else:
                execute_chunk = getattr(driver, "execute_chunk", None)
                if not callable(execute_chunk):
                    raise RuntimeError(f"{arm} arm driver cannot execute validated action chunks")
                result = execute_chunk(
                    chunk,
                    trajectories_by_chunk[chunk.id],
                    request.world,
                )
            if not isinstance(result, ArmTelemetry):
                raise TypeError(f"{arm} arm driver returned invalid telemetry")
            if result.arm != arm or result.step_id != step.id:
                raise RuntimeError(
                    f"{arm} arm telemetry identity does not match step {step.id}"
                )
            return result

        telemetry: list[ArmTelemetry] = []
        issues: list[str] = []
        completed_actions: list[str] = []

        def stop_all() -> tuple[str, ...]:
            return tuple(
                issue.replace(": ", " arm emergency stop failed: ", 1)
                for issue in self.emergency_stop()
            )

        try:
            for step in request.plan.candidate.steps:
                if request.execution_guard is not None:
                    request.execution_guard(step, tuple(completed_actions))
                commands: tuple[ActionChunk | None, ...] = (
                    tuple(chunks_by_step[step.id])
                    if request.policy is not None
                    else (None,)
                )
                for chunk in commands:
                    if self._stop_latch.is_set():
                        issues.append(f"{step.id}: emergency stop latch activated")
                        break
                    command_id = chunk.id if chunk is not None else step.id
                    duration = (
                        chunk.duration_seconds if chunk is not None else step.duration_seconds
                    )
                    pool = ThreadPoolExecutor(
                        max_workers=len(step.arms), thread_name_prefix="moira-arm"
                    )
                    futures = [pool.submit(execute, arm, step, chunk) for arm in step.arms]
                    _, pending = wait(
                        futures,
                        timeout=duration + self.timeout_margin_seconds,
                    )
                    if pending:
                        issues.append(f"{command_id}: arm command timed out")
                        issues.extend(stop_all())
                        for future in pending:
                            future.cancel()
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    pool.shutdown(wait=True)
                    results = tuple(future.result() for future in futures)
                    telemetry.extend(results)
                    if self._stop_latch.is_set():
                        issues.append(f"{command_id}: emergency stop latch activated")
                        break
                    reported_issues = [result.issue for result in results if result.issue]
                    if reported_issues:
                        issues.extend(reported_issues)
                        issues.extend(stop_all())
                        break
                    live_slip = [
                        result
                        for result in results
                        if result.slip_probability is not None
                        and result.slip_probability > self.max_slip_probability
                    ]
                    if live_slip:
                        issues.extend(stop_all())
                        issues.append(f"{command_id}: tactile reflex detected slip")
                        break
                    if not all(result.success for result in results):
                        issues.extend(stop_all())
                        issues.append(f"{command_id}: arm command failed")
                        break
                if issues:
                    break
                completed_actions.append(step.action)
        except Exception as exc:
            issues.extend(stop_all())
            issues.append(str(exc).strip() or type(exc).__name__)
        success = bool(telemetry) and all(item.success for item in telemetry) and not issues
        return ControlReport(
            request.plan.candidate.id,
            True,
            success,
            tuple(telemetry),
            tuple(issues),
        )
