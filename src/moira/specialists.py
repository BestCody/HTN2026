"""Pi-side specialists and deterministic model fixtures for physical-AI E2E runs.

Production neural specialists use the same contracts through ``BasetenComponent``.
The implementations here provide useful bounded algorithms for offline validation;
they are never selected as fallbacks for failed cloud deployments.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .robot_config import RobotModel

from .physical import (
    ActionChunk,
    CollisionCheckInput,
    CollisionReport,
    ContactEstimate,
    FailureClassificationInput,
    FailureReport,
    ForceEstimate,
    GraspPlan,
    GraspPlanningInput,
    GraspPose,
    GraspStability,
    KinematicsInput,
    KinematicsSolution,
    LoadEstimate,
    LoadEstimationInput,
    MotionTrajectory,
    OutcomeInput,
    OutcomeReport,
    PolicyInput,
    PolicyPlan,
    PredictedState,
    PredictionErrorInput,
    PredictionErrorReport,
    RewardInput,
    RewardScore,
    SafetyAssessment,
    SafetyInput,
    SlipEstimate,
    TactileInput,
    TrajectoryPlanningInput,
    TrajectoryPoint,
    WorldModelInput,
    WorldModelPrediction,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class AnalyticGraspPlanner:
    """Generate bounded top/side 6-DoF grasps from object pose and dimensions."""

    def run(self, request: GraspPlanningInput) -> tuple[GraspPlan, ...]:
        if not isinstance(request, GraspPlanningInput):
            raise TypeError("Grasp planner expects GraspPlanningInput")
        objects = {item.id: item for item in request.world.objects}
        max_width = float(request.gripper_geometry["max_width_m"])
        blocked = any(value in ("blocked", "collision") for value in request.world.hazards)
        plans = []
        for object_id in request.target_object_ids:
            item = objects[object_id]
            attributes = item.attributes or {}
            width = attributes.get("width_m")
            height = attributes.get("height_m")
            for name, value in (("width_m", width), ("height_m", height)):
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        f"Offline grasp fixture requires a measured {name} for {object_id}"
                    )
            if width > max_width:
                raise ValueError(f"{object_id} is wider than the configured gripper aperture")
            width = float(width)
            height = float(height)
            collision = 0.35 if blocked else 0.02
            excluded = " ".join(request.excluded_regions).lower()
            top_score = 0.70 if "top" in excluded else 0.94
            side_score = 0.65 if "side" in excluded else 0.88
            up_index = {"x": 0, "y": 1, "z": 2}[request.world.up_axis]
            side_index = next(index for index in (2, 1, 0) if index != up_index)
            top_position = list(item.position_m)
            top_position[up_index] += height / 2
            side_position = list(item.position_m)
            side_position[side_index] -= width / 2
            poses = (
                GraspPose(
                    f"{object_id}-top",
                    object_id,
                    tuple(top_position),
                    (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),
                    width,
                    top_score * (1 - collision),
                    collision,
                ),
                GraspPose(
                    f"{object_id}-side",
                    object_id,
                    tuple(side_position),
                    (0.0, 0.0, 0.0, 1.0),
                    width,
                    side_score * (1 - collision),
                    collision,
                ),
            )
            plans.append(GraspPlan(object_id, tuple(sorted(poses, key=lambda pose: -pose.score))))
        return tuple(plans)


class SpecializedManipulationPolicy:
    """Offline policy fixture converting semantic steps into bounded chunks."""

    def __init__(self, policy_name: str) -> None:
        if not isinstance(policy_name, str) or not policy_name.strip():
            raise ValueError("policy_name must be non-empty")
        self.policy_name = policy_name

    def run(self, request: PolicyInput) -> PolicyPlan:
        if not isinstance(request, PolicyInput):
            raise TypeError("Manipulation policy expects PolicyInput")
        objects = {item.id: item for item in request.world.objects}
        grasps = {item.target_object_id: item.grasps[0] for item in request.grasps}
        up_index = {"x": 0, "y": 1, "z": 2}[request.world.up_axis]

        def height(item: object) -> float:
            attributes = getattr(item, "attributes", None) or {}
            value = attributes.get("height_m")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("Placement planning requires measured object height_m")
            result = float(value)
            if not math.isfinite(result) or result <= 0:
                raise ValueError("Placement planning requires positive object height_m")
            return result

        chunks = []
        for index, step in enumerate(request.candidate.steps):
            target = objects.get(step.target_object_id) if step.target_object_id else None
            grasp = grasps.get(step.target_object_id)
            parameters = step.parameters or {}
            position_key = (
                "destination_position_m"
                if step.action.lower() in ("transfer", "release", "place", "handover")
                else "target_position_m"
            )
            raw_position = parameters.get(
                position_key,
                parameters.get(
                    "target_position_m",
                    target.position_m if target is not None else None,
                ),
            )
            if raw_position is None:
                raise ValueError(f"Offline policy fixture needs a target pose for {step.id}")
            action = step.action.lower()
            position = tuple(float(value) for value in raw_position)
            if len(position) != 3 or not all(math.isfinite(value) for value in position):
                raise ValueError(f"Offline policy fixture needs a finite 3D pose for {step.id}")
            if grasp is not None:
                orientation = grasp.orientation_xyzw
                width = grasp.width_m
                if action in ("approach", "grasp", "lift", "hold"):
                    position = grasp.position_m
            else:
                orientation = (0.0, 0.0, 0.0, 1.0)
                width_value = parameters.get("gripper_width_m")
                if width_value is None:
                    raise ValueError(
                        f"Offline policy fixture needs an explicit gripper width for {step.id}"
                    )
                width = float(width_value)
            clearance_value = parameters.get("clearance_m", 0.0)
            if (
                not isinstance(clearance_value, (int, float))
                or isinstance(clearance_value, bool)
                or not math.isfinite(clearance_value)
                or clearance_value < 0
            ):
                raise ValueError(f"Offline policy fixture needs valid clearance for {step.id}")
            clearance = float(clearance_value)
            if action in ("transfer", "release", "place"):
                destination_id = parameters.get("destination_object_id")
                destination = (
                    objects.get(destination_id) if isinstance(destination_id, str) else None
                )
                if destination is not None and target is not None and grasp is not None:
                    # The cloud planner names the destination object's centre.
                    # Convert it into a gripper pose that puts the manipulated
                    # object on the destination support surface while preserving
                    # the selected grasp offset.
                    placement = list(destination.position_m)
                    placement[up_index] += (height(destination) + height(target)) / 2
                    grasp_offset = tuple(
                        grasp.position_m[axis] - target.position_m[axis] for axis in range(3)
                    )
                    position = tuple(
                        placement[axis] + grasp_offset[axis] for axis in range(3)
                    )
                    orientation = grasp.orientation_xyzw
                    width = grasp.width_m
            if action in ("approach", "lift", "transfer"):
                raised = list(position)
                raised[up_index] += clearance
                position = tuple(raised)
            mass = (
                target.estimated_mass_kg
                if target is not None and target.estimated_mass_kg is not None
                else request.personal.learned_object_masses.get(target.label)
                if target is not None
                else None
            )
            if mass is None:
                raise ValueError(f"Offline policy fixture needs a measured mass for {step.id}")
            per_arm_force = mass * 9.81 * 1.5 / len(step.arms)
            chunks.append(
                ActionChunk(
                    f"{request.candidate.id}-chunk-{index}",
                    step.id,
                    self.policy_name,
                    step.arms,
                    step.duration_seconds,
                    step.target_object_id,
                    (*position, *orientation),
                    width,
                    max(2.0, per_arm_force),
                )
            )
        return PolicyPlan(request.candidate.id, self.policy_name, tuple(chunks))


class PlanarBimanualIK:
    """Bounded analytic IK for calibrated yaw/pitch tabletop arm layouts."""

    def __init__(
        self,
        *,
        upper_arm_m: float = 0.45,
        forearm_m: float = 0.45,
        shoulder_height_m: float = 0.60,
        shoulder_offset_m: float = 0.24,
        include_wrist_joint: bool = True,
        joint_limits_rad: tuple[tuple[float, float], ...] | None = None,
        up_axis: str = "z",
        kinematic_layout: str = "yaw_shoulder_elbow",
        effective_reach_m: float | None = None,
        fixed_link_reach_tolerance_m: float | None = None,
    ) -> None:
        if shoulder_height_m <= 0:
            raise ValueError("IK shoulder height must be positive")
        if kinematic_layout not in (
            "yaw_shoulder_elbow",
            "yaw_shoulder_fixed_link",
        ):
            raise ValueError("IK kinematic layout is unsupported")
        if kinematic_layout == "yaw_shoulder_elbow":
            if min(upper_arm_m, forearm_m) <= 0:
                raise ValueError("IK two-link lengths must be positive")
        elif (
            effective_reach_m is None
            or fixed_link_reach_tolerance_m is None
            or effective_reach_m <= 0
            or not 0 < fixed_link_reach_tolerance_m <= effective_reach_m
        ):
            raise ValueError(
                "Fixed-link IK needs a positive effective reach and bounded tolerance"
            )
        if not isinstance(shoulder_offset_m, (int, float)) or not math.isfinite(
            shoulder_offset_m
        ) or shoulder_offset_m < 0:
            raise ValueError("IK shoulder offset must be finite and nonnegative")
        self.upper_arm_m = upper_arm_m
        self.forearm_m = forearm_m
        self.shoulder_height_m = shoulder_height_m
        self.shoulder_offset_m = shoulder_offset_m
        self.include_wrist_joint = include_wrist_joint
        self.joint_limits_rad = joint_limits_rad
        self.kinematic_layout = kinematic_layout
        self.effective_reach_m = effective_reach_m
        self.fixed_link_reach_tolerance_m = fixed_link_reach_tolerance_m
        if up_axis not in ("y", "z"):
            raise ValueError("Planar IK supports only Y-up or Z-up robot frames")
        self.up_axis = up_axis
        if kinematic_layout == "yaw_shoulder_fixed_link" and include_wrist_joint:
            raise ValueError("Fixed-link IK cannot include a wrist joint")
        expected = (
            2
            if kinematic_layout == "yaw_shoulder_fixed_link"
            else (4 if include_wrist_joint else 3)
        )
        if joint_limits_rad is not None and (
            len(joint_limits_rad) != expected
            or any(lower >= upper for lower, upper in joint_limits_rad)
        ):
            raise ValueError(f"IK joint limits must contain {expected} valid ranges")

    @classmethod
    def from_robot_model(cls, model: RobotModel) -> PlanarBimanualIK:
        model.require_motion_ready()
        return cls(
            upper_arm_m=model.upper_arm_m,
            forearm_m=model.forearm_m,
            shoulder_height_m=model.shoulder_height_m,
            shoulder_offset_m=(
                model.shoulder_offset_m if model.shoulder_offset_m is not None else 0.0
            ),
            include_wrist_joint=False,
            joint_limits_rad=tuple(
                (joint.lower_rad, joint.upper_rad) for joint in model.kinematic_joints
            ),
            up_axis=model.up_axis,
            kinematic_layout=model.kinematic_layout,
            effective_reach_m=model.effective_reach_m,
            fixed_link_reach_tolerance_m=model.fixed_link_reach_tolerance_m,
        )

    def run(self, request: KinematicsInput) -> KinematicsSolution:
        if not isinstance(request, KinematicsInput):
            raise TypeError("IK solver expects KinematicsInput")
        if request.world.up_axis != self.up_axis:
            raise ValueError(
                f"IK expects a {self.up_axis.upper()}-up world, "
                f"but perception returned {request.world.up_axis.upper()}-up"
            )
        targets: dict[str, dict[str, tuple[float, ...]]] = {}
        reasons = []
        for chunk in request.policy.chunks:
            x, y, z = chunk.target_pose[:3]
            per_arm = {}
            for arm in chunk.arms:
                shoulder_lateral = (
                    self.shoulder_offset_m if arm == "left" else -self.shoulder_offset_m
                )
                if self.up_axis == "y":
                    lateral = z
                    vertical = y - self.shoulder_height_m
                else:
                    lateral = y
                    vertical = z - self.shoulder_height_m
                radial = math.hypot(x, lateral - shoulder_lateral)
                distance = math.hypot(radial, vertical)
                base = math.atan2(lateral - shoulder_lateral, x)
                if self.kinematic_layout == "yaw_shoulder_fixed_link":
                    if (
                        abs(distance - self.effective_reach_m)
                        > self.fixed_link_reach_tolerance_m
                    ):
                        reasons.append(
                            f"{chunk.id}/{arm}: target is outside the fixed-link arc"
                        )
                    shoulder = math.atan2(vertical, radial)
                    joint_target = (base, shoulder)
                else:
                    low = abs(self.upper_arm_m - self.forearm_m) + 1e-6
                    high = self.upper_arm_m + self.forearm_m - 1e-6
                    if not low <= distance <= high:
                        reasons.append(
                            f"{chunk.id}/{arm}: target is outside the IK workspace"
                        )
                    bounded = max(low, min(high, distance))
                    elbow_cos = (
                        bounded * bounded
                        - self.upper_arm_m * self.upper_arm_m
                        - self.forearm_m * self.forearm_m
                    ) / (2 * self.upper_arm_m * self.forearm_m)
                    elbow = math.acos(_clamp(elbow_cos, -1.0, 1.0))
                    shoulder = math.atan2(vertical, radial) - math.atan2(
                        self.forearm_m * math.sin(elbow),
                        self.upper_arm_m + self.forearm_m * math.cos(elbow),
                    )
                    wrist = -shoulder - elbow
                    joint_target = (
                        (base, shoulder, elbow, wrist)
                        if self.include_wrist_joint
                        else (base, shoulder, elbow)
                    )
                if self.joint_limits_rad is not None:
                    for index, (value, limits) in enumerate(
                        zip(joint_target, self.joint_limits_rad, strict=True), start=1
                    ):
                        if not limits[0] <= value <= limits[1]:
                            reasons.append(
                                f"{chunk.id}/{arm}: joint {index} is outside calibrated limits"
                            )
                per_arm[arm] = joint_target
            targets[chunk.id] = per_arm
        return KinematicsSolution(
            request.policy.plan_id,
            targets,
            not reasons,
            tuple(dict.fromkeys(reasons)),
        )


class BoundedTrajectoryPlanner:
    """Interpolate targets within calibrated joint and velocity envelopes."""

    def __init__(
        self,
        *,
        frequency_hz: float = 10.0,
        joint_limit_rad: float = math.pi,
        joint_limits_rad: tuple[tuple[float, float], ...] | None = None,
        max_joint_velocity_rad_s: float = math.pi,
        joint_velocity_limits_rad_s: tuple[float, ...] | None = None,
        max_gripper_velocity_m_s: float = 0.1,
        max_gripper_force_n: float = 20.0,
    ) -> None:
        for name, value in (
            ("frequency_hz", frequency_hz),
            ("joint_limit_rad", joint_limit_rad),
            ("max_joint_velocity_rad_s", max_joint_velocity_rad_s),
            ("max_gripper_velocity_m_s", max_gripper_velocity_m_s),
            ("max_gripper_force_n", max_gripper_force_n),
        ):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self.frequency_hz = frequency_hz
        self.joint_limit_rad = joint_limit_rad
        self.joint_limits_rad = joint_limits_rad
        self.max_joint_velocity_rad_s = max_joint_velocity_rad_s
        self.joint_velocity_limits_rad_s = joint_velocity_limits_rad_s
        self.max_gripper_velocity_m_s = max_gripper_velocity_m_s
        self.max_gripper_force_n = max_gripper_force_n
        if joint_limits_rad is not None and any(
            lower >= upper for lower, upper in joint_limits_rad
        ):
            raise ValueError("Per-joint trajectory limits must be ordered")
        if joint_velocity_limits_rad_s is not None and (
            any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in joint_velocity_limits_rad_s
            )
            or (
                joint_limits_rad is not None
                and len(joint_velocity_limits_rad_s) != len(joint_limits_rad)
            )
        ):
            raise ValueError("Per-joint velocity limits must be positive and match the joints")

    @classmethod
    def from_robot_model(cls, model: RobotModel) -> BoundedTrajectoryPlanner:
        model.require_motion_ready()
        return cls(
            frequency_hz=model.trajectory_frequency_hz,
            joint_limits_rad=tuple(
                (joint.lower_rad, joint.upper_rad) for joint in model.kinematic_joints
            ),
            joint_velocity_limits_rad_s=tuple(
                joint.max_velocity_rad_s for joint in model.kinematic_joints
            ),
            max_gripper_velocity_m_s=model.max_gripper_velocity_m_s,
            max_gripper_force_n=model.max_gripper_force_n,
        )

    def run(self, request: TrajectoryPlanningInput) -> MotionTrajectory:
        if not isinstance(request, TrajectoryPlanningInput):
            raise TypeError("Trajectory planner expects TrajectoryPlanningInput")
        current: dict[str, tuple[float, ...]] = (
            dict(request.robot_state.joint_positions) if request.robot_state is not None else {}
        )
        current_grippers: dict[str, float] = (
            dict(request.robot_state.gripper_widths_m)
            if request.robot_state is not None
            else {}
        )
        points = []
        elapsed = 0.0
        for chunk in request.policy.chunks:
            if chunk.force_limit_n > self.max_gripper_force_n:
                raise ValueError(
                    f"Action chunk {chunk.id} exceeds the calibrated gripper force limit"
                )
            targets = request.kinematics.joint_targets.get(chunk.id, {})
            bounded_targets: dict[str, tuple[float, ...]] = {}
            for arm, target in targets.items():
                if self.joint_limits_rad is not None:
                    if len(target) != len(self.joint_limits_rad):
                        raise ValueError("Trajectory target does not match robot joint count")
                    if any(
                        not lower <= value <= upper
                        for value, (lower, upper) in zip(
                            target, self.joint_limits_rad, strict=True
                        )
                    ):
                        raise ValueError("Trajectory target exceeds calibrated joint limits")
                    bounded_targets[arm] = target
                else:
                    bounded_targets[arm] = tuple(
                        max(-self.joint_limit_rad, min(self.joint_limit_rad, value))
                        for value in target
                    )
            count = max(1, math.ceil(chunk.duration_seconds * self.frequency_hz))
            starts = {
                arm: current.get(arm, tuple(0.0 for _ in target))
                for arm, target in bounded_targets.items()
            }
            target_grippers = {arm: chunk.gripper_width_m for arm in chunk.arms}
            start_grippers = {
                arm: current_grippers.get(arm, width) for arm, width in target_grippers.items()
            }
            if any(
                abs(width - start_grippers[arm]) / chunk.duration_seconds
                > self.max_gripper_velocity_m_s
                for arm, width in target_grippers.items()
            ):
                raise ValueError(
                    f"Action chunk {chunk.id} exceeds calibrated gripper velocity limits"
                )
            for arm, target in bounded_targets.items():
                start = starts[arm]
                if len(start) != len(target):
                    raise ValueError(f"Trajectory start joint count does not match {arm} target")
                if self.joint_limits_rad is not None and any(
                    not lower <= value <= upper
                    for value, (lower, upper) in zip(
                        start, self.joint_limits_rad, strict=True
                    )
                ):
                    raise ValueError(f"{arm} robot state exceeds calibrated joint limits")
                velocity_limits = self.joint_velocity_limits_rad_s or tuple(
                    self.max_joint_velocity_rad_s for _ in target
                )
                if len(velocity_limits) != len(target):
                    raise ValueError("Trajectory velocity limits do not match robot joint count")
                if any(
                    abs(finish - initial) / chunk.duration_seconds > limit
                    for initial, finish, limit in zip(
                        start, target, velocity_limits, strict=True
                    )
                ):
                    raise ValueError(
                        f"Action chunk {chunk.id} exceeds calibrated joint velocity limits"
                    )
            for index in range(1, count + 1):
                fraction = index / count
                sample = dict(current)
                for arm, bounded in bounded_targets.items():
                    sample[arm] = tuple(
                        start + fraction * (finish - start)
                        for start, finish in zip(starts[arm], bounded, strict=True)
                    )
                sample_grippers = dict(current_grippers)
                for arm, width in target_grippers.items():
                    sample_grippers[arm] = start_grippers[arm] + fraction * (
                        width - start_grippers[arm]
                    )
                elapsed += chunk.duration_seconds / count
                points.append(
                    TrajectoryPoint(
                        round(elapsed, 6),
                        sample,
                        chunk.id,
                        sample_grippers,
                    )
                )
            current.update(bounded_targets)
            current_grippers.update(target_grippers)
        return MotionTrajectory(request.policy.plan_id, tuple(points), elapsed)


class ConservativeCollisionChecker:
    """Apply hard workspace and joint-envelope checks before motor authority."""

    def __init__(
        self,
        *,
        joint_limit_rad: float = math.pi,
        joint_limits_rad: tuple[tuple[float, float], ...] | None = None,
        required_clearance_m: float = 0.02,
    ) -> None:
        for name, value in (
            ("joint_limit_rad", joint_limit_rad),
            ("required_clearance_m", required_clearance_m),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self.joint_limit_rad = joint_limit_rad
        self.joint_limits_rad = joint_limits_rad
        self.required_clearance_m = required_clearance_m

    @classmethod
    def from_robot_model(cls, model: RobotModel) -> ConservativeCollisionChecker:
        model.require_motion_ready()
        return cls(
            joint_limits_rad=tuple(
                (joint.lower_rad, joint.upper_rad) for joint in model.kinematic_joints
            ),
            required_clearance_m=model.required_clearance_m,
        )

    def run(self, request: CollisionCheckInput) -> CollisionReport:
        if not isinstance(request, CollisionCheckInput):
            raise TypeError("Collision checker expects CollisionCheckInput")
        collisions = []
        if any(item.lower() in ("blocked", "collision") for item in request.world.hazards):
            collisions.append("workspace perception reports an obstructed path")
        for point in request.trajectory.points:
            invalid = False
            for values in point.joint_positions.values():
                if self.joint_limits_rad is None:
                    invalid = invalid or any(abs(joint) > self.joint_limit_rad for joint in values)
                else:
                    invalid = (
                        invalid
                        or len(values) != len(self.joint_limits_rad)
                        or any(
                            not lower <= joint <= upper
                            for joint, (lower, upper) in zip(
                                values, self.joint_limits_rad, strict=True
                            )
                        )
                    )
            if invalid:
                collisions.append(f"joint envelope exceeded at {point.time_s:.3f}s")
                break
        clearance = request.world.workspace.get("minimum_clearance_m")
        if (
            not isinstance(clearance, (int, float))
            or isinstance(clearance, bool)
            or not math.isfinite(clearance)
            or clearance < 0
        ):
            reported = 0.0
            collisions.append("workspace perception did not report a valid minimum clearance")
        else:
            reported = float(clearance)
        if reported < self.required_clearance_m:
            collisions.append("workspace clearance is below the configured safety margin")
        return CollisionReport(
            request.trajectory.plan_id,
            not collisions,
            max(0.0, reported),
            tuple(dict.fromkeys(collisions)),
        )


class TactileSignalModel:
    """Estimate force, slip, and grasp stability from bounded tactile samples."""

    def __init__(self, *, min_stability_score: float = 0.60) -> None:
        if (
            not isinstance(min_stability_score, (int, float))
            or isinstance(min_stability_score, bool)
            or not math.isfinite(min_stability_score)
            or not 0 < min_stability_score <= 1
        ):
            raise ValueError("min_stability_score must be in (0, 1]")
        self.min_stability_score = float(min_stability_score)

    @classmethod
    def from_robot_model(cls, model: RobotModel) -> TactileSignalModel:
        model.require_motion_ready()
        return cls(min_stability_score=model.min_grasp_stability_score)

    def run(
        self, request: TactileInput
    ) -> ContactEstimate | SlipEstimate | ForceEstimate | GraspStability:
        if not isinstance(request, TactileInput):
            raise TypeError("Tactile model expects TactileInput")
        return _TactileHead(request.capability, self.min_stability_score).run(request)

    @classmethod
    def for_capability(
        cls,
        capability: Literal["contact", "slip", "force", "stability"],
        *,
        min_stability_score: float = 0.60,
    ) -> _TactileHead:
        return _TactileHead(capability, min_stability_score)


class _TactileHead:
    def __init__(
        self,
        capability: Literal["contact", "slip", "force", "stability"],
        min_stability_score: float,
    ) -> None:
        self.capability = capability
        self.min_stability_score = min_stability_score

    def run(
        self, request: TactileInput
    ) -> ContactEstimate | SlipEstimate | ForceEstimate | GraspStability:
        if not isinstance(request, TactileInput):
            raise TypeError("Tactile model expects TactileInput")
        normal: dict[str, float] = defaultdict(float)
        shear: dict[str, float] = defaultdict(float)
        for sample in request.samples:
            if sample.contact:
                normal[sample.arm] += sample.normal_force_n
                shear[sample.arm] += sample.shear_force_n
        arms = set(normal) | set(shear)
        per_arm_slip = {
            arm: _clamp(shear[arm] / max(request.friction_coefficient * normal[arm], 1e-6))
            for arm in arms
        }
        slip = max(per_arm_slip.values(), default=0.0)
        if self.capability == "contact":
            per_arm_contact = {
                arm: any(sample.arm == arm and sample.contact for sample in request.samples)
                for arm in ("left", "right")
            }
            return ContactEstimate(any(per_arm_contact.values()), per_arm_contact)
        if self.capability == "slip":
            return SlipEstimate(slip, per_arm_slip)
        if self.capability == "force":
            return ForceEstimate(sum(normal.values()), dict(normal))
        contact_expected = bool(request.samples)
        score = 1.0 - slip if contact_expected else 1.0
        return GraspStability(
            not contact_expected or score >= self.min_stability_score,
            score,
        )


class StateSpaceWorldModel:
    """Fast structured 2-3 second predictor for model-predictive selection."""

    def __init__(self, model_kind: str, *, per_arm_payload_kg: float = 0.8) -> None:
        supported = {
            "forward-dynamics",
            "rigid-dynamics",
            "grasp-contact",
            "bimanual-coordination",
            "deformable-dynamics",
            "human-motion",
        }
        if model_kind not in supported:
            raise ValueError(f"Unsupported world-model kind: {model_kind}")
        self.model_kind = model_kind
        if per_arm_payload_kg <= 0 or not math.isfinite(per_arm_payload_kg):
            raise ValueError("per_arm_payload_kg must be finite and positive")
        self.per_arm_payload_kg = per_arm_payload_kg

    @classmethod
    def from_robot_model(cls, model: RobotModel, model_kind: str) -> StateSpaceWorldModel:
        model.require_motion_ready()
        return cls(model_kind, per_arm_payload_kg=model.payload_limit_kg)

    def run(self, request: WorldModelInput) -> WorldModelPrediction:
        if not isinstance(request, WorldModelInput):
            raise TypeError("World model expects WorldModelInput")
        objects = {item.id: item for item in request.world.objects}
        target_ids = {
            chunk.target_object_id
            for chunk in request.policy.chunks
            if chunk.target_object_id is not None
        }
        arms = {arm for chunk in request.policy.chunks for arm in chunk.arms}
        total_force = sum(chunk.force_limit_n * len(chunk.arms) for chunk in request.policy.chunks)
        mean_force = total_force / max(1, len(request.policy.chunks) * max(1, len(arms)))
        base_slip = 0.0
        load_ratio = 0.0
        for chunk in request.policy.chunks:
            if chunk.target_object_id not in objects:
                continue
            target = objects[chunk.target_object_id]
            mass = target.estimated_mass_kg or request.personal.learned_object_masses.get(
                target.label, 0.4
            )
            arm_count = len(chunk.arms)
            required_force = max(0.1, mass * 9.81 / arm_count)
            base_slip = max(
                base_slip,
                _clamp((required_force - chunk.force_limit_n * 0.6) / required_force),
            )
            load_ratio = max(
                load_ratio,
                mass / max(0.001, self.per_arm_payload_kg * arm_count),
            )
        if load_ratio > 1:
            base_slip = max(base_slip, _clamp((load_ratio - 1) * 0.9))
        if self.model_kind == "grasp-contact":
            base_slip = _clamp(base_slip * 1.15)
        coordination_error = 0.0
        if self.model_kind == "bimanual-coordination" and len(arms) == 2:
            durations = [chunk.duration_seconds for chunk in request.policy.chunks]
            coordination_error = min(0.25, abs(max(durations) - min(durations)) * 0.1)
        collision_probability = 0.4 if request.world.hazards else 0.01
        if self.model_kind == "human-motion" and request.world.workspace.get("people_present"):
            collision_probability = max(collision_probability, 0.12)
        states = []
        step_count = max(1, math.ceil(request.horizon_seconds / request.step_seconds))
        points = request.trajectory.points
        for index in range(1, step_count + 1):
            time_s = min(request.horizon_seconds, index * request.step_seconds)
            trajectory_point = min(points, key=lambda point: abs(point.time_s - time_s))
            phase = time_s / request.horizon_seconds
            poses = {}
            for item in objects.values():
                x, y, z = item.position_m
                if item.id in target_ids:
                    lift = 0.10 * math.sin(math.pi * min(1.0, phase))
                    if self.model_kind == "deformable-dynamics":
                        lift *= 0.75
                    z += lift
                poses[item.id] = (x, y, z, 0.0, 0.0, 0.0, 1.0)
            states.append(
                PredictedState(
                    round(time_s, 6),
                    poses,
                    dict(trajectory_point.joint_positions),
                    {arm: mean_force for arm in arms},
                    _clamp(base_slip + coordination_error * phase),
                    _clamp(collision_probability + coordination_error * phase),
                )
            )
        max_slip = max(item.slip_probability for item in states)
        max_collision = max(item.collision_probability for item in states)
        risks = []
        if max_slip > 0.35:
            risks.append(f"{self.model_kind}: predicted grasp slip")
        if load_ratio > 1:
            risks.append(f"{self.model_kind}: predicted arm overload")
        if max_collision > 0.20:
            risks.append(f"{self.model_kind}: predicted collision")
        if self.model_kind == "bimanual-coordination" and coordination_error > 0.10:
            risks.append("bimanual-coordination: arms may desynchronize")
        success = _clamp(1.0 - 0.65 * max_slip - 0.75 * max_collision)
        uncertainty = {
            "forward-dynamics": 0.10,
            "rigid-dynamics": 0.06,
            "grasp-contact": 0.12,
            "bimanual-coordination": 0.10,
            "deformable-dynamics": 0.20,
            "human-motion": 0.22,
        }.get(self.model_kind, 0.25)
        return WorldModelPrediction(
            request.policy.plan_id,
            self.model_kind,
            tuple(states),
            success,
            uncertainty,
            tuple(risks),
        )


class ModelBasedTaskReward:
    """Score predicted progress for model-predictive or RL policy selection."""

    def run(self, request: RewardInput) -> RewardScore:
        if not isinstance(request, RewardInput):
            raise TypeError("Reward model expects RewardInput")
        success = sum(item.success_probability for item in request.predictions) / len(
            request.predictions
        )
        uncertainty = max(item.uncertainty for item in request.predictions)
        efficiency = 1 / (1 + 0.08 * len(request.candidate.steps))
        clearance = _clamp(request.collision.minimum_clearance_m / 0.10)
        score = _clamp(
            0.60 * success + 0.15 * (1 - uncertainty) + 0.15 * efficiency + 0.10 * clearance
        )
        if not request.collision.safe:
            score = 0.0
        return RewardScore(
            request.candidate.id,
            score,
            {
                "predicted_success": success,
                "certainty": 1 - uncertainty,
                "efficiency": efficiency,
                "clearance": clearance,
            },
        )


class HardSafetyRiskModel:
    """Fuse learned-model risks with non-negotiable local safety checks."""

    def __init__(
        self,
        *,
        per_arm_payload_kg: float = 0.8,
        max_slip_probability: float = 0.35,
        max_collision_probability: float = 0.20,
        min_grasp_stability_score: float = 0.60,
    ) -> None:
        if (
            not isinstance(per_arm_payload_kg, (int, float))
            or not math.isfinite(per_arm_payload_kg)
            or per_arm_payload_kg <= 0
        ):
            raise ValueError("per_arm_payload_kg must be finite and positive")
        self.per_arm_payload_kg = per_arm_payload_kg
        for name, value in (
            ("max_slip_probability", max_slip_probability),
            ("max_collision_probability", max_collision_probability),
            ("min_grasp_stability_score", min_grasp_stability_score),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be in (0, 1]")
        self.max_slip_probability = float(max_slip_probability)
        self.max_collision_probability = float(max_collision_probability)
        self.min_grasp_stability_score = float(min_grasp_stability_score)

    @classmethod
    def from_robot_model(cls, model: RobotModel) -> HardSafetyRiskModel:
        model.require_motion_ready()
        return cls(
            per_arm_payload_kg=model.payload_limit_kg,
            max_slip_probability=model.max_slip_probability,
            max_collision_probability=model.max_collision_probability,
            min_grasp_stability_score=model.min_grasp_stability_score,
        )

    def run(self, request: SafetyInput) -> SafetyAssessment:
        if not isinstance(request, SafetyInput):
            raise TypeError("Safety model expects SafetyInput")
        predicted_slip = max(
            state.slip_probability
            for prediction in request.predictions
            for state in prediction.states
        )
        predicted_collision = max(
            state.collision_probability
            for prediction in request.predictions
            for state in prediction.states
        )
        reasons = []
        if not request.collision.safe:
            reasons.extend(request.collision.collisions)
        if (
            predicted_slip > self.max_slip_probability
            or request.slip.probability > self.max_slip_probability
        ):
            reasons.append("slip probability exceeds the local safety threshold")
        if predicted_collision > self.max_collision_probability:
            reasons.append("collision probability exceeds the local safety threshold")
        if (
            not request.stability.stable
            or request.stability.score < self.min_grasp_stability_score
        ):
            reasons.append("tactile grasp stability is below threshold")
        overload = False
        if (
            request.world is not None
            and request.policy is not None
            and request.personal is not None
        ):
            objects = {item.id: item for item in request.world.objects}
            for chunk in request.policy.chunks:
                target = objects.get(chunk.target_object_id)
                mass = (
                    None
                    if target is None
                    else target.estimated_mass_kg
                    or request.personal.learned_object_masses.get(target.label)
                )
                if (
                    mass is not None and mass > self.per_arm_payload_kg * len(chunk.arms)
                ):
                    overload = True
                    reasons.append(
                        f"{chunk.id}: object mass exceeds calibrated arm payload"
                    )
        if any(
            "overload" in risk.casefold()
            for item in request.predictions
            for risk in item.risks
        ):
            overload = True
            reasons.append("world model predicts arm overload")
        risk = max(
            predicted_slip,
            predicted_collision,
            request.slip.probability,
            1 - request.stability.score,
            0.0 if request.collision.safe else 1.0,
            1.0 if overload else 0.0,
        )
        return SafetyAssessment(
            request.candidate.id,
            not reasons,
            _clamp(risk),
            tuple(dict.fromkeys(reasons)),
        )


class TelemetryOutcomeVerifier:
    def __init__(self, *, placement_tolerance_m: float = 0.05) -> None:
        if (
            not isinstance(placement_tolerance_m, (int, float))
            or isinstance(placement_tolerance_m, bool)
            or not math.isfinite(placement_tolerance_m)
            or placement_tolerance_m <= 0
        ):
            raise ValueError("placement_tolerance_m must be finite and positive")
        self.placement_tolerance_m = float(placement_tolerance_m)

    @staticmethod
    def _height(item: object) -> float | None:
        attributes = getattr(item, "attributes", None) or {}
        value = attributes.get("height_m")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            return None
        return float(value)

    def run(self, request: OutcomeInput) -> OutcomeReport:
        if not isinstance(request, OutcomeInput):
            raise TypeError("Outcome verifier expects OutcomeInput")
        if not request.control.executed:
            return OutcomeReport(request.plan.candidate.id, "planned", 1.0, {"executed": False})
        status = "succeeded" if request.control.success else "failed"
        successful = sum(item.success for item in request.control.telemetry)
        total = len(request.control.telemetry)
        camera_verified = False
        placement_error_m: float | None = None
        verification_issue: str | None = None
        if request.control.success:
            if request.world_after is None:
                status = "uncertain"
                verification_issue = "post-action camera observation is missing"
            else:
                before = {item.id: item for item in request.world_before.objects}
                after = {item.id: item for item in request.world_after.objects}
                placement_step = next(
                    (
                        step
                        for step in reversed(request.plan.candidate.steps)
                        if step.action.casefold() in ("release", "place")
                        and step.target_object_id is not None
                        and isinstance((step.parameters or {}).get("destination_object_id"), str)
                    ),
                    None,
                )
                if placement_step is None:
                    camera_verified = True
                else:
                    destination_id = str(placement_step.parameters["destination_object_id"])
                    target_id = str(placement_step.target_object_id)
                    target_after = after.get(target_id)
                    destination_after = after.get(destination_id)
                    target_before = before.get(target_id)
                    if target_after is None or destination_after is None or target_before is None:
                        status = "failed"
                        verification_issue = "camera did not find the target and destination"
                    else:
                        target_height = self._height(target_before)
                        destination_height = self._height(destination_after)
                        if target_height is None or destination_height is None:
                            status = "uncertain"
                            verification_issue = "camera placement check lacks object dimensions"
                        else:
                            up_index = {"x": 0, "y": 1, "z": 2}[request.world_after.up_axis]
                            expected = list(destination_after.position_m)
                            expected[up_index] += (target_height + destination_height) / 2
                            placement_error_m = math.dist(target_after.position_m, expected)
                            camera_verified = placement_error_m <= self.placement_tolerance_m
                            if not camera_verified:
                                status = "failed"
                                verification_issue = (
                                    "camera placement error exceeds configured tolerance"
                                )
        confidence = (
            0.98
            if status == "succeeded" and camera_verified
            else 0.9
            if status == "succeeded"
            else 0.5
            if status == "uncertain"
            else min(0.99, 0.65 + 0.05 * total)
        )
        return OutcomeReport(
            request.plan.candidate.id,
            status,
            confidence,
            {
                "successful_commands": successful,
                "total_commands": total,
                "issues": request.control.issues,
                "maximum_slip_probability": max(
                    (
                        item.slip_probability
                        for item in request.control.telemetry
                        if item.slip_probability is not None
                    ),
                    default=0.0,
                ),
                "maximum_normal_force_n": max(
                    (
                        item.normal_force_n
                        for item in request.control.telemetry
                        if item.normal_force_n is not None
                    ),
                    default=0.0,
                ),
                "camera_verified": camera_verified,
                "placement_error_m": placement_error_m,
                "placement_tolerance_m": self.placement_tolerance_m,
                "verification_issue": verification_issue,
                "objects_before": len(request.world_before.objects),
                "objects_after": (
                    len(request.world_after.objects) if request.world_after is not None else None
                ),
            },
        )


class TelemetryFailureClassifier:
    def __init__(self, *, max_slip_probability: float = 0.35) -> None:
        if (
            not isinstance(max_slip_probability, (int, float))
            or isinstance(max_slip_probability, bool)
            or not math.isfinite(max_slip_probability)
            or not 0 < max_slip_probability <= 1
        ):
            raise ValueError("max_slip_probability must be in (0, 1]")
        self.max_slip_probability = float(max_slip_probability)

    @classmethod
    def from_robot_model(cls, model: RobotModel) -> TelemetryFailureClassifier:
        model.require_motion_ready()
        return cls(max_slip_probability=model.max_slip_probability)

    def run(self, request: FailureClassificationInput) -> FailureReport:
        if not isinstance(request, FailureClassificationInput):
            raise TypeError("Failure classifier expects FailureClassificationInput")
        if request.outcome.status != "failed":
            return FailureReport(None, request.outcome.confidence, {})
        issues = " ".join(request.control.issues).lower()
        if request.slip.probability > self.max_slip_probability or "slip" in issues:
            return FailureReport("object_slipped", 0.94, {"replan_grasp": True})
        if "collision" in issues or "blocked" in issues:
            return FailureReport("collision_or_obstruction", 0.91, {"replan": True})
        if "load" in issues or "overload" in issues:
            return FailureReport("overload", 0.90, {"use_both_arms": True})
        if "stall" in issues:
            return FailureReport("actuator_stall", 0.88, {"reduce_speed": True})
        return FailureReport("execution_error", 0.70, {"inspect_telemetry": True})


class TelemetryLoadEstimator:
    def run(self, request: LoadEstimationInput) -> LoadEstimate:
        if not isinstance(request, LoadEstimationInput):
            raise TypeError("Load estimator expects LoadEstimationInput")
        by_step = defaultdict(list)
        for item in request.control.telemetry:
            if item.measured_mass_kg is not None:
                by_step[item.step_id].append(item.measured_mass_kg)
        objects = {item.id: item for item in request.world.objects}
        by_object: dict[str, list[float]] = defaultdict(list)
        for step_id, values in by_step.items():
            target_id = (
                next(
                    (
                        step.target_object_id
                        for step in request.plan.candidate.steps
                        if step.id == step_id
                    ),
                    None,
                )
                if request.plan is not None
                else None
            )
            # Telemetry lacks object IDs, so use the sole observed object when unambiguous.
            if target_id is None and len(objects) == 1:
                target_id = next(iter(objects))
            label = objects[target_id].label if target_id in objects else target_id or "object"
            by_object[label].extend(values)
        estimates = {
            label: sum(values) / len(values) for label, values in by_object.items() if values
        }
        confidence = 0.95 if estimates else 0.0
        return LoadEstimate(estimates, confidence)


class WorldModelErrorTracker:
    """Compare selected-model forecasts with observed task success for retraining."""

    def run(self, request: PredictionErrorInput) -> PredictionErrorReport:
        if not isinstance(request, PredictionErrorInput):
            raise TypeError("Prediction-error tracker expects PredictionErrorInput")
        if request.outcome.status not in ("succeeded", "failed"):
            return PredictionErrorReport(None, {}, 0.0, 0)
        observed = 1.0 if request.outcome.status == "succeeded" else 0.0
        errors = {
            item.model_kind: abs(item.success_probability - observed)
            for item in request.predictions
        }
        mean = sum(errors.values()) / len(errors)
        return PredictionErrorReport(observed, errors, mean, len(errors))
