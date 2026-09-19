from threading import Event
from time import time

import pytest

from moira.cloud import JsonHttpClient
from moira.contracts import decode_physical_response
from moira.physical import (
    ActionChunk,
    ArmTelemetry,
    BimanualControlComponent,
    CameraFrame,
    CandidatePlan,
    CollisionCheckInput,
    ControlInput,
    ControlReport,
    DetectedObject,
    FinalPlan,
    GraspPlanningInput,
    GraspStability,
    KinematicsInput,
    KinematicsSolution,
    MotionTrajectory,
    OutcomeReport,
    PersonalContext,
    PlanStep,
    PolicyInput,
    PolicyPlan,
    PredictionErrorInput,
    RewardInput,
    RobotState,
    SafetyInput,
    SimulationOutcome,
    SlipEstimate,
    TactileInput,
    TactileSample,
    TrajectoryPlanningInput,
    TrajectoryPoint,
    WorldModelInput,
    WorldState,
)
from moira.specialists import (
    AnalyticGraspPlanner,
    BoundedTrajectoryPlanner,
    ConservativeCollisionChecker,
    HardSafetyRiskModel,
    ModelBasedTaskReward,
    PlanarBimanualIK,
    SpecializedManipulationPolicy,
    StateSpaceWorldModel,
    TactileSignalModel,
    WorldModelErrorTracker,
)


def _physical_inputs(arms=("left", "right")):
    world = WorldState(
        (
            DetectedObject(
                "mug-1",
                "mug",
                0.99,
                (0.4, 0.1, 0.75),
                1.2,
                {"width_m": 0.07, "height_m": 0.10},
            ),
        ),
        {"minimum_clearance_m": 0.08},
    )
    personal = PersonalContext("sam", {}, (), (), {})
    step = PlanStep(
        "grasp",
        "grasp",
        arms,
        0.8,
        "mug-1",
        {"target_position_m": (0.4, 0.1, 0.75)},
    )
    candidate = CandidatePlan("candidate", (step,), "grasp the mug")
    grasps = AnalyticGraspPlanner().run(
        GraspPlanningInput(world, ("mug-1",), {"max_width_m": 0.08})
    )
    policy = SpecializedManipulationPolicy("test-policy").run(
        PolicyInput(candidate, world, personal, grasps)
    )
    ik = PlanarBimanualIK().run(KinematicsInput(policy, world))
    trajectory = BoundedTrajectoryPlanner().run(TrajectoryPlanningInput(policy, ik))
    collision = ConservativeCollisionChecker().run(CollisionCheckInput(trajectory, world))
    return world, personal, candidate, policy, trajectory, collision


def test_structured_world_predictions_drive_reward_and_local_safety():
    world, personal, candidate, policy, trajectory, collision = _physical_inputs()
    world_input = WorldModelInput(policy, trajectory, world, personal, 2.5, 0.1)
    predictions = tuple(
        StateSpaceWorldModel(kind).run(world_input)
        for kind in ("forward-dynamics", "rigid-dynamics", "grasp-contact")
    )
    assert all(len(item.states) == 25 for item in predictions)
    assert all(item.states[-1].time_s == 2.5 for item in predictions)
    assert all("mug-1" in item.states[0].object_poses for item in predictions)

    reward = ModelBasedTaskReward().run(RewardInput(candidate, predictions, collision))
    safety = HardSafetyRiskModel().run(
        SafetyInput(
            candidate,
            predictions,
            collision,
            SlipEstimate(0.05, {"left": 0.05, "right": 0.04}),
            GraspStability(True, 0.95),
        )
    )
    assert reward.score > 0.7
    assert safety.safe


def test_world_model_rejects_single_arm_overload_before_control():
    world, personal, candidate, policy, trajectory, collision = _physical_inputs(("left",))
    prediction = StateSpaceWorldModel("grasp-contact").run(
        WorldModelInput(policy, trajectory, world, personal, 2.5, 0.1)
    )
    safety = HardSafetyRiskModel().run(
        SafetyInput(
            candidate,
            (prediction,),
            collision,
            SlipEstimate(0.0, {}),
            GraspStability(True, 1.0),
        )
    )
    assert not safety.safe
    assert any("slip" in reason for reason in safety.reasons)
    assert any("overload" in risk for risk in prediction.risks)


def test_tactile_heads_detect_force_and_slip_locally():
    request = TactileInput(
        (
            TactileSample("left", time(), 5.0, 4.0),
            TactileSample("right", time(), 6.0, 0.5),
        ),
        0.60,
    )
    slip = TactileSignalModel.for_capability("slip").run(request)
    force = TactileSignalModel.for_capability("force").run(request)
    stability = TactileSignalModel.for_capability("stability").run(request)
    assert slip.per_arm["left"] == 1.0
    assert force.total_normal_force_n == 11.0
    assert not stability.stable


def test_controller_refuses_unvalidated_driver_for_policy_action_chunks():
    class LegacyDriver:
        def __init__(self, arm):
            self.arm = arm
            self.stopped = False

        def execute(self, step, world):
            return ArmTelemetry(self.arm, step.id, True)

        def stop(self):
            self.stopped = True

    world, _, candidate, policy, trajectory, _ = _physical_inputs()
    simulation = SimulationOutcome(candidate.id, 0.9, True, 2.5)
    plan = FinalPlan(candidate, simulation, candidate.rationale)
    left, right = LegacyDriver("left"), LegacyDriver("right")
    report = BimanualControlComponent(left, right).run(
        ControlInput(plan, world, True, policy, trajectory, GraspStability(True, 1.0))
    )
    assert not report.success
    assert left.stopped and right.stopped
    assert any("validated action chunks" in issue for issue in report.issues)


def test_controller_stops_both_arms_on_live_tactile_slip():
    class SlipDriver:
        def __init__(self, arm):
            self.arm = arm
            self.stopped = False

        def execute(self, step, world):
            return ArmTelemetry(self.arm, step.id, True)

        def execute_chunk(self, chunk, trajectory, world):
            return ArmTelemetry(
                self.arm,
                chunk.step_id,
                True,
                1.2,
                normal_force_n=8.0,
                slip_probability=0.9 if self.arm == "left" else 0.05,
            )

        def stop(self):
            self.stopped = True

    world, _, candidate, policy, trajectory, _ = _physical_inputs()
    plan = FinalPlan(candidate, SimulationOutcome(candidate.id, 0.9, True, 2.5), "test")
    left, right = SlipDriver("left"), SlipDriver("right")
    report = BimanualControlComponent(left, right).run(
        ControlInput(plan, world, True, policy, trajectory, GraspStability(True, 1.0))
    )
    assert not report.success
    assert left.stopped and right.stopped
    assert report.issues == ("candidate-chunk-0: tactile reflex detected slip",)


def test_remote_contract_decodes_grasps_policy_and_structured_world_states():
    world, personal, candidate, policy, trajectory, _ = _physical_inputs()
    grasp_request = GraspPlanningInput(world, ("mug-1",), {"max_width_m": 0.08})
    grasps = decode_physical_response(
        {
            "plans": [
                {
                    "target_object_id": "mug-1",
                    "grasps": [
                        {
                            "id": "top",
                            "position_m": [0.4, 0.1, 0.8],
                            "orientation_xyzw": [0, 0, 0, 1],
                            "width_m": 0.07,
                            "score": 0.9,
                            "collision_probability": 0.02,
                        }
                    ],
                }
            ]
        },
        grasp_request,
    )
    assert grasps[0].grasps[0].id == "top"

    decoded_policy = decode_physical_response(
        {
            "plan_id": "candidate",
            "policy": "bimanual-act",
            "chunks": [
                {
                    "id": "chunk",
                    "step_id": "grasp",
                    "skill": "grasp",
                    "arms": ["left", "right"],
                    "duration_seconds": 0.8,
                    "target_object_id": "mug-1",
                    "target_pose": [0.4, 0.1, 0.8, 0, 0, 0, 1],
                    "gripper_width_m": 0.07,
                    "force_limit_n": 8.0,
                }
            ],
        },
        PolicyInput(candidate, world, personal, grasps),
    )
    assert decoded_policy.policy == "bimanual-act"

    decoded_world = decode_physical_response(
        {
            "plan_id": "candidate",
            "model_kind": "grasp-contact",
            "states": [
                {
                    "time_s": 0.1,
                    "object_poses": {"mug-1": [0.4, 0.1, 0.8, 0, 0, 0, 1]},
                    "joint_positions": {"left": [0, 0, 0, 0]},
                    "contact_forces_n": {"left": 8.0},
                    "slip_probability": 0.02,
                    "collision_probability": 0.01,
                }
            ],
            "success_probability": 0.95,
            "uncertainty": 0.1,
        },
        WorldModelInput(policy, trajectory, world, personal, 2.5, 0.1),
    )
    assert decoded_world.states[0].object_poses["mug-1"][2] == 0.8


def test_motion_trajectory_contract_rejects_non_monotonic_time():
    point = TrajectoryPoint(0.1, {"left": (0.0, 0.0)})
    try:
        MotionTrajectory("plan", (point, point), 0.2)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("Duplicate trajectory time was accepted")


def test_world_model_prediction_error_is_measured_against_observed_outcome():
    world, personal, candidate, policy, trajectory, _ = _physical_inputs()
    prediction = StateSpaceWorldModel("grasp-contact").run(
        WorldModelInput(policy, trajectory, world, personal, 2.5, 0.1)
    )
    outcome = OutcomeReport(candidate.id, "failed", 0.95, {})
    control = ControlReport(candidate.id, True, False, (), ("object slipped",))
    report = WorldModelErrorTracker().run(PredictionErrorInput((prediction,), outcome, control))
    assert report.sample_count == 1
    assert report.observed_success == 0.0
    assert report.mean_absolute_error == prediction.success_probability


def test_camera_frame_remains_compatible_with_structured_inputs():
    frame = CameraFrame("csi0", b"jpeg", time())
    assert frame.media_type == "image/jpeg"


def test_policy_must_match_every_step_arm_and_target_before_motion():
    world, _, candidate, policy, trajectory, _ = _physical_inputs()
    chunk = policy.chunks[0]
    wrong_arm = ActionChunk(
        chunk.id,
        chunk.step_id,
        chunk.skill,
        ("left",),
        chunk.duration_seconds,
        chunk.target_object_id,
        chunk.target_pose,
        chunk.gripper_width_m,
        chunk.force_limit_n,
    )
    invalid = PolicyPlan(candidate.id, policy.policy, (wrong_arm,))
    plan = FinalPlan(candidate, SimulationOutcome(candidate.id, 0.9, True, 2.5), "test")
    with pytest.raises(ValueError, match="changes the commanded arms"):
        ControlInput(plan, world, True, invalid, trajectory)


def test_infeasible_or_incomplete_kinematics_never_reaches_trajectory_planning():
    _, _, _, policy, _, _ = _physical_inputs()
    infeasible = KinematicsSolution(policy.plan_id, {}, False, ("outside workspace",))
    with pytest.raises(ValueError, match="infeasible kinematics"):
        TrajectoryPlanningInput(policy, infeasible)

    incomplete = KinematicsSolution(
        policy.plan_id,
        {policy.chunks[0].id: {"left": (0.0, 0.0, 0.0, 0.0)}},
        True,
    )
    with pytest.raises(ValueError, match="wrong arms"):
        TrajectoryPlanningInput(policy, incomplete)


def test_controller_sends_only_each_action_chunks_trajectory_to_driver():
    world = WorldState(
        (DetectedObject("mug-1", "mug", 1.0, (0.3, 0.0, 0.4)),),
        {},
    )
    steps = tuple(
        PlanStep(name, name, ("left",), 0.5, "mug-1") for name in ("grasp", "place")
    )
    candidate = CandidatePlan("candidate", steps, "move mug")
    chunks = tuple(
        ActionChunk(
            f"{step.id}-chunk",
            step.id,
            step.action,
            step.arms,
            step.duration_seconds,
            step.target_object_id,
            (0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
            0.05,
            5.0,
        )
        for step in steps
    )
    policy = PolicyPlan(candidate.id, "test", chunks)
    kinematics = KinematicsSolution(
        candidate.id,
        {chunk.id: {"left": (index * 0.2, 0.1)} for index, chunk in enumerate(chunks, 1)},
        True,
    )
    trajectory = BoundedTrajectoryPlanner(frequency_hz=4).run(
        TrajectoryPlanningInput(policy, kinematics)
    )

    class RecordingDriver:
        def __init__(self, arm):
            self.arm = arm
            self.received = []

        def execute(self, step, world):
            raise AssertionError("legacy execution should not run")

        def execute_chunk(self, chunk, segment, world):
            self.received.append(
                (
                    chunk.id,
                    segment.chunk_ids,
                    segment.duration_seconds,
                    segment.points[-1].gripper_widths_m[self.arm],
                )
            )
            return ArmTelemetry(self.arm, chunk.step_id, True)

        def stop(self):
            pass

    left, right = RecordingDriver("left"), RecordingDriver("right")
    plan = FinalPlan(candidate, SimulationOutcome(candidate.id, 0.9, True, 2.5), "test")
    report = BimanualControlComponent(left, right).run(
        ControlInput(plan, world, True, policy, trajectory)
    )
    assert report.success
    assert left.received == [
        ("grasp-chunk", ("grasp-chunk",), 0.5, 0.05),
        ("place-chunk", ("place-chunk",), 0.5, 0.05),
    ]


def test_controller_rejects_telemetry_for_the_wrong_arm_or_step():
    class WrongIdentityDriver:
        def __init__(self, arm):
            self.arm = arm
            self.stopped = False

        def execute(self, step, world):
            return ArmTelemetry("right", "unrelated-step", True)

        def stop(self):
            self.stopped = True

    world, _, candidate, _, _, _ = _physical_inputs(("left",))
    plan = FinalPlan(candidate, SimulationOutcome(candidate.id, 0.9, True, 2.5), "test")
    left, right = WrongIdentityDriver("left"), WrongIdentityDriver("right")
    report = BimanualControlComponent(left, right).run(ControlInput(plan, world, True))
    assert not report.success
    assert left.stopped and right.stopped
    assert "telemetry identity" in report.issues[0]


def test_controller_deadline_triggers_both_emergency_stops():
    released = Event()

    class BlockingDriver:
        def __init__(self, arm):
            self.arm = arm
            self.stopped = False

        def execute(self, step, world):
            released.wait(timeout=1)
            return ArmTelemetry(self.arm, step.id, True)

        def stop(self):
            self.stopped = True
            released.set()

    world, _, candidate, _, _, _ = _physical_inputs()
    plan = FinalPlan(candidate, SimulationOutcome(candidate.id, 0.9, True, 2.5), "test")
    left, right = BlockingDriver("left"), BlockingDriver("right")
    report = BimanualControlComponent(
        left,
        right,
        timeout_margin_seconds=0.0,
    ).run(ControlInput(plan, world, True))
    assert not report.success
    assert left.stopped and right.stopped
    assert report.issues == ("grasp: arm command timed out",)


def test_generic_trajectory_clamps_both_samples_and_next_chunk_start():
    world = WorldState((DetectedObject("x", "x", 1.0, (0.0, 0.0, 0.0)),), {})
    del world
    steps = (
        PlanStep("one", "move", ("left",), 1.0, "x"),
        PlanStep("two", "move", ("left",), 1.0, "x"),
    )
    candidate = CandidatePlan("candidate", steps, "bounded")
    chunks = tuple(
        ActionChunk(
            step.id,
            step.id,
            "move",
            step.arms,
            1.0,
            "x",
            (0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0),
            0.01,
            1.0,
        )
        for step in steps
    )
    policy = PolicyPlan(candidate.id, "bounded", chunks)
    kinematics = KinematicsSolution(
        candidate.id,
        {"one": {"left": (2.0,)}, "two": {"left": (-2.0,)}},
        True,
    )
    trajectory = BoundedTrajectoryPlanner(frequency_hz=10, joint_limit_rad=1.0).run(
        TrajectoryPlanningInput(policy, kinematics)
    )
    assert all(abs(point.joint_positions["left"][0]) <= 1.0 for point in trajectory.points)


def test_trajectory_rejects_motion_faster_than_configured_joint_velocity():
    step = PlanStep("fast", "move", ("left",), 0.1, "x")
    candidate = CandidatePlan("candidate", (step,), "too fast")
    chunk = ActionChunk(
        "fast",
        "fast",
        "move",
        ("left",),
        0.1,
        "x",
        (0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0),
        0.01,
        1.0,
    )
    policy = PolicyPlan(candidate.id, "test", (chunk,))
    kinematics = KinematicsSolution(candidate.id, {chunk.id: {"left": (1.0,)}}, True)
    with pytest.raises(ValueError, match="velocity limits"):
        BoundedTrajectoryPlanner(max_joint_velocity_rad_s=1.0).run(
            TrajectoryPlanningInput(policy, kinematics)
        )


def test_trajectory_rejects_unbounded_gripper_motion():
    step = PlanStep("grip", "grasp", ("left",), 0.1, "x")
    candidate = CandidatePlan("candidate", (step,), "close gripper")
    chunk = ActionChunk(
        "grip",
        "grip",
        "grasp",
        ("left",),
        0.1,
        "x",
        (0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0),
        0.05,
        1.0,
    )
    policy = PolicyPlan(candidate.id, "test", (chunk,))
    kinematics = KinematicsSolution(candidate.id, {chunk.id: {"left": (0.0,)}}, True)
    state = RobotState({"left": (0.0,)}, {"left": 0.0}, time())
    with pytest.raises(ValueError, match="gripper velocity limits"):
        BoundedTrajectoryPlanner(max_gripper_velocity_m_s=0.1).run(
            TrajectoryPlanningInput(policy, kinematics, state)
        )


def test_trajectory_rejects_force_above_configured_gripper_limit():
    step = PlanStep("grip", "grasp", ("left",), 1.0, "x")
    candidate = CandidatePlan("candidate", (step,), "limit force")
    chunk = ActionChunk(
        "grip",
        "grip",
        "grasp",
        ("left",),
        1.0,
        "x",
        (0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0),
        0.01,
        3.1,
    )
    policy = PolicyPlan(candidate.id, "test", (chunk,))
    kinematics = KinematicsSolution(candidate.id, {chunk.id: {"left": (0.0,)}}, True)
    with pytest.raises(ValueError, match="gripper force limit"):
        BoundedTrajectoryPlanner(max_gripper_force_n=3.0).run(
            TrajectoryPlanningInput(policy, kinematics)
        )


def test_collision_check_fails_closed_without_measured_clearance():
    trajectory = MotionTrajectory(
        "candidate",
        (TrajectoryPoint(0.1, {"left": (0.0,)}, "chunk"),),
        0.1,
    )
    world = WorldState((), {})
    collision = ConservativeCollisionChecker().run(
        CollisionCheckInput(trajectory, world)
    )
    assert not collision.safe
    assert collision.minimum_clearance_m == 0.0
    assert any("did not report" in reason for reason in collision.collisions)


def test_payload_safety_is_evaluated_per_step_not_across_all_used_arms():
    world = WorldState(
        (DetectedObject("mug", "mug", 1.0, (0.3, 0.0, 0.4), 0.06),),
        {"minimum_clearance_m": 0.1},
    )
    steps = (
        PlanStep("single", "lift", ("left",), 1.0, "mug"),
        PlanStep("both", "place", ("left", "right"), 1.0, "mug"),
    )
    candidate = CandidatePlan("candidate", steps, "mixed arm use")
    chunks = tuple(
        ActionChunk(
            step.id,
            step.id,
            step.action,
            step.arms,
            step.duration_seconds,
            step.target_object_id,
            (0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
            0.05,
            8.0,
        )
        for step in steps
    )
    policy = PolicyPlan(candidate.id, "test", chunks)
    kinematics = KinematicsSolution(
        candidate.id,
        {
            "single": {"left": (0.1,)},
            "both": {"left": (0.1,), "right": (0.1,)},
        },
        True,
    )
    trajectory = BoundedTrajectoryPlanner().run(TrajectoryPlanningInput(policy, kinematics))
    collision = ConservativeCollisionChecker().run(CollisionCheckInput(trajectory, world))
    prediction = StateSpaceWorldModel("rigid-dynamics", per_arm_payload_kg=0.05).run(
        WorldModelInput(
            policy,
            trajectory,
            world,
            PersonalContext("sam", {}, (), (), {}),
            2.5,
            0.1,
        )
    )
    safety = HardSafetyRiskModel(per_arm_payload_kg=0.05).run(
        SafetyInput(
            candidate,
            (prediction,),
            collision,
            SlipEstimate(0.0, {}),
            GraspStability(True, 1.0),
            world,
            policy,
            PersonalContext("sam", {}, (), (), {}),
        )
    )
    assert any("overload" in risk for risk in prediction.risks)
    assert not safety.safe
    assert safety.risk_score == 1.0


def test_cloud_json_rejects_nan_before_network_io():
    with pytest.raises(ValueError, match="Out of range float"):
        JsonHttpClient(attempts=1).post(
            "https://example.invalid/model",
            {"unsafe": float("nan")},
            {},
        )
