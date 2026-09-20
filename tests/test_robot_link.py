from dataclasses import asdict, replace

import pytest

from moira.physical import (
    ActionChunk,
    CandidatePlan,
    ControlInput,
    FinalPlan,
    GraspStability,
    MotionTrajectory,
    PlanStep,
    PolicyPlan,
    SimulationOutcome,
    TrajectoryPoint,
    WorldState,
)
from moira.robot_config import load_bundled_robot_model
from moira.robot_link import (
    PiRobotControlComponent,
    PiRobotControllerApplication,
    decode_control_command,
)


def control_input():
    step = PlanStep("step-1", "pick", ("left",), 1.0, "block")
    candidate = CandidatePlan("plan-1", (step,), "pick the block")
    simulation = SimulationOutcome("plan-1", 0.9, True, 2.5)
    chunk = ActionChunk(
        "chunk-1",
        "step-1",
        "waypoint",
        ("left",),
        1.0,
        "block",
        (0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0),
        0.02,
        1.0,
    )
    trajectory = MotionTrajectory(
        "plan-1",
        (TrajectoryPoint(1.0, {"left": (0.1, 0.2, 0.3)}, "chunk-1", {"left": 0.02}),),
        1.0,
    )
    return ControlInput(
        FinalPlan(candidate, simulation, "safe pick"),
        WorldState((), {}, observed_at=1.0),
        True,
        PolicyPlan("plan-1", "waypoint", (chunk,)),
        trajectory,
        GraspStability(True, 0.9),
    )


class RobotHttp:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def post(self, url, payload, headers):
        self.calls.append((url, payload, headers))
        if url.endswith("/v1/stop"):
            return {"schema_version": 1, "request_id": payload["request_id"], "issues": []}
        if self.fail:
            raise ConnectionError("robot link lost")
        return {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "robot_model_id": payload["robot_model_id"],
            "control": {
                "plan_id": payload["plan_id"],
                "executed": True,
                "success": True,
                "telemetry": [
                    {"arm": "left", "step_id": payload["step"]["id"], "success": True}
                ],
                "issues": [],
            },
        }


def test_laptop_sends_bounded_chunk_that_pi_revalidates():
    model = load_bundled_robot_model()
    http = RobotHttp()
    component = PiRobotControlComponent(
        "http://172.20.10.6:8770", "token", model, http=http
    )
    report = component.run(control_input())

    assert report.executed and report.success
    assert len(report.telemetry) == 1
    url, payload, headers = http.calls[0]
    assert url.endswith("/v1/control")
    assert headers == {"Authorization": "Bearer token"}
    _, decoded = decode_control_command(payload, model)
    assert decoded.policy.chunks[0].id == "chunk-1"
    assert decoded.trajectory.chunk_ids == ("chunk-1",)


def test_laptop_requests_emergency_stop_if_robot_link_fails():
    model = load_bundled_robot_model()
    http = RobotHttp(fail=True)
    report = PiRobotControlComponent(
        "http://172.20.10.6:8770", "token", model, http=http
    ).run(control_input())

    assert report.executed and not report.success
    assert "robot link lost" in report.issues
    assert http.calls[-1][0].endswith("/v1/stop")


def test_laptop_rechecks_scene_guard_before_sending_motion():
    model = load_bundled_robot_model()
    http = RobotHttp()

    def reject_moved_scene(step, completed_actions):
        assert step.id == "step-1"
        assert completed_actions == ()
        raise RuntimeError("scene changed before execution")

    request = replace(control_input(), execution_guard=reject_moved_scene)
    report = PiRobotControlComponent(
        "http://172.20.10.6:8770", "token", model, http=http
    ).run(request)

    assert report.executed and not report.success
    assert report.issues == ("scene changed before execution",)
    assert [call[0].rsplit("/", 1)[-1] for call in http.calls] == ["stop"]


def test_pi_service_starts_locked_without_opening_i2c():
    model = load_bundled_robot_model()
    application = PiRobotControllerApplication(model)

    assert application.health()["status"] == "locked"
    assert not application.health()["motion_enabled"]
    with pytest.raises(PermissionError, match="disabled"):
        application.control(
            {
                "schema_version": 1,
                "request_id": "0" * 32,
                "robot_model_id": model.model_id,
                "robot_source_sha256": model.source_sha256,
                "control": asdict(control_input()),
            }
        )
