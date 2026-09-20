from threading import Event
from time import time

import pytest

from moira.components import ComponentRegistry, ComponentSpec, Layer
from moira.human_interaction import (
    HumanAwarePhysicalSession,
    PlanProposal,
    SpokenEmergencyStopMonitor,
)
from moira.physical import (
    ArmTelemetry,
    CameraFrame,
    CandidatePlan,
    ClarificationResult,
    ConfirmedPlanChangedError,
    ControlReport,
    DetectedObject,
    EmergencyStopResult,
    FeedbackReport,
    FinalPlan,
    GroundedIntent,
    PersonalContext,
    PhysicalAI,
    PhysicalAIResult,
    PlanStep,
    RobotState,
    SceneChangedError,
    SimulationOutcome,
    TaskRequest,
    WorldState,
    is_emergency_stop_command,
    validate_pre_execution_scene,
)
from moira.session import PhysicalSession


def _world(*, red_x: float = 0.1, hazards: tuple[str, ...] = ()) -> WorldState:
    return WorldState(
        (
            DetectedObject("red-block-1", "red block", 0.99, (red_x, 0.02, 0.1)),
            DetectedObject("blue-tray-1", "blue tray", 0.99, (0.2, 0.01, 0.1)),
        ),
        {},
        hazards,
        time(),
    )


def _result(*, executed: bool, transcript: str = "Put the red block in the blue tray"):
    world = _world()
    intent = GroundedIntent(
        transcript,
        "pick_place",
        ("red-block-1", "blue-tray-1"),
        {"red-block-1": "manipulated", "blue-tray-1": "destination"},
    )
    candidate = CandidatePlan(
        "direct",
        (PlanStep("move", "place", ("left",), 0.5, "red-block-1"),),
        "direct transfer",
    )
    simulation = SimulationOutcome("direct", 0.9, True, 2.5)
    return PhysicalAIResult(
        intent,
        world,
        PersonalContext("demo", {}, (), (), {}),
        (candidate,),
        (simulation,),
        FinalPlan(candidate, simulation, "direct transfer"),
        ControlReport(
            "direct",
            executed,
            True,
            (ArmTelemetry("left", "move", True),) if executed else (),
        ),
        FeedbackReport((), ()),
        "ready",
        None,
        (),
    )


class _InteractionSystem:
    def __init__(self):
        self.stops = 0

    def emergency_stop(self):
        self.stops += 1
        return ()

    def synthesize_speech(self, text, user_id):
        return f"{user_id}:{text}".encode()

    def transcribe_audio(self, audio):
        assert audio == b"yes-audio"
        return "yes"


class _InteractionSession(PhysicalSession):
    def __init__(self, results):
        self.system = _InteractionSystem()
        self.results = iter(results)
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


def test_conversation_clarifies_then_requires_fresh_confirmation_before_execution():
    personal = PersonalContext("demo", {}, (), (), {})
    clarification = ClarificationResult(
        "Move the red block",
        "Where should I put the object?",
        _world(),
        personal,
        b"question",
        (),
    )
    preview = _result(executed=False)
    executed = _result(executed=True)
    session = _InteractionSession((clarification, preview, executed))
    conversation = HumanAwarePhysicalSession(
        session,
        user_id="demo",
        workspace={"human_error_policy": {"max_pre_execution_target_drift_m": 0.02}},
    )

    first = conversation.plan(instruction="Move the red block")
    assert isinstance(first, ClarificationResult)
    proposal = conversation.plan(instruction="Put it in the blue tray")
    assert isinstance(proposal, PlanProposal)
    assert not proposal.result.control.executed

    result = conversation.respond(
        confirmation_id=proposal.confirmation_id,
        audio=b"yes-audio",
        robot_state=RobotState({"left": (0.0,)}, observed_at=time()),
    )
    assert isinstance(result, PhysicalAIResult)
    assert result.control.executed
    assert session.calls[0]["execute"] is False
    assert session.calls[1]["execute"] is False
    assert session.calls[2]["execute"] is True
    assert session.calls[2]["execution_confirmed"] is True
    assert session.calls[2]["confirmed_intent"] == proposal.result.intent
    assert session.calls[2]["confirmed_candidate"] == proposal.result.plan.candidate
    assert any(turn.text == "Where should I put the object?" for turn in conversation.dialogue)


def test_stale_confirmation_cannot_authorize_motion():
    session = _InteractionSession((_result(executed=False),))
    conversation = HumanAwarePhysicalSession(session, user_id="demo", workspace={})
    proposal = conversation.plan(instruction="Put the red block in the blue tray", speak=False)
    assert isinstance(proposal, PlanProposal)
    with pytest.raises(RuntimeError, match="stale plan"):
        conversation.respond(
            confirmation_id="0000000000000000",
            instruction="yes",
            robot_state=RobotState({"left": (0.0,)}, observed_at=time()),
        )
    assert len(session.calls) == 1
    assert conversation.pending is None


def test_changed_plan_returns_to_confirmation_without_executing():
    preview = _result(executed=False)
    refreshed = _result(executed=False, transcript="Put the red block in the blue tray")
    session = _InteractionSession(
        (
            preview,
            ConfirmedPlanChangedError("changed"),
            refreshed,
        )
    )
    conversation = HumanAwarePhysicalSession(session, user_id="demo", workspace={})
    proposal = conversation.plan(instruction="Put the red block in the blue tray", speak=False)
    assert isinstance(proposal, PlanProposal)

    replacement = conversation.respond(
        confirmation_id=proposal.confirmation_id,
        instruction="yes",
        robot_state=RobotState({"left": (0.0,)}, observed_at=time()),
        speak=False,
    )

    assert isinstance(replacement, PlanProposal)
    assert replacement.result.control.executed is False
    assert session.calls[1]["execute"] is True
    assert session.calls[2]["execute"] is False


def test_local_stop_command_bypasses_perception_and_planning():
    events = []
    system = PhysicalAI(
        ComponentRegistry(ram_budget_mb=16),
        robot_model_id="robot",
        gripper_geometry={"max_width_m": 0.05},
        event_sink=events.append,
    )
    result = system.run(
        TaskRequest(
            "demo",
            (CameraFrame("camera", b"jpeg", time()),),
            instruction="Stop now please",
            speak=False,
        )
    )
    assert isinstance(result, EmergencyStopResult)
    assert any(event.kind == "safety.emergency_stop" for event in events)
    assert not any(event.kind == "scene.perceived" for event in events)


@pytest.mark.parametrize(
    "phrase",
    ("please stop", "wait", "could you pause please", "never mind", "hold on"),
)
def test_common_stop_phrases_are_recognized_locally(phrase):
    assert is_emergency_stop_command(phrase)


def test_spoken_stop_monitor_interrupts_active_execution():
    stopped = Event()
    monitor = SpokenEmergencyStopMonitor(
        lambda: b"wav",
        lambda audio: "Please stop now" if audio == b"wav" else "",
        stopped.set,
    )

    monitor.start()
    assert stopped.wait(1)
    monitor.close()

    assert monitor.triggered
    assert monitor.transcript == "Please stop now"
    assert monitor.failure is None


def test_spoken_stop_monitor_fails_closed_if_microphone_breaks():
    stopped = Event()

    def broken_capture():
        raise RuntimeError("microphone disconnected")

    monitor = SpokenEmergencyStopMonitor(
        broken_capture,
        lambda audio: "",
        stopped.set,
    )

    monitor.start()
    assert stopped.wait(1)
    monitor.close()

    assert monitor.triggered
    assert isinstance(monitor.failure, RuntimeError)


def test_registry_emergency_stop_interrupts_loaded_hardware_component():
    class Stoppable:
        def __init__(self):
            self.stopped = False

        def run(self, request):
            return request

        def stop(self):
            self.stopped = True

    component = Stoppable()
    registry = ComponentRegistry(ram_budget_mb=16)
    registry.register(
        ComponentSpec(
            "hardware",
            Layer.CONTROL,
            ("control.single_arm",),
            "hardware-v1",
            runtime="hardware",
        ),
        lambda: component,
    )
    decision = registry.decide(Layer.CONTROL, "control.single_arm")
    assert registry.invoke(decision, "load") == "load"
    assert registry.emergency_stop() == ()
    assert component.stopped


def test_scene_revalidation_blocks_moved_target_and_new_hazard():
    stable = validate_pre_execution_scene(
        _world(),
        _world(red_x=0.11),
        ("red-block-1", "blue-tray-1"),
        0.02,
    )
    assert stable.target_drift_m["red-block-1"] == pytest.approx(0.01)
    with pytest.raises(SceneChangedError, match="moved targets"):
        validate_pre_execution_scene(
            _world(),
            _world(red_x=0.14),
            ("red-block-1", "blue-tray-1"),
            0.02,
        )
    with pytest.raises(SceneChangedError, match="new hazards"):
        validate_pre_execution_scene(
            _world(),
            _world(hazards=("person-1",)),
            ("red-block-1", "blue-tray-1"),
            0.02,
        )
    hazard_only = validate_pre_execution_scene(_world(), _world(), (), 0.02)
    assert hazard_only.target_drift_m == {}
