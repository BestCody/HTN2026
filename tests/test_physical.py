from pathlib import Path
from threading import Barrier
from time import time

import pytest

from moira.components import ComponentRegistry, ComponentSpec, Layer
from moira.edge_components import (
    SceneGroundedVoiceNLP,
    SQLitePersonalMemory,
    StructuredFramePerception,
    Utf8SpeechSynthesisFixture,
)
from moira.edge_demo import run_edge_demo
from moira.physical import (
    ArmTelemetry,
    BimanualControlComponent,
    CameraFrame,
    CandidatePlan,
    ClarificationResult,
    ControlInput,
    DetectedObject,
    DialogueTurn,
    FinalPlan,
    MemoryQuery,
    PersonalContext,
    PhysicalAI,
    PlanStep,
    RobotState,
    SimulationOutcome,
    TaskRequest,
    VoiceGroundingInput,
    WorldState,
    policy_routing_text,
)
from moira.pi import PiRuntimeProfile


def test_complete_edge_pipeline_uses_both_arms_and_learns_object_mass(tmp_path):
    result = run_edge_demo(tmp_path / "memory.db")
    assert result.selected_plan == "bimanual"
    assert result.executed and result.success
    assert result.camera_verified
    assert "camera-object-detector" in result.routed_components
    assert "rigid-world" in result.routed_components
    assert "bimanual-world" in result.routed_components
    assert result.world_models == (
        "forward-dynamics",
        "rigid-dynamics",
        "grasp-contact",
        "bimanual-coordination",
    )
    assert result.learned_facts == ("mug measured 0.040 kg",)
    assert "accessible right side" in result.response

    memory = SQLitePersonalMemory(tmp_path / "memory.db")
    context = memory.run(MemoryQuery("sam", "next task", {"room": "kitchen"}))
    assert context.accommodations == ("left arm is broken",)
    assert any("Bring me the mug" in comment for comment in context.recent_comments)
    assert context.learned_object_masses == {"mug": 0.04}


def test_bimanual_controller_starts_both_arm_commands_together():
    barrier = Barrier(2)

    class Driver:
        def __init__(self, arm):
            self.arm = arm
            self.stopped = False

        def execute(self, step, world):
            barrier.wait(timeout=2)
            return ArmTelemetry(self.arm, step.id, True, 1.1)

        def stop(self):
            self.stopped = True

    step = PlanStep("lift", "lift", ("left", "right"), 0.5, "box")
    candidate = CandidatePlan("both", (step,), "lift together")
    simulation = SimulationOutcome("both", 0.9, True, 2.5)
    plan = FinalPlan(candidate, simulation, "lift together")
    world = WorldState((DetectedObject("box", "box", 1, (0, 0, 0), 1.1),), {})
    component = BimanualControlComponent(Driver("left"), Driver("right"))
    report = component.run(ControlInput(plan, world, True))
    assert report.success
    assert {item.arm for item in report.telemetry} == {"left", "right"}


def test_policy_routing_query_uses_grounded_plan_instead_of_a_route_table():
    candidate = CandidatePlan(
        "handover-plan",
        (PlanStep("offer", "handover", ("right",), 0.5, "tool-1"),),
        "present the requested tool to the user",
    )

    query = policy_routing_text("Please pass me the screwdriver", candidate)

    assert "Please pass me the screwdriver" in query
    assert "action=handover" in query
    assert "arms=right" in query
    assert "target=tool-1" in query


def test_controller_runs_installed_primary_arm_and_blocks_unavailable_arm():
    class Driver:
        def __init__(self):
            self.stopped = False

        def execute(self, step, world):
            return ArmTelemetry("left", step.id, True)

        def stop(self):
            self.stopped = True

    driver = Driver()
    component = BimanualControlComponent(driver)
    world = WorldState((DetectedObject("box", "box", 1, (0, 0, 0), 0.01),), {})

    left_step = PlanStep("left-step", "lift", ("left",), 0.1, "box")
    left_candidate = CandidatePlan("left-plan", (left_step,), "single arm")
    left_plan = FinalPlan(
        left_candidate,
        SimulationOutcome("left-plan", 0.9, True, 2.5),
        "single arm",
    )
    report = component.run(ControlInput(left_plan, world, True))
    assert report.success
    assert tuple(item.arm for item in report.telemetry) == ("left",)

    right_step = PlanStep("right-step", "lift", ("right",), 0.1, "box")
    right_candidate = CandidatePlan("right-plan", (right_step,), "missing arm")
    right_plan = FinalPlan(
        right_candidate,
        SimulationOutcome("right-plan", 0.9, True, 2.5),
        "missing arm",
    )
    blocked = component.run(ControlInput(right_plan, world, True))
    assert not blocked.executed
    assert not blocked.success
    assert blocked.issues == ("Plan requires unavailable arm control slots: right",)


def test_controller_emergency_stop_remains_latched_before_execution():
    class Driver:
        def __init__(self):
            self.calls = 0
            self.stopped = False

        def execute(self, step, world):
            self.calls += 1
            return ArmTelemetry("left", step.id, True)

        def stop(self):
            self.stopped = True

    driver = Driver()
    component = BimanualControlComponent(driver)
    assert component.emergency_stop() == ()
    candidate = CandidatePlan(
        "left-plan",
        (PlanStep("left-step", "lift", ("left",), 0.1, "box"),),
        "single arm",
    )
    plan = FinalPlan(
        candidate,
        SimulationOutcome("left-plan", 0.9, True, 2.5),
        "single arm",
    )
    world = WorldState((DetectedObject("box", "box", 1, (0, 0, 0), 0.01),), {})

    blocked = component.run(ControlInput(plan, world, True))

    assert driver.stopped
    assert driver.calls == 0
    assert not blocked.executed
    assert "stop latch" in blocked.issues[0].casefold()


def test_controller_scene_guard_stops_before_the_next_step():
    class Driver:
        def __init__(self):
            self.calls = []
            self.stopped = False

        def execute(self, step, world):
            self.calls.append(step.id)
            return ArmTelemetry("left", step.id, True)

        def stop(self):
            self.stopped = True

    driver = Driver()
    component = BimanualControlComponent(driver)
    candidate = CandidatePlan(
        "guarded",
        (
            PlanStep("approach", "approach", ("left",), 0.1, "box"),
            PlanStep("grasp", "grasp", ("left",), 0.1, "box"),
        ),
        "guarded motion",
    )
    plan = FinalPlan(
        candidate,
        SimulationOutcome("guarded", 0.9, True, 2.5),
        "guarded motion",
    )
    world = WorldState((DetectedObject("box", "box", 1, (0, 0, 0), 0.01),), {})

    def guard(step, completed_actions):
        if completed_actions:
            raise RuntimeError("object moved during approach")

    report = component.run(ControlInput(plan, world, True, execution_guard=guard))

    assert driver.calls == ["approach"]
    assert driver.stopped
    assert report.executed and not report.success
    assert "object moved" in report.issues[-1]


def test_execution_request_requires_timestamped_measured_robot_state():
    frame = CameraFrame("camera", b"jpeg", time())
    with pytest.raises(ValueError, match="requires a robot_state"):
        TaskRequest("sam", (frame,), instruction="move", execute=True)
    with pytest.raises(ValueError, match="timestamped robot_state"):
        TaskRequest(
            "sam",
            (frame,),
            instruction="move",
            robot_state=RobotState({"left": (0.0,)}, observed_at=0.0),
            execute=True,
            execution_confirmed=True,
        )


def test_physical_demo_does_not_leave_sqlite_sidecars_locked(tmp_path):
    path = tmp_path / "memory.db"
    run_edge_demo(path)
    Path(path).unlink()
    assert not path.exists()


def test_dum_e_style_voice_nlp_resolves_dialogue_references_and_asks_when_ungrounded():
    world = WorldState((DetectedObject("mug-1", "mug", 1, (0.4, 0.1, 0.9)),), {})
    personal = PersonalContext("sam", {}, ("left arm is broken",), (), {})
    nlp = SceneGroundedVoiceNLP()
    grounded = nlp.run(
        VoiceGroundingInput(
            "bring that to me",
            world,
            personal,
            (DialogueTurn("user", "I mean the mug"),),
        )
    )
    assert grounded.action == "bring"
    assert grounded.target_object_ids == ("mug-1",)
    assert grounded.constraints == ("left arm is broken",)
    assert not grounded.needs_clarification

    unclear = nlp.run(VoiceGroundingInput("bring the wrench", world, personal, ()))
    assert unclear.needs_clarification
    assert unclear.clarification_question == "Which object do you mean?"


def test_ungrounded_voice_request_returns_spoken_clarification_before_planning(tmp_path):
    memory = SQLitePersonalMemory(tmp_path / "memory.db")
    registry = ComponentRegistry(ram_budget_mb=64)
    entries = (
        (
            ComponentSpec(
                "vision",
                Layer.PERCEPTION,
                ("perception.scene",),
                "vision-fixture",
                estimated_ram_mb=1,
            ),
            StructuredFramePerception(),
        ),
        (
            ComponentSpec(
                "memory",
                Layer.PERSONAL,
                ("personal.recall", "personal.record"),
                "memory-fixture",
                estimated_ram_mb=1,
            ),
            memory,
        ),
        (
            ComponentSpec(
                "voice-nlp", Layer.VOICE, ("voice.ground",), "voice-fixture", estimated_ram_mb=1
            ),
            SceneGroundedVoiceNLP(),
        ),
        (
            ComponentSpec(
                "tts", Layer.VOICE, ("voice.synthesize",), "tts-fixture", estimated_ram_mb=1
            ),
            Utf8SpeechSynthesisFixture(),
        ),
    )
    for component_spec, component in entries:
        registry.register(component_spec, lambda value=component: value)
    system = PhysicalAI(
        registry,
        profile=PiRuntimeProfile(64),
        robot_model_id="offline-test-robot",
        gripper_geometry={"type": "parallel-jaw", "max_width_m": 0.08},
    )
    frame = CameraFrame(
        "camera",
        {"objects": [{"id": "mug-1", "label": "mug", "confidence": 1, "position_m": (0, 0, 0)}]},
        time(),
    )
    result = system.run(TaskRequest("sam", (frame,), instruction="bring the wrench"))
    assert isinstance(result, ClarificationResult)
    assert result.question == "Which object do you mean?"
    assert result.response_audio == b"TTS:Which object do you mean?"
    context = memory.run(MemoryQuery("sam", "follow-up", {}))
    assert any("Clarification requested" in comment for comment in context.recent_comments)
