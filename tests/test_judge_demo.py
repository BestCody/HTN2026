from io import StringIO

from moira.components import Layer
from moira.judge_demo import JudgeDemoTrace
from moira.physical import PipelineEvent


def test_terminal_trace_explains_router_prediction_and_motor_lock():
    output = StringIO()
    trace = JudgeDemoTrace(output, color=False)

    trace(
        PipelineEvent(
            "semantic_router.completed",
            1.2,
            layer=Layer.MANIPULATION,
            capability="manipulation.skill.waypoint",
            component_id="baseten-waypoint-policy",
            model="untrained-integration-waypoint-v1",
            runtime="remote",
            router="remote_semantic",
            details={"candidate_id": "direct-pick", "latency_seconds": 0.09},
        )
    )
    trace(
        PipelineEvent(
            "simulation.completed",
            1.5,
            details={
                "candidate_id": "direct-pick",
                "score": 0.87,
                "safe": True,
                "horizon_seconds": 2.5,
                "world_models": ("forward-dynamics", "rigid-dynamics"),
            },
        )
    )
    trace(
        PipelineEvent(
            "control.finished",
            1.8,
            details={"executed": False, "success": True},
        )
    )

    rendered = output.getvalue()
    assert "MiniLM selected\nMovement Specialist" in rendered
    assert "internal route baseten-waypoint-policy" in rendered
    assert "2.50s" in rendered
    assert "score=0.870" in rendered
    assert "MOTOR OUTPUT LOCKED" in rendered


def test_terminal_trace_shows_exact_component_model_route():
    output = StringIO()
    trace = JudgeDemoTrace(output, color=False)
    trace(
        PipelineEvent(
            "route.selected",
            0.1,
            layer=Layer.PERCEPTION,
            capability="perception.scene",
            component_id="baseten-vision-scene",
            model="zai-org/GLM-5.3-Flash",
            runtime="remote",
            router="registry",
        )
    )

    rendered = output.getvalue()
    assert "PERCEPTION  >>>  Vision Specialist" in rendered
    assert "model=zai-org/GLM-5.3-Flash" in rendered
    assert "via=registry" in rendered


def test_terminal_heading_uses_charlie_public_brand():
    from moira.software_integration import load_software_integration_config

    output = StringIO()
    trace = JudgeDemoTrace(output, color=False)
    trace.heading(load_software_integration_config("config/software_integration.json"))

    rendered = output.getvalue()
    assert "CHARLIE" in rendered
    assert "TASK ROUTER" in rendered
    assert "MoIRA ROUTER" not in rendered
