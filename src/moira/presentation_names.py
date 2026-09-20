"""Plain-language names for judge-facing pipeline output.

Runtime component IDs remain stable API contracts.  Presentation code should
translate them at the boundary instead of exposing deployment-oriented names to
an audience that needs to understand each specialist's job immediately.
"""

from __future__ import annotations


SPECIALIST_NAMES: dict[str, str] = {
    "baseten-stt": "Speech Recognition Specialist",
    "baseten-vision-scene": "Vision Specialist",
    "baseten-voice-nlp": "Language Understanding Specialist",
    "sqlite-personal-memory": "Personal Memory Specialist",
    "baseten-task-planner": "Task Planning Specialist",
    "baseten-deterministic-planner-chain": "Task Planning Specialist",
    "baseten-grasp-pose": "Grasp Planning Specialist",
    "baseten-waypoint-policy": "Movement Specialist",
    "baseten-pour-policy": "Pouring Specialist",
    "baseten-insert-policy": "Insertion Specialist",
    "baseten-open-lid-policy": "Lid-Opening Specialist",
    "baseten-handover-policy": "Handover Specialist",
    "baseten-bimanual-act": "Two-Arm Coordination Specialist",
    "baseten-forward-dynamics": "Motion Prediction Specialist",
    "baseten-rigid-world": "Rigid-Body Physics Specialist",
    "baseten-grasp-contact-world": "Grasp Contact Specialist",
    "baseten-deformable-world": "Flexible-Object Physics Specialist",
    "baseten-human-motion-world": "Human Motion Specialist",
    "baseten-task-reward": "Action Scoring Specialist",
    "baseten-outcome-verifier": "Outcome Verification Specialist",
    "baseten-tts": "Speech Synthesis Specialist",
    "local-bimanual-ik": "Inverse Kinematics Specialist",
    "local-trajectory-planner": "Trajectory Planning Specialist",
    "local-collision-checker": "Collision Safety Specialist",
    "local-safety-risk": "Safety Risk Specialist",
    "local-tactile-signal": "Grip Feedback Specialist",
    "local-failure-classifier": "Failure Diagnosis Specialist",
    "local-load-estimator": "Object Weight Specialist",
    "local-prediction-error": "Prediction Learning Specialist",
    "local-load-feedback": "Payload Learning Specialist",
    "dual-arm-hardware": "Robot Motor Controller",
}

WORLD_MODEL_NAMES: dict[str, str] = {
    "forward-dynamics": "Motion Prediction",
    "rigid-dynamics": "Rigid-Body Physics",
    "grasp-contact": "Grasp Contact Physics",
    "deformable-dynamics": "Flexible-Object Physics",
    "human-motion": "Human Motion Prediction",
    "bimanual-dynamics": "Two-Arm Physics",
}

ROUTING_STRATEGY_NAMES: dict[str, str] = {
    "contract_singleton": "Only compatible specialist",
    "remote_semantic": "Best semantic match",
    "semantic_similarity": "Best semantic match",
    "exact-contract": "Required capability match",
    "registry": "Required capability match",
}


def _title_from_identifier(identifier: str) -> str:
    words = identifier.removeprefix("baseten-").removeprefix("local-").split("-")
    return " ".join(word.capitalize() for word in words) + " Specialist"


def specialist_display_name(component_id: str | None) -> str:
    """Return a concise role name without changing the runtime component ID."""

    if not component_id:
        return "No Specialist Selected"
    return SPECIALIST_NAMES.get(component_id, _title_from_identifier(component_id))


def world_model_display_name(model_id: str) -> str:
    """Translate a simulation model kind into language suitable for a pitch."""

    return WORLD_MODEL_NAMES.get(model_id, _title_from_identifier(model_id))


def routing_strategy_display_name(strategy: str | None) -> str:
    """Explain why the router selected a specialist."""

    if not strategy:
        return "Selection reason unavailable"
    return ROUTING_STRATEGY_NAMES.get(strategy, strategy.replace("_", " ").capitalize())
