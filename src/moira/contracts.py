"""Typed JSON response contract for specialized remote physical-AI models."""

from __future__ import annotations

import base64
from typing import Any

from .physical import (
    ActionChunk,
    CandidatePlan,
    CandidatePlanningInput,
    DetectedObject,
    FinalPlan,
    GraspPlan,
    GraspPlanningInput,
    GraspPose,
    GroundedIntent,
    OutcomeInput,
    OutcomeReport,
    PerceptionInput,
    PlanSelectionInput,
    PlanStep,
    PolicyInput,
    PolicyPlan,
    PredictedState,
    RewardInput,
    RewardScore,
    SimulationInput,
    SimulationOutcome,
    SpeechInput,
    SpeechSynthesisInput,
    VoiceGroundingInput,
    WorldModelInput,
    WorldModelPrediction,
    WorldState,
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Remote {name} response must be a JSON object")
    return value


def _step(value: Any) -> PlanStep:
    value = _mapping(value, "plan step")
    return PlanStep(
        value["id"],
        value["action"],
        tuple(value["arms"]),
        value["duration_seconds"],
        value.get("target_object_id"),
        value.get("parameters"),
    )


def _candidate(value: Any) -> CandidatePlan:
    value = _mapping(value, "candidate plan")
    return CandidatePlan(
        value["id"],
        tuple(_step(step) for step in value["steps"]),
        value["rationale"],
    )


def _simulation(value: Any) -> SimulationOutcome:
    value = _mapping(value, "simulation")
    return SimulationOutcome(
        value["plan_id"],
        value["score"],
        value["safe"],
        value["horizon_seconds"],
        tuple(value.get("risks", ())),
        value.get("predicted"),
    )


def _action_chunk(value: Any) -> ActionChunk:
    value = _mapping(value, "action chunk")
    return ActionChunk(
        value["id"],
        value["step_id"],
        value["skill"],
        tuple(value["arms"]),
        value["duration_seconds"],
        value.get("target_object_id"),
        tuple(value["target_pose"]),
        value["gripper_width_m"],
        value["force_limit_n"],
    )


def _world_prediction(value: Any) -> WorldModelPrediction:
    value = _mapping(value, "world-model prediction")
    states = tuple(
        PredictedState(
            state["time_s"],
            {key: tuple(pose) for key, pose in state.get("object_poses", {}).items()},
            {key: tuple(joints) for key, joints in state.get("joint_positions", {}).items()},
            state.get("contact_forces_n", {}),
            state["slip_probability"],
            state["collision_probability"],
        )
        for state in value["states"]
    )
    return WorldModelPrediction(
        value["plan_id"],
        value["model_kind"],
        states,
        value["success_probability"],
        value["uncertainty"],
        tuple(value.get("risks", ())),
    )


def decode_physical_response(response: Any, request: Any) -> Any:
    """Decode the response schema used by the Baseten component deployments."""
    if isinstance(request, SpeechInput):
        if isinstance(response, str):
            return response
        response = _mapping(response, "speech-to-text")
        if isinstance(response.get("text"), str):
            return response["text"]
        segments = response.get("segments")
        if isinstance(segments, list) and all(
            isinstance(segment, dict) and isinstance(segment.get("text"), str)
            for segment in segments
        ):
            return " ".join(segment["text"].strip() for segment in segments).strip()
        raise ValueError("Speech-to-text response needs text or segments")
    if isinstance(request, VoiceGroundingInput):
        response = _mapping(response, "voice NLP")
        value = response.get("intent", response)
        return GroundedIntent(
            value["transcript"],
            value["action"],
            tuple(value.get("target_object_ids", ())),
            tuple(value.get("constraints", ())),
            value.get("needs_clarification", False),
            value.get("clarification_question"),
        )
    if isinstance(request, PerceptionInput):
        response = _mapping(response, "perception")
        objects = tuple(
            DetectedObject(
                value["id"],
                value["label"],
                value["confidence"],
                tuple(value["position_m"]),
                value.get("estimated_mass_kg"),
                value.get("attributes"),
            )
            for value in response.get("objects", ())
        )
        return WorldState(
            objects,
            response.get("workspace", {}),
            tuple(response.get("hazards", ())),
            response.get("observed_at", 0.0),
            response.get("geometry", {}),
            response.get("coordinate_frame", "robot_base"),
            response.get("up_axis", "z"),
        )
    if isinstance(request, GraspPlanningInput):
        response = _mapping(response, "grasp planning")
        values = response.get("plans", response.get("grasps"))
        if not isinstance(values, list):
            raise ValueError("Grasp response needs a plans list")
        plans = []
        for value in values:
            value = _mapping(value, "grasp plan")
            target_id = value["target_object_id"]
            poses = tuple(
                GraspPose(
                    pose["id"],
                    target_id,
                    tuple(pose["position_m"]),
                    tuple(pose["orientation_xyzw"]),
                    pose["width_m"],
                    pose["score"],
                    pose["collision_probability"],
                )
                for pose in value["grasps"]
            )
            plans.append(GraspPlan(target_id, poses))
        return tuple(plans)
    if isinstance(request, CandidatePlanningInput):
        response = _mapping(response, "candidate planning")
        return tuple(_candidate(value) for value in response["candidates"])
    if isinstance(request, PolicyInput):
        response = _mapping(response, "manipulation policy")
        value = response["policy"] if isinstance(response.get("policy"), dict) else response
        return PolicyPlan(
            value["plan_id"],
            value["policy"],
            tuple(_action_chunk(item) for item in value["chunks"]),
        )
    if isinstance(request, WorldModelInput):
        response = _mapping(response, "world model")
        return _world_prediction(response.get("prediction", response))
    if isinstance(request, RewardInput):
        response = _mapping(response, "task reward")
        value = response.get("reward", response)
        return RewardScore(value["plan_id"], value["score"], value.get("components", {}))
    if isinstance(request, SimulationInput):
        response = _mapping(response, "simulation")
        return _simulation(response.get("outcome", response))
    if isinstance(request, PlanSelectionInput):
        response = _mapping(response, "plan selection")
        value = response.get("plan", response)
        return FinalPlan(
            _candidate(value["candidate"]),
            _simulation(value["simulation"]),
            value["summary"],
        )
    if isinstance(request, SpeechSynthesisInput):
        response = _mapping(response, "text-to-speech")
        audio = response.get("audio_b64") or response.get("base64")
        if not isinstance(audio, str):
            raise ValueError("Text-to-speech response needs audio_b64")
        return base64.b64decode(audio, validate=True)
    if isinstance(request, OutcomeInput):
        response = _mapping(response, "outcome verification")
        value = response.get("outcome", response)
        return OutcomeReport(
            value["plan_id"],
            value["status"],
            value["confidence"],
            value.get("observations", {}),
        )
    return response
