"""Baseten Chain entrypoint for the hackathon component router.

Deploy with: truss chains push --watch router.py
"""

from __future__ import annotations

import truss_chains as chains

ROUTES = {
    "perception.scene": "baseten-vision-scene",
    "grasp.pose_6d": "baseten-grasp-pose",
    "personal.recall": "sqlite-personal-memory",
    "personal.record": "sqlite-personal-memory",
    "voice.transcribe": "baseten-stt",
    "voice.ground": "baseten-voice-nlp",
    "voice.synthesize": "baseten-tts",
    "planning.candidates": "baseten-task-planner",
    "planning.select": "baseten-task-planner",
    "manipulation.skill.waypoint": "baseten-waypoint-policy",
    "manipulation.bimanual": "baseten-bimanual-act",
    "manipulation.skill.pour": "baseten-pour-policy",
    "manipulation.skill.insert": "baseten-insert-policy",
    "manipulation.skill.open_lid": "baseten-open-lid-policy",
    "manipulation.skill.handover": "baseten-handover-policy",
    "kinematics.inverse": "local-bimanual-ik",
    "motion.trajectory": "local-trajectory-planner",
    "motion.collision_check": "local-collision-checker",
    "dynamics.predict": "baseten-forward-dynamics",
    "world.rigid_dynamics": "baseten-rigid-world",
    "world.grasp_contact": "baseten-grasp-contact-world",
    "world.bimanual_coordination": "baseten-bimanual-world",
    "world.deformable_dynamics": "baseten-deformable-world",
    "world.human_motion": "baseten-human-motion-world",
    "reward.task_progress": "baseten-task-reward",
    "safety.risk": "local-safety-risk",
    "tactile.contact": "local-tactile-sparsh",
    "tactile.slip": "local-tactile-sparsh",
    "tactile.force": "local-tactile-sparsh",
    "tactile.grasp_stability": "local-tactile-sparsh",
    "control.single_arm": "dual-arm-hardware",
    "control.bimanual": "dual-arm-hardware",
    "outcome.verify": "baseten-outcome-verifier",
    "failure.classify": "local-failure-classifier",
    "load.estimate": "local-load-estimator",
    "feedback.prediction_error": "local-prediction-error",
    "feedback.learn": "local-load-feedback",
}


@chains.mark_entrypoint
class PhysicalComponentRouter(chains.ChainletBase):
    async def run_remote(
        self,
        layer: str,
        capability: str,
        context: dict,
        allowed_components: list[str],
    ) -> dict:
        del context
        if not capability.startswith(f"{layer}."):
            raise ValueError("Capability does not belong to the requested layer")
        try:
            component_id = ROUTES[capability]
        except KeyError:
            raise ValueError(f"No route for exact capability: {capability}") from None
        if component_id not in allowed_components:
            raise ValueError(f"Component is not allowed by the edge catalog: {component_id}")
        return {"component_id": component_id}
