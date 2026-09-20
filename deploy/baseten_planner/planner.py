"""Typed physical task planner and simulated-candidate selector for MoIRA."""

import json
import re
from pathlib import Path

import truss_chains as chains
from pydantic import BaseModel, ConfigDict, Field, JsonValue

PROFILE_PATH = Path(__file__).with_name("planner_profile.json")
TRANSFER_ACTIONS = {"pick_place", "place", "move", "bring", "handover", "pour", "insert"}
GRASP_ACTIONS = TRANSFER_ACTIONS | {"pick", "hold", "open_lid"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GroundedIntentModel(StrictModel):
    transcript: str
    action: str
    target_object_ids: list[str]
    object_roles: dict[str, str]
    constraints: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None


class DetectedObjectModel(StrictModel):
    id: str
    label: str
    confidence: float
    position_m: list[float]
    estimated_mass_kg: float | None = None
    attributes: dict[str, JsonValue] | None = None


class WorldStateModel(StrictModel):
    objects: list[DetectedObjectModel]
    workspace: dict[str, JsonValue]
    hazards: list[str] = Field(default_factory=list)
    observed_at: float = 0.0
    geometry: dict[str, JsonValue] = Field(default_factory=dict)
    coordinate_frame: str = "robot_base"
    up_axis: str = "z"


class PersonalContextModel(StrictModel):
    user_id: str
    preferences: dict[str, JsonValue]
    accommodations: list[str]
    recent_comments: list[str]
    workspace: dict[str, JsonValue]
    learned_object_masses: dict[str, float] = Field(default_factory=dict)


class GraspPoseModel(StrictModel):
    id: str
    target_object_id: str
    position_m: list[float]
    orientation_xyzw: list[float]
    width_m: float
    score: float
    collision_probability: float


class GraspPlanModel(StrictModel):
    target_object_id: str
    grasps: list[GraspPoseModel]


class PlanStepModel(StrictModel):
    id: str
    action: str
    arms: list[str]
    duration_seconds: float
    target_object_id: str | None = None
    parameters: dict[str, JsonValue] | None = None


class CandidatePlanModel(StrictModel):
    id: str
    steps: list[PlanStepModel]
    rationale: str


class SimulationOutcomeModel(StrictModel):
    plan_id: str
    score: float
    safe: bool
    horizon_seconds: float
    risks: list[str] = Field(default_factory=list)
    predicted: dict[str, JsonValue] | None = None


class FinalPlanModel(StrictModel):
    candidate: CandidatePlanModel
    simulation: SimulationOutcomeModel
    summary: str


class PlannerResponse(StrictModel):
    candidates: list[CandidatePlanModel] | None = None
    plan: FinalPlanModel | None = None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "task"


@chains.mark_entrypoint
class PhysicalTaskPlanner(chains.ChainletBase):
    """Generate physical candidates or choose the best safe simulated candidate."""

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            requirements_file=chains.make_abs_path_here("requirements.txt"),
        ),
    )

    def __init__(self) -> None:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        if profile.get("schema_version") != 1:
            raise ValueError("planner profile schema_version must be 1")
        installed = profile.get("installed_arms")
        if (
            not isinstance(installed, list)
            or not installed
            or any(arm not in ("left", "right") for arm in installed)
            or len(set(installed)) != len(installed)
        ):
            raise ValueError("planner profile needs unique installed arms")
        variants = profile.get("variants")
        durations = profile.get("step_durations_s")
        if not isinstance(variants, list) or not variants or not isinstance(durations, dict):
            raise ValueError("planner profile needs variants and step durations")
        self._profile = profile
        self._installed_arms = installed
        self._variants = variants
        self._durations = durations

    @staticmethod
    def _destination(
        intent: GroundedIntentModel,
        objects: dict[str, DetectedObjectModel],
        world: WorldStateModel,
        personal: PersonalContextModel,
    ) -> tuple[str, list[float]] | None:
        destination_ids = [
            object_id
            for object_id in intent.target_object_ids
            if intent.object_roles.get(object_id) in ("destination", "recipient", "support")
        ]
        if destination_ids:
            destination = objects[destination_ids[0]]
            return destination.id, destination.position_m
        side = personal.preferences.get("delivery_side")
        lowered = " ".join(personal.accommodations).lower()
        if "right arm" in lowered and "broken" in lowered:
            side = "left"
        elif "left arm" in lowered and "broken" in lowered:
            side = "right"
        zones = world.workspace.get("delivery_zones")
        if side in ("left", "right") and isinstance(zones, dict):
            position = zones.get(side)
            if (
                isinstance(position, list)
                and len(position) == 3
                and all(isinstance(value, (int, float)) for value in position)
            ):
                return f"delivery-zone-{side}", [float(value) for value in position]
        return None

    def _generate(
        self,
        intent: GroundedIntentModel,
        world: WorldStateModel,
        personal: PersonalContextModel,
        limit: int,
        grasps: list[GraspPlanModel],
    ) -> list[CandidatePlanModel]:
        if intent.needs_clarification:
            raise ValueError("planner cannot run while intent needs clarification")
        if limit < 1 or limit > 8:
            raise ValueError("candidate limit must be in [1, 8]")
        objects = {item.id: item for item in world.objects}
        unknown = [item for item in intent.target_object_ids if item not in objects]
        if unknown:
            raise ValueError(f"intent references unknown objects: {', '.join(unknown)}")
        if set(intent.object_roles) != set(intent.target_object_ids):
            raise ValueError("intent object roles must cover every grounded object")
        manipulated_ids = [
            object_id
            for object_id in intent.target_object_ids
            if intent.object_roles[object_id] in ("manipulated", "tool")
        ]
        if intent.action in GRASP_ACTIONS and not manipulated_ids:
            raise ValueError("physical object action requires a grounded target")
        if intent.action not in GRASP_ACTIONS and intent.action != "inspect":
            raise ValueError(f"planner does not support action {intent.action}")

        target = objects[manipulated_ids[0]] if manipulated_ids else None
        destination = self._destination(intent, objects, world, personal)
        if intent.action in TRANSFER_ACTIONS and destination is None:
            raise ValueError("transfer action requires a grounded destination or delivery zone")
        grasp_options: list[GraspPoseModel] = []
        if target is not None and intent.action in GRASP_ACTIONS:
            grasp_plan = next(
                (item for item in grasps if item.target_object_id == target.id),
                None,
            )
            if grasp_plan is None or not grasp_plan.grasps:
                raise ValueError(f"no grasp candidates were supplied for {target.id}")
            grasp_options = sorted(
                grasp_plan.grasps,
                key=lambda item: (item.score, -item.collision_probability),
                reverse=True,
            )

        arm_sets = [[arm] for arm in self._installed_arms]
        if len(self._installed_arms) == 2:
            arm_sets.append(list(self._installed_arms))
        candidates: list[CandidatePlanModel] = []
        for variant_index, variant in enumerate(self._variants):
            if not isinstance(variant, dict):
                raise ValueError("planner variant must be an object")
            arms = arm_sets[variant_index % len(arm_sets)]
            variant_id = str(variant["id"])
            prefix = (
                f"{_slug(intent.action)}-"
                f"{_slug(target.id if target else 'workspace')}-{_slug(variant_id)}"
            )
            common: dict[str, JsonValue] = {
                "clearance_m": float(variant["clearance_m"]),
                "speed_scale": float(variant["speed_scale"]),
                "constraints": intent.constraints,
            }
            steps: list[PlanStepModel] = []
            if target is None:
                steps.append(
                    PlanStepModel(
                        id=f"{prefix}-inspect",
                        action="inspect",
                        arms=arms,
                        duration_seconds=float(self._durations["inspect"]),
                        parameters=common,
                    )
                )
            else:
                grasp = grasp_options[variant_index % len(grasp_options)]
                object_parameters = {
                    **common,
                    "target_position_m": target.position_m,
                    "grasp_id": grasp.id,
                    "grasp_position_m": grasp.position_m,
                    "grasp_orientation_xyzw": grasp.orientation_xyzw,
                    "gripper_width_m": grasp.width_m,
                }
                for action in ("approach", "grasp", "lift"):
                    steps.append(
                        PlanStepModel(
                            id=f"{prefix}-{action}",
                            action=action,
                            arms=arms,
                            duration_seconds=float(self._durations[action]),
                            target_object_id=target.id,
                            parameters=object_parameters,
                        )
                    )
                if intent.action in TRANSFER_ACTIONS:
                    destination_id, destination_position = destination
                    transfer_parameters = {
                        **common,
                        "destination_object_id": destination_id,
                        "destination_position_m": destination_position,
                        "gripper_width_m": grasp.width_m,
                    }
                    for action in ("transfer", "release"):
                        steps.append(
                            PlanStepModel(
                                id=f"{prefix}-{action}",
                                action=action,
                                arms=arms,
                                duration_seconds=float(self._durations[action]),
                                target_object_id=target.id,
                                parameters=transfer_parameters,
                            )
                        )
            candidates.append(
                CandidatePlanModel(
                    id=prefix,
                    steps=steps,
                    rationale=(
                        f"Use {variant_id} motion with {'+'.join(arms)} for "
                        f"{intent.action} under the grounded personal constraints"
                    ),
                )
            )
            if len(candidates) == limit:
                break
        return candidates

    @staticmethod
    def _select(
        candidates: list[CandidatePlanModel],
        simulations: list[SimulationOutcomeModel],
    ) -> FinalPlanModel:
        if not candidates:
            raise ValueError("plan selection requires candidates")
        candidate_ids = {item.id for item in candidates}
        outcomes = {item.plan_id: item for item in simulations}
        if len(outcomes) != len(simulations) or candidate_ids != set(outcomes):
            raise ValueError("plan selection requires exactly one simulation per candidate")
        safe = [item for item in candidates if outcomes[item.id].safe]
        if not safe:
            raise RuntimeError("no simulated candidate is safe")
        winner = max(
            safe,
            key=lambda item: (outcomes[item.id].score, -len(outcomes[item.id].risks)),
        )
        outcome = outcomes[winner.id]
        return FinalPlanModel(
            candidate=winner,
            simulation=outcome,
            summary=f"Selected {winner.id} with simulated score {outcome.score:.3f}",
        )

    async def run_remote(
        self,
        intent: GroundedIntentModel,
        world: WorldStateModel,
        personal: PersonalContextModel,
        limit: int = 0,
        # Baseten Chains rejects Optional container parameters, so its supported
        # empty-list defaults are used here. The method never mutates them.
        grasps: list[GraspPlanModel] = [],  # noqa: B006
        candidates: list[CandidatePlanModel] = [],  # noqa: B006
        simulations: list[SimulationOutcomeModel] = [],  # noqa: B006
    ) -> PlannerResponse:
        if candidates or simulations:
            if not candidates or not simulations or limit != 0 or grasps:
                raise ValueError("selection needs candidates and simulations only")
            return PlannerResponse(plan=self._select(candidates, simulations))
        if limit == 0:
            raise ValueError("candidate generation requires limit")
        generated = self._generate(intent, world, personal, limit, grasps)
        return PlannerResponse(candidates=generated)
