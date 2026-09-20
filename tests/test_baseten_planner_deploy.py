import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def load_module(monkeypatch):
    chains = ModuleType("truss_chains")

    class ChainletBase:
        pass

    class Configuration:
        def __init__(self, **values):
            self.values = values

    chains.ChainletBase = ChainletBase
    chains.RemoteConfig = Configuration
    chains.DockerImage = Configuration
    chains.make_abs_path_here = lambda value: value
    chains.mark_entrypoint = lambda value: value
    monkeypatch.setitem(sys.modules, "truss_chains", chains)

    pydantic = ModuleType("pydantic")

    class Factory:
        def __init__(self, create):
            self.create = create

    class BaseModel:
        def __init__(self, **values):
            annotations = {}
            for parent in reversed(type(self).__mro__):
                annotations.update(getattr(parent, "__annotations__", {}))
            unknown = set(values) - set(annotations)
            if unknown:
                raise TypeError(f"unexpected fields: {unknown}")
            for key in annotations:
                if key in values:
                    value = values[key]
                else:
                    default = getattr(type(self), key)
                    value = (
                        default.create()
                        if isinstance(default, Factory)
                        else copy.deepcopy(default)
                    )
                setattr(self, key, value)

    pydantic.BaseModel = BaseModel
    pydantic.ConfigDict = lambda **values: values
    pydantic.Field = lambda *, default_factory: Factory(default_factory)
    pydantic.JsonValue = Any
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)

    spec = importlib.util.spec_from_file_location(
        "test_baseten_planner_module", "deploy/baseten_planner/planner.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inputs(module):
    intent = module.GroundedIntentModel(
        transcript="Put the red block in the blue tray",
        action="pick_place",
        target_object_ids=["red-block-1", "blue-tray-1"],
        object_roles={
            "red-block-1": "manipulated",
            "blue-tray-1": "destination",
        },
    )
    world = module.WorldStateModel(
        objects=[
            module.DetectedObjectModel(
                id="red-block-1",
                label="red block",
                confidence=0.9,
                position_m=[0.12, 0.03, 0.08],
            ),
            module.DetectedObjectModel(
                id="blue-tray-1",
                label="blue tray",
                confidence=0.95,
                position_m=[0.18, 0.02, -0.04],
            ),
        ],
        workspace={},
    )
    personal = module.PersonalContextModel(
        user_id="demo-user",
        preferences={},
        accommodations=["move slowly"],
        recent_comments=[],
        workspace={},
    )
    grasps = [
        module.GraspPlanModel(
            target_object_id="red-block-1",
            grasps=[
                module.GraspPoseModel(
                    id="top-grasp-1",
                    target_object_id="red-block-1",
                    position_m=[0.12, 0.05, 0.08],
                    orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                    width_m=0.035,
                    score=0.92,
                    collision_probability=0.03,
                )
            ],
        )
    ]
    return intent, world, personal, grasps


def test_planner_profile_matches_current_single_arm_robot():
    profile = Path("deploy/baseten_planner/planner_profile.json").read_text(
        encoding="utf-8"
    )
    assert '"installed_arms": ["left"]' in profile
    assert '"future_arms": ["right"]' in profile
    assert "three-actuator-desktop-arm-v2" in profile


def test_planner_generates_several_left_arm_candidates(monkeypatch):
    module = load_module(monkeypatch)
    planner = module.PhysicalTaskPlanner()
    intent, world, personal, grasps = inputs(module)
    result = asyncio.run(
        planner.run_remote(intent, world, personal, limit=3, grasps=grasps)
    )
    assert result.plan is None
    assert len(result.candidates) == 3
    assert {step.arms[0] for candidate in result.candidates for step in candidate.steps} == {
        "left"
    }
    assert all(len(candidate.steps) == 5 for candidate in result.candidates)


def test_planner_selects_highest_scoring_safe_simulation(monkeypatch):
    module = load_module(monkeypatch)
    planner = module.PhysicalTaskPlanner()
    intent, world, personal, grasps = inputs(module)
    generated = asyncio.run(
        planner.run_remote(intent, world, personal, limit=3, grasps=grasps)
    ).candidates
    outcomes = [
        module.SimulationOutcomeModel(
            plan_id=candidate.id,
            score=score,
            safe=safe,
            horizon_seconds=2.5,
        )
        for candidate, score, safe in zip(
            generated,
            [0.70, 0.97, 0.88],
            [True, False, True],
            strict=True,
        )
    ]
    result = asyncio.run(
        planner.run_remote(
            intent,
            world,
            personal,
            candidates=generated,
            simulations=outcomes,
        )
    )
    assert result.candidates is None
    assert result.plan.candidate.id == generated[2].id
    assert result.plan.simulation.safe is True


def test_planner_rejects_selection_without_a_safe_candidate(monkeypatch):
    module = load_module(monkeypatch)
    planner = module.PhysicalTaskPlanner()
    intent, world, personal, grasps = inputs(module)
    generated = asyncio.run(
        planner.run_remote(intent, world, personal, limit=1, grasps=grasps)
    ).candidates
    outcome = module.SimulationOutcomeModel(
        plan_id=generated[0].id,
        score=0.99,
        safe=False,
        horizon_seconds=2.5,
        risks=["collision"],
    )
    with pytest.raises(RuntimeError, match="no simulated candidate is safe"):
        asyncio.run(
            planner.run_remote(
                intent,
                world,
                personal,
                candidates=generated,
                simulations=[outcome],
            )
        )
