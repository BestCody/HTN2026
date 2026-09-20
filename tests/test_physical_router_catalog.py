import json
from collections import Counter
from pathlib import Path

from moira.evaluation import load_samples
from moira.experts import ExpertRegistry

CATALOG = Path("examples/physical_ai_specialists.json")
SAMPLES = Path("examples/physical_ai_routing_samples.jsonl")
VALIDATION = Path("examples/physical_ai_routing_validation.jsonl")
FINAL_HOLDOUT = Path("examples/physical_ai_routing_holdout.jsonl")
BLIND_TEST = Path("examples/physical_ai_routing_test.jsonl")
COMPONENTS = Path("examples/pi4_components.json")


def test_general_specialist_catalog_maps_to_remote_replaceable_components():
    experts = ExpertRegistry.from_json(CATALOG).snapshot()
    manifest = json.loads(COMPONENTS.read_text(encoding="utf-8"))
    remote_ids = {
        component["id"]
        for component in manifest["components"]
        if component["runtime"] == "remote"
    }

    # The general router catalog can describe future hardware, while the
    # installed single-arm runtime exposes only its deployable specialists.
    assert remote_ids <= {expert.id for expert in experts}
    assert "baseten-bimanual-act" not in remote_ids
    assert "baseten-bimanual-world" not in remote_ids
    assert all(expert.simple != expert.abstract for expert in experts)
    assert all(expert.interfaces != ("general",) for expert in experts)
    assert all(len(expert.routing_examples) >= 3 for expert in experts)


def test_compatibility_interfaces_form_semantic_pools_without_route_tables():
    registry = ExpertRegistry.from_json(CATALOG)

    assert len(registry.for_interface("manipulation.policy.v1").snapshot()) == 6
    assert len(registry.for_interface("world.predict.v1").snapshot()) == 6
    assert {
        expert.id for expert in registry.for_interface("voice.intent.v1").snapshot()
    } == {"baseten-voice-nlp"}


def test_general_router_calibration_set_has_paraphrases_for_every_specialist():
    expert_ids = {
        expert.id for expert in ExpertRegistry.from_json(CATALOG).snapshot()
    }
    samples = load_samples(SAMPLES)
    counts = Counter(sample.expert_id for sample in samples)

    assert set(counts) == expert_ids
    assert all(count >= 3 for count in counts.values())
    assert len({sample.instruction.casefold() for sample in samples}) == len(samples)


def test_general_router_validation_covers_every_specialist_without_calibration_overlap():
    expert_ids = {
        expert.id for expert in ExpertRegistry.from_json(CATALOG).snapshot()
    }
    calibration = load_samples(SAMPLES)
    validation = load_samples(VALIDATION)
    counts = Counter(sample.expert_id for sample in validation)

    assert set(counts) == expert_ids
    assert all(count == 2 for count in counts.values())
    assert not (
        {sample.instruction.casefold() for sample in calibration}
        & {sample.instruction.casefold() for sample in validation}
    )


def test_general_router_final_holdout_is_disjoint_and_covers_every_specialist():
    expert_ids = {
        expert.id for expert in ExpertRegistry.from_json(CATALOG).snapshot()
    }
    prior = [*load_samples(SAMPLES), *load_samples(VALIDATION)]
    holdout = load_samples(FINAL_HOLDOUT)

    assert {sample.expert_id for sample in holdout} == expert_ids
    assert len(holdout) == len(expert_ids)
    assert not (
        {sample.instruction.casefold() for sample in prior}
        & {sample.instruction.casefold() for sample in holdout}
    )


def test_general_router_blind_test_is_disjoint_and_covers_every_specialist():
    expert_ids = {
        expert.id for expert in ExpertRegistry.from_json(CATALOG).snapshot()
    }
    prior = [
        *load_samples(SAMPLES),
        *load_samples(VALIDATION),
        *load_samples(FINAL_HOLDOUT),
    ]
    blind = load_samples(BLIND_TEST)

    assert {sample.expert_id for sample in blind} == expert_ids
    assert len(blind) == len(expert_ids)
    assert not (
        {sample.instruction.casefold() for sample in prior}
        & {sample.instruction.casefold() for sample in blind}
    )

