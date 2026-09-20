import importlib.util
from pathlib import Path

import pytest


def load_module():
    spec = importlib.util.spec_from_file_location(
        "test_baseten_voice_nlp_module", "deploy/baseten_voice_nlp/model/model.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_voice_nlp_deployment_is_pinned_and_gpu_backed():
    config = Path("deploy/baseten_voice_nlp/config.yaml").read_text(encoding="utf-8")
    assert "Qwen/Qwen2.5-3B-Instruct@aa8e725" in config
    assert "instance_type: L4:4x16" in config


def test_voice_nlp_preserves_grounding_and_personal_constraints():
    module = load_module()
    result = module.Model._validate_result(
        {
            "action": "pick_place",
            "object_references": [
                {"object_id": "red-block-1", "role": "manipulated"},
                {"object_id": "blue-tray-1", "role": "destination"},
            ],
            "constraints": ["move slowly"],
            "needs_clarification": False,
            "clarification_question": None,
        },
        transcript="Put the red block in the blue tray",
        object_ids={"red-block-1", "blue-tray-1"},
        accommodations=["deliver to my left side"],
    )
    assert result["target_object_ids"] == ["red-block-1", "blue-tray-1"]
    assert result["object_roles"]["blue-tray-1"] == "destination"
    assert result["constraints"] == ["move slowly", "deliver to my left side"]


def test_voice_nlp_canonicalizes_resolved_intent_question_to_null():
    module = load_module()
    result = module.Model._validate_result(
        {
            "action": "pick_place",
            "object_references": [
                {"object_id": "red-block-1", "role": "manipulated"},
                {"object_id": "blue-tray-1", "role": "destination"},
            ],
            "constraints": [],
            "needs_clarification": False,
            "clarification_question": "Should I do that?",
        },
        transcript="Put the red block in the blue tray",
        object_ids={"red-block-1", "blue-tray-1"},
        accommodations=[],
    )
    assert result["needs_clarification"] is False
    assert result["clarification_question"] is None


def test_voice_nlp_rejects_hallucinated_object_ids():
    module = load_module()
    with pytest.raises(ValueError, match="invented object IDs"):
        module.Model._validate_result(
            {
                "action": "pick",
                "object_references": [
                    {"object_id": "object-that-is-not-visible", "role": "manipulated"}
                ],
                "constraints": [],
                "needs_clarification": False,
                "clarification_question": None,
            },
            transcript="Pick it up",
            object_ids={"red-block-1"},
            accommodations=[],
        )


def test_voice_nlp_forces_clarification_for_missing_target():
    module = load_module()
    result = module.Model._validate_result(
        {
            "action": "pick",
            "object_references": [],
            "constraints": [],
            "needs_clarification": False,
            "clarification_question": None,
        },
        transcript="Pick it up",
        object_ids={"red-block-1", "blue-block-1"},
        accommodations=[],
    )
    assert result["needs_clarification"] is True
    assert result["clarification_question"]


def test_voice_nlp_forces_clarification_for_missing_transfer_destination():
    module = load_module()
    result = module.Model._validate_result(
        {
            "action": "pick_place",
            "object_references": [
                {"object_id": "red-block-1", "role": "manipulated"}
            ],
            "constraints": [],
            "needs_clarification": False,
            "clarification_question": None,
        },
        transcript="Move the red block",
        object_ids={"red-block-1", "blue-tray-1"},
        accommodations=[],
    )
    assert result["needs_clarification"] is True
    assert result["clarification_question"] == "Where should I put the object?"


def test_voice_nlp_removes_scene_object_that_user_did_not_reference():
    module = load_module()
    result = module.Model._validate_result(
        {
            "action": "move",
            "object_references": [
                {"object_id": "red-block-1", "role": "manipulated"},
                {"object_id": "blue-tray-1", "role": "destination"},
            ],
            "constraints": [],
            "needs_clarification": False,
            "clarification_question": None,
        },
        transcript="Move the red block",
        object_ids={"red-block-1", "blue-tray-1"},
        accommodations=[],
        explicit_object_ids={"red-block-1"},
    )
    assert result["target_object_ids"] == ["red-block-1"]
    assert result["object_roles"] == {"red-block-1": "manipulated"}
    assert result["needs_clarification"] is True
    assert result["clarification_question"] == "Where should I put the object?"


def test_voice_nlp_context_does_not_treat_memory_as_current_target_authority():
    module = load_module()
    _, context, _, explicit = module.Model._context(
        {
            "transcript": "Move the red block",
            "world": {
                "objects": [
                    {"id": "red-block-1", "label": "red block"},
                    {"id": "blue-tray-1", "label": "blue tray"},
                ]
            },
            "personal": {
                "accommodations": [],
                "preferences": {},
                "recent_comments": ["Previously put an object in the blue tray"],
            },
            "dialogue": [],
        }
    )

    assert context["RECENT_COMMENTS"] == ["Previously put an object in the blue tray"]
    assert explicit == {"red-block-1"}
