"""Scene-grounded command parser matching MoIRA's voice NLP contract."""

from __future__ import annotations

import json
import re
from typing import Any

MAX_PROMPT_CHARS = 24_000
ALLOWED_ACTIONS = frozenset(
    {
        "pick_place",
        "pick",
        "place",
        "move",
        "bring",
        "handover",
        "hold",
        "pour",
        "insert",
        "open_lid",
        "inspect",
        "home",
        "gesture",
        "stop",
        "other",
    }
)
OBJECT_ACTIONS = ALLOWED_ACTIONS - {"home", "gesture", "stop", "other"}
EXPLICIT_DESTINATION_ACTIONS = frozenset(
    {"pick_place", "place", "move", "pour", "insert"}
)
EXPLICIT_REFERENCE_ROLES = frozenset(
    {"manipulated", "destination", "tool", "recipient", "support"}
)

SYSTEM_PROMPT = """You are the scene-grounding specialist for a physical robot.
Return exactly one JSON object and no prose with these keys:
action, object_references, constraints, needs_clarification,
clarification_question.
Choose action from: pick_place, pick, place, move, bring, handover, hold, pour,
insert, open_lid, inspect, home, gesture, stop, other.
Use only object IDs present in SCENE_OBJECTS. Every object_references entry has
object_id and exactly one role from: manipulated, destination, tool, recipient,
support, context.
Put the manipulated object first and the destination or receiving object second.
Resolve pronouns using recent dialogue
and comments only when the referenced object is unambiguous. Never invent an ID,
position, measurement, ability, or completed action. A visible container or
surface is not a destination unless the user named it or an unambiguous recent
reference resolves to it. Ask one short clarification question if a required
object or destination is missing or ambiguous. Set clarification_question to
null whenever needs_clarification is false."""

CLARIFICATION_EXAMPLE = {
    "TRANSCRIPT": "Move the red cube.",
    "SCENE_OBJECTS": [
        {"id": "red-cube-1", "label": "red cube"},
        {"id": "green-bowl-1", "label": "green bowl"},
    ],
    "ACCOMMODATIONS": [],
    "PREFERENCES": {},
    "RECENT_COMMENTS": [],
    "DIALOGUE": [],
}

CLARIFICATION_RESPONSE = {
    "action": "move",
    "object_references": [{"object_id": "red-cube-1", "role": "manipulated"}],
    "constraints": [],
    "needs_clarification": True,
    "clarification_question": "Where should I move the red cube?",
}


def _nonempty_strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("voice NLP model did not return JSON")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("voice NLP model returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("voice NLP result must be a JSON object")
    return value


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    normalized_phrase = " ".join(re.findall(r"[a-z0-9]+", phrase.casefold()))
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalized_text} "


class Model:
    def __init__(self, **kwargs: Any) -> None:
        config = kwargs["config"]
        self._model_path = config["model_metadata"]["model_id"]
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda").eval()

    @staticmethod
    def _response_schema(object_ids: set[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                "object_references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "object_id": {"type": "string", "enum": sorted(object_ids)},
                            "role": {
                                "type": "string",
                                "enum": [
                                    "manipulated",
                                    "destination",
                                    "tool",
                                    "recipient",
                                    "support",
                                    "context",
                                ],
                            },
                        },
                        "required": ["object_id", "role"],
                        "additionalProperties": False,
                    },
                },
                "constraints": {"type": "array", "items": {"type": "string"}},
                "needs_clarification": {"type": "boolean"},
                "clarification_question": {"type": ["string", "null"]},
            },
            "required": [
                "action",
                "object_references",
                "constraints",
                "needs_clarification",
                "clarification_question",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _context(
        request: dict[str, Any],
    ) -> tuple[str, dict[str, Any], set[str], set[str]]:
        transcript = request.get("transcript")
        world = request.get("world")
        personal = request.get("personal")
        dialogue = request.get("dialogue")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("transcript must be a non-empty string")
        if not isinstance(world, dict) or not isinstance(world.get("objects"), list):
            raise ValueError("world.objects must be a list")
        if not isinstance(personal, dict) or not isinstance(dialogue, list):
            raise ValueError("personal and dialogue context are required")
        objects = []
        object_ids: set[str] = set()
        object_labels: dict[str, str] = {}
        for item in world["objects"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or not item["id"].strip()
                or not isinstance(item.get("label"), str)
            ):
                raise ValueError("every scene object needs an id and label")
            if item["id"] in object_ids:
                raise ValueError("scene object IDs must be unique")
            object_ids.add(item["id"])
            object_labels[item["id"]] = item["label"]
            objects.append(
                {
                    "id": item["id"],
                    "label": item["label"],
                    "position_m": item.get("position_m"),
                    "attributes": item.get("attributes"),
                }
            )
        context = {
            "TRANSCRIPT": " ".join(transcript.strip().split()),
            "SCENE_OBJECTS": objects,
            "ACCOMMODATIONS": personal.get("accommodations", [])[:12],
            "PREFERENCES": personal.get("preferences", {}),
            "RECENT_COMMENTS": personal.get("recent_comments", [])[:8],
            "DIALOGUE": dialogue[-8:],
        }
        serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > MAX_PROMPT_CHARS:
            raise ValueError("scene-grounding context exceeds 24,000 characters")
        dialogue_text = [
            item["text"]
            for item in dialogue[-8:]
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        grounding_text = "\n".join([context["TRANSCRIPT"], *dialogue_text])
        explicit_object_ids = {
            object_id
            for object_id, label in object_labels.items()
            if _contains_phrase(grounding_text, label)
            or _contains_phrase(grounding_text, object_id)
        }
        return context["TRANSCRIPT"], context, object_ids, explicit_object_ids

    @staticmethod
    def _validate_result(
        value: dict[str, Any],
        *,
        transcript: str,
        object_ids: set[str],
        accommodations: Any,
        explicit_object_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        action = value.get("action")
        if action not in ALLOWED_ACTIONS:
            raise ValueError("voice NLP returned an unsupported action")
        references = value.get("object_references")
        if not isinstance(references, list) or any(
            not isinstance(item, dict)
            or set(item) != {"object_id", "role"}
            or not isinstance(item["object_id"], str)
            or not isinstance(item["role"], str)
            for item in references
        ):
            raise ValueError("object_references must contain object_id and role pairs")
        raw_targets = [item["object_id"] for item in references]
        if len(raw_targets) != len(set(raw_targets)):
            raise ValueError("voice NLP returned duplicate target object IDs")
        unknown = [target for target in raw_targets if target not in object_ids]
        if unknown:
            raise ValueError(f"voice NLP invented object IDs: {', '.join(unknown)}")
        explicit_object_ids = object_ids if explicit_object_ids is None else explicit_object_ids
        unsupported_references = {
            item["object_id"]
            for item in references
            if item["role"] in EXPLICIT_REFERENCE_ROLES
            and item["object_id"] not in explicit_object_ids
        }
        references = [
            item for item in references if item["object_id"] not in unsupported_references
        ]
        targets = [item["object_id"] for item in references]
        roles = {item["object_id"]: item["role"] for item in references}
        allowed_roles = {
            "manipulated",
            "destination",
            "tool",
            "recipient",
            "support",
            "context",
        }
        if (
            set(roles) != set(targets)
            or any(role not in allowed_roles for role in roles.values())
        ):
            raise ValueError("object_roles must assign every target a supported role")
        constraints = _nonempty_strings(value.get("constraints"), "constraints")
        if not isinstance(accommodations, list) or any(
            not isinstance(item, str) or not item.strip() for item in accommodations
        ):
            raise ValueError("personal accommodations must be a list of strings")
        constraints = list(
            dict.fromkeys([*constraints, *[item.strip() for item in accommodations]])
        )
        needs_clarification = value.get("needs_clarification")
        if not isinstance(needs_clarification, bool):
            raise ValueError("needs_clarification must be boolean")
        question = value.get("clarification_question")
        if unsupported_references:
            needs_clarification = True
            question = None
        if action in OBJECT_ACTIONS and not targets:
            needs_clarification = True
            question = question or "Which detected object should I use?"
        if action in OBJECT_ACTIONS and targets and not any(
            role in ("manipulated", "tool") for role in roles.values()
        ):
            raise ValueError("physical object action needs a manipulated or tool role")
        if action in EXPLICIT_DESTINATION_ACTIONS and not any(
            role in ("destination", "recipient", "support")
            for role in roles.values()
        ):
            needs_clarification = True
            question = question or "Where should I put the object?"
        if needs_clarification:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("a clarification question is required")
            question = question.strip()
        else:
            question = None
        return {
            "transcript": transcript,
            "action": action,
            "target_object_ids": targets,
            "object_roles": roles,
            "constraints": constraints,
            "needs_clarification": needs_clarification,
            "clarification_question": question,
        }

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("Voice NLP model is not loaded")
        transcript, context, object_ids, explicit_object_ids = self._context(request)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    CLARIFICATION_EXAMPLE,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    CLARIFICATION_RESPONSE,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        prefix_allowed_tokens = build_transformers_prefix_allowed_tokens_fn(
            self._tokenizer,
            JsonSchemaParser(self._response_schema(object_ids)),
        )
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
            )
        output = self._tokenizer.decode(
            generated[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        value = _extract_json(output)
        return self._validate_result(
            value,
            transcript=transcript,
            object_ids=object_ids,
            accommodations=request["personal"].get("accommodations", []),
            explicit_object_ids=explicit_object_ids,
        )
