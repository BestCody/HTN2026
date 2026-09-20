"""Baseten Model API specialists for language and multimodal scene reasoning."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .cloud import BasetenModelAPI
from .physical import (
    DetectedObject,
    GroundedIntent,
    PerceptionInput,
    VoiceGroundingInput,
    WorldState,
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_LABELS = 32
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
ALLOWED_ROLES = (
    "manipulated",
    "destination",
    "tool",
    "recipient",
    "support",
    "context",
)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "object"


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    normalized_phrase = " ".join(re.findall(r"[a-z0-9]+", phrase.casefold()))
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalized_text} "


class ModelAPIVoiceGrounder:
    """Ground a transcript to perceived IDs with strict post-generation checks."""

    SYSTEM_PROMPT = """You are the scene-grounding specialist for a physical robot.
Select the action and only the scene object IDs explicitly named by the user or
resolved unambiguously from recent dialogue. A visible container or surface is
not a destination unless the user named or clearly referenced it. Put the
manipulated object first. object_references MUST include every explicitly named
physical object needed for the action. For "pick up X and put it in Y", return X
with role manipulated and Y with role destination. Ask one short clarification
question when a required object or destination is missing or ambiguous. Never
invent an object, pose, measurement, capability, or completed action."""

    EXAMPLE_CONTEXT = {
        "transcript": "Pick up the red cube and put it in the green bowl.",
        "scene_objects": [
            {"id": "red-cube-1", "label": "red cube"},
            {"id": "green-bowl-1", "label": "green bowl"},
        ],
        "accommodations": [],
        "preferences": {},
        "recent_comments": [],
        "dialogue": [],
    }
    EXAMPLE_RESPONSE = {
        "action": "pick_place",
        "object_references": [
            {"object_id": "red-cube-1", "role": "manipulated"},
            {"object_id": "green-bowl-1", "role": "destination"},
        ],
        "constraints": [],
        "needs_clarification": False,
        "clarification_question": None,
    }

    def __init__(self, model: BasetenModelAPI) -> None:
        if not isinstance(model, BasetenModelAPI):
            raise TypeError("voice grounder needs a BasetenModelAPI")
        self.model = model

    @staticmethod
    def _schema(object_ids: set[str]) -> dict[str, Any]:
        object_id_schema: dict[str, Any] = {"type": "string"}
        if object_ids:
            object_id_schema["enum"] = sorted(object_ids)
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                "object_references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "object_id": object_id_schema,
                            "role": {"type": "string", "enum": list(ALLOWED_ROLES)},
                        },
                        "required": ["object_id", "role"],
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
        }

    @staticmethod
    def _context(request: VoiceGroundingInput) -> tuple[dict[str, Any], set[str], set[str]]:
        objects = [
            {
                "id": item.id,
                "label": item.label,
                "position_m": list(item.position_m),
                "attributes": dict(item.attributes or {}),
            }
            for item in request.world.objects
        ]
        object_ids = {item["id"] for item in objects}
        if len(object_ids) != len(objects):
            raise ValueError("scene object IDs must be unique")
        context = {
            "transcript": " ".join(request.transcript.strip().split()),
            "scene_objects": objects,
            "accommodations": list(request.personal.accommodations[:12]),
            "preferences": dict(request.personal.preferences),
            "recent_comments": list(request.personal.recent_comments[:8]),
            "dialogue": [
                {"role": item.role, "text": item.text} for item in request.dialogue[-8:]
            ],
        }
        encoded = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > MAX_PROMPT_CHARS:
            raise ValueError("scene-grounding context exceeds 24,000 characters")
        grounding_text = "\n".join(
            [
                context["transcript"],
                *[item["text"] for item in context["dialogue"]],
            ]
        )
        explicit = {
            item["id"]
            for item in objects
            if _contains_phrase(grounding_text, item["id"])
            or _contains_phrase(grounding_text, item["label"])
        }
        return context, object_ids, explicit

    @staticmethod
    def _validated(
        value: Mapping[str, Any],
        request: VoiceGroundingInput,
        object_ids: set[str],
        explicit_ids: set[str],
    ) -> GroundedIntent:
        action = value.get("action")
        if action not in ALLOWED_ACTIONS:
            raise ValueError("voice Model API returned an unsupported action")
        references = value.get("object_references")
        if not isinstance(references, list) or any(
            not isinstance(item, dict)
            or set(item) != {"object_id", "role"}
            or not isinstance(item["object_id"], str)
            or item["role"] not in ALLOWED_ROLES
            for item in references
        ):
            raise ValueError("object_references must contain valid object_id and role pairs")
        raw_targets = [item["object_id"] for item in references]
        if len(raw_targets) != len(set(raw_targets)):
            raise ValueError("voice Model API returned duplicate object IDs")
        unknown = sorted(set(raw_targets) - object_ids)
        if unknown:
            raise ValueError(f"voice Model API invented object IDs: {', '.join(unknown)}")
        unsupported = {
            item["object_id"]
            for item in references
            if item["role"] in EXPLICIT_REFERENCE_ROLES and item["object_id"] not in explicit_ids
        }
        references = [item for item in references if item["object_id"] not in unsupported]
        targets = tuple(item["object_id"] for item in references)
        roles = {item["object_id"]: item["role"] for item in references}
        constraints = value.get("constraints")
        if not isinstance(constraints, list) or any(
            not isinstance(item, str) or not item.strip() for item in constraints
        ):
            raise ValueError("voice constraints must contain non-empty strings")
        merged_constraints = tuple(
            dict.fromkeys(
                [item.strip() for item in constraints]
                + [item.strip() for item in request.personal.accommodations]
            )
        )
        needs_clarification = value.get("needs_clarification")
        if not isinstance(needs_clarification, bool):
            raise ValueError("needs_clarification must be boolean")
        question = value.get("clarification_question")
        if unsupported:
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
            role in ("destination", "recipient", "support") for role in roles.values()
        ):
            needs_clarification = True
            question = question or "Where should I put the object?"
        if needs_clarification:
            if not isinstance(question, str) or not question.strip():
                question = "Could you clarify the object or destination?"
            question = question.strip()
        else:
            question = None
        return GroundedIntent(
            " ".join(request.transcript.strip().split()),
            action,
            targets,
            roles,
            merged_constraints,
            needs_clarification,
            question,
        )

    def run(self, request: VoiceGroundingInput) -> GroundedIntent:
        if not isinstance(request, VoiceGroundingInput):
            raise TypeError("voice Model API component expects VoiceGroundingInput")
        context, object_ids, explicit_ids = self._context(request)
        value = self.model.complete_json(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        self.EXAMPLE_CONTEXT,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        self.EXAMPLE_RESPONSE,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            schema=self._schema(object_ids),
            name="grounded_physical_intent",
            max_tokens=500,
        )
        return self._validated(value, request, object_ids, explicit_ids)


def _project_to_plane(
    calibration: Mapping[str, Any],
    pixel_x: float,
    pixel_y: float,
    up_axis: str,
    plane_height_m: float,
) -> list[float]:
    import cv2
    import numpy as np

    def matrix(name: str, rows: int, columns: int) -> Any:
        value = calibration.get(name)
        if (
            not isinstance(value, list)
            or len(value) != rows
            or any(not isinstance(row, list) or len(row) != columns for row in value)
        ):
            raise ValueError(f"{name} must be a {rows}x{columns} numeric array")
        return np.asarray(
            [[_finite_number(item, name) for item in row] for row in value],
            dtype=np.float64,
        )

    camera_matrix = matrix("camera_matrix", 3, 3)
    camera_to_base = matrix("camera_to_base_matrix", 4, 4)
    distortion = calibration.get("distortion_coefficients")
    if not isinstance(distortion, list) or len(distortion) not in (4, 5, 8, 12, 14):
        raise ValueError("distortion_coefficients must use an OpenCV coefficient count")
    distortion_array = np.asarray(
        [_finite_number(value, "distortion coefficient") for value in distortion],
        dtype=np.float64,
    )
    rotation = camera_to_base[:3, :3]
    if not np.allclose(camera_to_base[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("camera_to_base_matrix must be a homogeneous transform")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-3
    ):
        raise ValueError("camera_to_base_matrix rotation must be orthonormal")
    normalized = cv2.undistortPoints(
        np.asarray([[[pixel_x, pixel_y]]], dtype=np.float64),
        camera_matrix,
        distortion_array,
    )[0, 0]
    ray_base = rotation @ np.asarray([normalized[0], normalized[1], 1.0])
    origin_base = camera_to_base[:3, 3]
    up_index = {"x": 0, "y": 1, "z": 2}[up_axis]
    if math.isclose(float(ray_base[up_index]), 0.0, abs_tol=1e-9):
        raise ValueError("camera ray is parallel to the support plane")
    distance = (plane_height_m - origin_base[up_index]) / ray_base[up_index]
    if not math.isfinite(float(distance)) or distance <= 0:
        raise ValueError("detected object projects behind the calibrated camera")
    return [float(value) for value in origin_base + distance * ray_base]


class ModelAPIScenePerception:
    """Use multimodal Model API semantics and calibrated local geometry."""

    SYSTEM_PROMPT = """You are the visual perception specialist for a robot arm.
Return every visible requested object. Use only labels from the supplied list.
Each bbox is a tight [x1,y1,x2,y2] integer box normalized to 0..1000. Do not
invent occluded objects and do not return the robot arm unless it is requested."""

    def __init__(self, model: BasetenModelAPI) -> None:
        if not isinstance(model, BasetenModelAPI):
            raise TypeError("scene perception needs a BasetenModelAPI")
        self.model = model

    @staticmethod
    def _raw_image(frame_data: Any) -> bytes:
        if isinstance(frame_data, bytes):
            raw = frame_data
        elif isinstance(frame_data, dict) and isinstance(frame_data.get("base64"), str):
            try:
                raw = base64.b64decode(frame_data["base64"], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("frame.data.base64 is invalid") from exc
        else:
            raise ValueError("camera frame data must be image bytes or base64")
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("camera image must contain 1 byte to 8 MiB")
        return raw

    @staticmethod
    def _image_size(raw: bytes) -> tuple[int, int]:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("camera frame is not a supported encoded image")
        height, width = image.shape[:2]
        return int(width), int(height)

    @staticmethod
    def _schema(labels: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string", "enum": labels},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "bbox_xyxy_normalized_1000": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                        },
                        "required": ["label", "confidence", "bbox_xyxy_normalized_1000"],
                    },
                }
            },
            "required": ["objects"],
        }

    def _detect_frame(
        self,
        frame: Any,
        labels: list[str],
    ) -> tuple[str, int, int, list[dict[str, Any]]]:
        raw = self._raw_image(frame.data)
        width, height = self._image_size(raw)
        media_type = frame.media_type if frame.media_type.startswith("image/") else "image/jpeg"
        data_url = f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"
        value = self.model.complete_json(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Requested labels: " + json.dumps(labels, ensure_ascii=False),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            schema=self._schema(labels),
            name="physical_scene_detections",
            max_tokens=1000,
        )
        objects = value.get("objects")
        if not isinstance(objects, list):
            raise ValueError("vision Model API result needs an objects list")
        return frame.camera_id, width, height, objects

    def run(self, request: PerceptionInput) -> WorldState:
        if not isinstance(request, PerceptionInput):
            raise TypeError("vision Model API component expects PerceptionInput")
        workspace = dict(request.workspace)
        config = workspace.get("perception")
        if not isinstance(config, Mapping):
            raise ValueError("workspace.perception configuration is required")
        raw_labels = config.get("labels")
        if (
            not isinstance(raw_labels, list)
            or not raw_labels
            or len(raw_labels) > MAX_LABELS
            or any(not isinstance(item, str) or not item.strip() for item in raw_labels)
        ):
            raise ValueError("workspace perception needs 1-32 labels")
        labels = [item.strip().casefold() for item in raw_labels]
        if len(labels) != len(set(labels)):
            raise ValueError("workspace perception labels must be unique")
        dimensions = config.get("object_dimensions_m")
        object_masses = config.get("object_masses_kg", {})
        calibrations = config.get("camera_calibrations")
        if (
            not isinstance(dimensions, Mapping)
            or not isinstance(object_masses, Mapping)
            or not isinstance(calibrations, Mapping)
        ):
            raise ValueError("perception object dimensions and camera calibrations are required")
        threshold = _finite_number(config.get("confidence_threshold", 0.32), "threshold")
        if not 0 < threshold <= 1:
            raise ValueError("confidence threshold must be in (0, 1]")
        table_height = _finite_number(workspace.get("table_height_m"), "table height")
        up_axis = workspace.get("up_axis")
        if up_axis not in ("x", "y", "z"):
            raise ValueError("workspace up_axis must be x, y, or z")
        coordinate_frame = workspace.get("coordinate_frame")
        if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
            raise ValueError("workspace coordinate_frame is required")

        with ThreadPoolExecutor(max_workers=min(2, len(request.frames))) as executor:
            detected_frames = list(
                executor.map(
                    lambda frame: self._detect_frame(frame, labels),
                    request.frames,
                )
            )

        detections: list[dict[str, Any]] = []
        for camera_id, width, height, values in detected_frames:
            calibration = calibrations.get(camera_id)
            if not isinstance(calibration, Mapping):
                raise ValueError(f"calibration is missing for camera {camera_id}")
            if calibration.get("resolution_px") != [width, height]:
                raise ValueError(f"camera {camera_id} frame resolution does not match calibration")
            for value in values:
                if not isinstance(value, dict) or set(value) != {
                    "label",
                    "confidence",
                    "bbox_xyxy_normalized_1000",
                }:
                    raise ValueError("vision detection has an invalid shape")
                label = value["label"]
                confidence = _finite_number(value["confidence"], "detection confidence")
                box = value["bbox_xyxy_normalized_1000"]
                if label not in labels or not isinstance(box, list) or len(box) != 4:
                    raise ValueError("vision detection label or bbox is invalid")
                if confidence < threshold:
                    continue
                normalized = [_finite_number(item, "bbox coordinate") for item in box]
                if any(item < 0 or item > 1000 for item in normalized):
                    raise ValueError("normalized bbox coordinates must be in [0, 1000]")
                x1, y1, x2, y2 = (
                    normalized[0] * width / 1000,
                    normalized[1] * height / 1000,
                    normalized[2] * width / 1000,
                    normalized[3] * height / 1000,
                )
                if not (x1 < x2 and y1 < y2):
                    raise ValueError("vision detection bbox must have positive area")
                size = dimensions.get(label)
                if (
                    not isinstance(size, list)
                    or len(size) != 3
                    or any(_finite_number(item, f"{label} dimension") <= 0 for item in size)
                ):
                    raise ValueError(f"positive 3D dimensions are required for {label}")
                mass = object_masses.get(label)
                if mass is not None and _finite_number(mass, f"{label} mass") <= 0:
                    raise ValueError(f"configured mass must be positive for {label}")
                position = _project_to_plane(
                    calibration,
                    (x1 + x2) / 2,
                    y2,
                    up_axis,
                    table_height,
                )
                height_index = {"x": 0, "y": 1, "z": 2}[up_axis]
                position[height_index] += float(size[height_index]) / 2
                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "position": position,
                        "attributes": {
                            "bbox_px": [x1, y1, x2, y2],
                            "dimensions_m": [float(item) for item in size],
                            "width_m": min(
                                float(item)
                                for index, item in enumerate(size)
                                if index != height_index
                            ),
                            "height_m": float(size[height_index]),
                            "camera_id": camera_id,
                            "detector": self.model.model,
                        },
                    }
                )
        detections.sort(key=lambda item: (item["label"], *item["position"][:2]))
        counts: defaultdict[str, int] = defaultdict(int)
        objects: list[DetectedObject] = []
        geometry: dict[str, Any] = {}
        for value in detections:
            counts[value["label"]] += 1
            object_id = f"{_slug(value['label'])}-{counts[value['label']]}"
            geometry[object_id] = value["attributes"]
            objects.append(
                DetectedObject(
                    object_id,
                    value["label"],
                    value["confidence"],
                    tuple(value["position"]),
                    float(object_masses[value["label"]])
                    if value["label"] in object_masses
                    else None,
                    attributes=value["attributes"],
                )
            )
        hazard_labels = {
            str(item).strip().casefold() for item in config.get("hazard_labels", [])
        }
        hazards = tuple(item.id for item in objects if item.label.casefold() in hazard_labels)
        return WorldState(
            tuple(objects),
            workspace,
            hazards,
            max(frame.captured_at for frame in request.frames),
            geometry,
            coordinate_frame,
            up_axis,
        )
