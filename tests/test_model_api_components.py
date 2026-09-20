import cv2
import numpy as np

from moira.cloud import BasetenModelAPI
from moira.model_api_components import ModelAPIScenePerception, ModelAPIVoiceGrounder
from moira.physical import (
    CameraFrame,
    DetectedObject,
    PerceptionInput,
    PersonalContext,
    VoiceGroundingInput,
    WorldState,
)


class FakeModelAPI(BasetenModelAPI):
    def __init__(self, responses):
        self.model = "fake/model"
        self.responses = iter(responses)

    def complete_json(self, messages, *, schema, name, max_tokens=800):
        assert messages and schema["type"] == "object" and name
        return next(self.responses)


def test_model_api_vision_projects_detection_to_calibrated_table():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    model = FakeModelAPI(
        [
            {
                "objects": [
                    {
                        "label": "red block",
                        "confidence": 0.98,
                        "bbox_xyxy_normalized_1000": [400, 500, 600, 700],
                    }
                ]
            }
        ]
    )
    workspace = {
        "coordinate_frame": "robot_base",
        "up_axis": "y",
        "table_height_m": 0.0,
        "perception": {
            "labels": ["red block"],
            "object_dimensions_m": {"red block": [0.04, 0.04, 0.04]},
            "object_masses_kg": {"red block": 0.02},
            "camera_calibrations": {
                "camera": {
                    "resolution_px": [640, 480],
                    "camera_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                    "distortion_coefficients": [0, 0, 0, 0, 0],
                    "camera_to_base_matrix": [
                        [1, 0, 0, 0],
                        [0, 0, -1, 0.45],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                    ],
                }
            },
        },
    }
    result = ModelAPIScenePerception(model).run(
        PerceptionInput((CameraFrame("camera", encoded.tobytes(), 1.0),), workspace)
    )
    assert result.objects[0].id == "red-block-1"
    assert result.objects[0].estimated_mass_kg == 0.02
    assert result.objects[0].position_m[1] == 0.02
    assert result.coordinate_frame == "robot_base"


def test_voice_grounder_does_not_invent_unspoken_destination():
    world = WorldState(
        (
            DetectedObject("red-block-1", "red block", 0.99, (0.1, 0.02, 0.1)),
            DetectedObject("blue-tray-1", "blue tray", 0.99, (0.2, 0.01, 0.1)),
        ),
        {},
    )
    personal = PersonalContext("u", {}, ("move slowly",), (), {})
    model = FakeModelAPI(
        [
            {
                "action": "move",
                "object_references": [
                    {"object_id": "red-block-1", "role": "manipulated"},
                    {"object_id": "blue-tray-1", "role": "destination"},
                ],
                "constraints": [],
                "needs_clarification": False,
                "clarification_question": None,
            }
        ]
    )
    intent = ModelAPIVoiceGrounder(model).run(
        VoiceGroundingInput("Move the red block", world, personal, ())
    )
    assert intent.target_object_ids == ("red-block-1",)
    assert intent.needs_clarification is True
    assert "move slowly" in intent.constraints


def test_voice_grounder_does_not_treat_memory_as_current_target_authority():
    world = WorldState(
        (
            DetectedObject("red-block-1", "red block", 0.99, (0.1, 0.02, 0.1)),
            DetectedObject("blue-tray-1", "blue tray", 0.99, (0.2, 0.01, 0.1)),
        ),
        {},
    )
    personal = PersonalContext(
        "u",
        {},
        ("move slowly",),
        ("Yesterday I asked you to put an object in the blue tray.",),
        {},
    )
    model = FakeModelAPI(
        [
            {
                "action": "move",
                "object_references": [
                    {"object_id": "red-block-1", "role": "manipulated"},
                    {"object_id": "blue-tray-1", "role": "destination"},
                ],
                "constraints": [],
                "needs_clarification": False,
                "clarification_question": None,
            }
        ]
    )

    intent = ModelAPIVoiceGrounder(model).run(
        VoiceGroundingInput("Move the red block", world, personal, ())
    )

    assert intent.target_object_ids == ("red-block-1",)
    assert intent.object_roles == {"red-block-1": "manipulated"}
    assert intent.needs_clarification is True
    assert intent.clarification_question == "Where should I put the object?"
