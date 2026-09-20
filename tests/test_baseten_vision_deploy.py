import importlib.util
from pathlib import Path

import pytest


def load_module():
    spec = importlib.util.spec_from_file_location(
        "test_baseten_vision_module", "deploy/baseten_vision/model/model.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vision_deployment_is_pinned_and_gpu_backed():
    config = Path("deploy/baseten_vision/config.yaml").read_text(encoding="utf-8")
    assert "IDEA-Research/grounding-dino-tiny@a2bb814" in config
    assert "transformers==4.57.6" in config
    assert "instance_type: L4:4x16" in config


def test_vision_keeps_grounding_dino_mixed_branches_in_float32():
    code = Path("deploy/baseten_vision/model/model.py").read_text(encoding="utf-8")
    assert "torch_dtype=" not in code
    assert 'inputs["pixel_values"]' not in code


def test_vision_projects_camera_ray_onto_calibrated_support_plane():
    module = load_module()
    position = module._project_to_plane(
        {
            "camera_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
            "camera_to_base_matrix": [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, 1],
            ],
        },
        420.0,
        240.0,
        "z",
        0.0,
    )
    assert position == pytest.approx([0.2, 0.0, 0.0])


def test_vision_requires_labels_and_each_camera_calibration():
    module = load_module()
    with pytest.raises(ValueError, match="perception configuration"):
        module.Model._perception_config({})
    config = module.Model._perception_config({"perception": {"labels": ["red block"]}})
    with pytest.raises(ValueError, match="missing for camera"):
        module.Model._camera_calibration(config, "co6-usb")


def test_vision_matches_only_requested_grounding_labels():
    module = load_module()
    labels = ["red block", "blue tray"]
    assert module.Model._match_label("a red block", labels) == "red block"
    assert module.Model._match_label("person", labels) is None
