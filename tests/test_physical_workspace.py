import json
from dataclasses import replace

import pytest

from moira.episode_dataset import create_camera_calibration_template
from moira.production import (
    load_physical_runtime_config,
    prepare_physical_workspace,
)


def calibrated_config(tmp_path):
    config = load_physical_runtime_config("config/pi4_runtime.json")
    model = json.loads(config.robot_model.read_text(encoding="utf-8"))
    calibration = create_camera_calibration_template(model)
    calibration.update(
        captured_at="2026-09-19T19:00:00-04:00",
        resolution_px=[640, 480],
        camera_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        distortion_coefficients=[0, 0, 0, 0, 0],
        camera_to_base_matrix=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        workspace={"table_height_m": 0.0, "bounds_base_frame_m": None},
    )
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    return replace(config, camera_calibration=path)


def task_workspace():
    return {
        "perception": {
            "labels": ["red block", "blue tray"],
            "object_dimensions_m": {
                "red block": [0.04, 0.04, 0.04],
                "blue tray": [0.12, 0.02, 0.08],
            },
        }
    }


def test_runtime_injects_lineage_checked_camera_calibration(tmp_path):
    workspace = prepare_physical_workspace(
        calibrated_config(tmp_path),
        task_workspace(),
        camera_id="co6-usb",
    )
    assert workspace["coordinate_frame"] == "robot_base"
    assert workspace["up_axis"] == "y"
    assert workspace["table_height_m"] == 0.0
    camera = workspace["perception"]["camera_calibrations"]["co6-usb"]
    assert camera["resolution_px"] == [640, 480]


def test_runtime_pins_laptop_camera_and_authenticated_pi_controller():
    config = load_physical_runtime_config("config/pi4_runtime.json")
    assert config.camera.transport == "opencv"
    assert config.camera.camera_id == "co6-usb"
    assert config.camera.device_env == "MOIRA_CAMERA_DEVICE"
    assert config.controller.transport == "pi_http"
    assert config.controller.url_env == "MOIRA_ROBOT_URL"
    assert config.controller.token_env == "MOIRA_ROBOT_TOKEN"


def test_task_workspace_cannot_override_trusted_camera_geometry(tmp_path):
    workspace = task_workspace()
    workspace["perception"]["camera_calibrations"] = {"co6-usb": {}}
    with pytest.raises(ValueError, match="cannot override"):
        prepare_physical_workspace(
            calibrated_config(tmp_path),
            workspace,
            camera_id="co6-usb",
        )


def test_incomplete_real_camera_calibration_blocks_physical_workspace():
    config = load_physical_runtime_config("config/pi4_runtime.json")
    with pytest.raises(ValueError, match="captured_at"):
        prepare_physical_workspace(config, task_workspace(), camera_id="co6-usb")
