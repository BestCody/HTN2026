import json
from dataclasses import replace

import pytest

from moira.episode_dataset import create_camera_calibration_template
from moira.pi import PiRuntimeProfile
from moira.production import (
    inspect_physical_runtime,
    load_physical_runtime_config,
    physical_runtime_factories,
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
    assert workspace["calibration_reference_plane_height_m"] == 0.0
    camera = workspace["perception"]["camera_calibrations"]["co6-usb"]
    assert camera["resolution_px"] == [640, 480]


def test_runtime_allows_explicit_raised_task_surface(tmp_path):
    workspace = task_workspace()
    workspace["table_height_m"] = 0.082

    prepared = prepare_physical_workspace(
        calibrated_config(tmp_path),
        workspace,
        camera_id="co6-usb",
    )

    assert prepared["table_height_m"] == 0.082
    assert prepared["calibration_reference_plane_height_m"] == 0.0


def test_runtime_pins_authenticated_pi_camera_and_controller():
    config = load_physical_runtime_config("config/pi4_runtime.json")
    assert config.camera.transport == "lan_http"
    assert config.camera.camera_id == "co6-usb"
    assert config.camera.url_env == "MOIRA_CAMERA_URL"
    assert config.camera.token_env == "MOIRA_ROBOT_TOKEN"
    assert config.camera.rotation_degrees == 180
    assert config.audio.transport == "ffmpeg_dshow"
    assert config.audio.device_env == "MOIRA_MICROPHONE_DEVICE"
    assert config.audio.sample_rate_hz == 16000
    assert config.audio.channels == 1
    assert config.audio.wake_word == "charlie"
    assert config.audio.wake_window_seconds == 4
    assert config.controller.transport == "pi_http"
    assert config.controller.url_env == "MOIRA_ROBOT_URL"
    assert config.controller.token_env == "MOIRA_ROBOT_TOKEN"
    assert config.local_specialists is not None
    assert config.local_specialists.component_ids == {
        "baseten-grasp-pose",
        "baseten-waypoint-policy",
        "baseten-forward-dynamics",
        "baseten-rigid-world",
        "baseten-grasp-contact-world",
        "baseten-task-reward",
        "baseten-outcome-verifier",
    }
    assert config.local_specialists.simulation_checkpoint.is_file()


def test_task_workspace_cannot_override_trusted_camera_geometry(tmp_path):
    workspace = task_workspace()
    workspace["perception"]["camera_calibrations"] = {"co6-usb": {}}
    with pytest.raises(ValueError, match="cannot override"):
        prepare_physical_workspace(
            calibrated_config(tmp_path),
            workspace,
            camera_id="co6-usb",
        )


def test_incomplete_real_camera_calibration_blocks_physical_workspace(tmp_path):
    config = load_physical_runtime_config("config/pi4_runtime.json")
    model = json.loads(config.robot_model.read_text(encoding="utf-8"))
    path = tmp_path / "incomplete-camera.json"
    path.write_text(
        json.dumps(create_camera_calibration_template(model)),
        encoding="utf-8",
    )
    config = replace(config, camera_calibration=path)
    with pytest.raises(ValueError, match="captured_at"):
        prepare_physical_workspace(config, task_workspace(), camera_id="co6-usb")


def test_physical_preflight_accepts_declared_local_component_overrides(monkeypatch):
    config = load_physical_runtime_config("config/pi4_runtime.json")
    monkeypatch.setenv("BASETEN_PLANNER_CHAIN_ID", "")
    monkeypatch.setenv("MOIRA_POUR_POLICY_URL", "")

    default = inspect_physical_runtime(config)
    scripted = inspect_physical_runtime(
        config,
        local_component_ids={"baseten-task-planner", "baseten-pour-policy"},
    )

    assert "BASETEN_PLANNER_CHAIN_ID" in default["missing_core_endpoint_variables"]
    assert "MOIRA_POUR_POLICY_URL" in default["missing_optional_endpoint_variables"]
    assert "BASETEN_PLANNER_CHAIN_ID" not in scripted["missing_core_endpoint_variables"]
    assert "MOIRA_POUR_POLICY_URL" not in scripted["missing_optional_endpoint_variables"]
    assert scripted["component_endpoints"]["baseten-task-planner"][
        "overridden_locally"
    ]


def test_physical_preflight_rejects_unknown_local_component_override():
    config = load_physical_runtime_config("config/pi4_runtime.json")
    config = replace(config, local_specialists=None)
    with pytest.raises(ValueError, match="unknown component IDs"):
        inspect_physical_runtime(config, local_component_ids={"not-in-the-manifest"})


def test_physical_runtime_factory_override_replaces_remote_component(monkeypatch):
    class MotionReadyModel:
        def require_motion_ready(self):
            return None

    config = load_physical_runtime_config("config/pi4_runtime.json")
    monkeypatch.setenv("MOIRA_ROBOT_URL", "http://robot.local:8770")
    monkeypatch.setenv("MOIRA_ROBOT_TOKEN", "test-token")
    replacement = object()

    factories = physical_runtime_factories(
        config,
        MotionReadyModel(),
        PiRuntimeProfile.for_pi4(),
        factory_overrides={"baseten-task-planner": lambda: replacement},
    )

    assert factories["baseten-task-planner"]() is replacement
