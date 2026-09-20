import json
from pathlib import Path

import pytest

from moira.calibration import create_servo_calibration_template
from moira.episode_dataset import (
    EpisodeRecorder,
    create_camera_calibration_template,
    initialize_episode_dataset,
    validate_camera_calibration,
)
from moira.robot_config import RobotModel

MODEL = Path("robot_models/four_dof_desktop_arm/physical_three_actuator_model.json")


def _model():
    return json.loads(MODEL.read_text(encoding="utf-8"))


def _servo_record():
    record = create_servo_calibration_template(_model(), "left")
    record["captured_at"] = "2026-09-19T18:00:00-04:00"
    record["controller"].update(
        i2c_address=0x40,
        reference_clock_hz=25_000_000,
        pwm_frequency_hz=50,
        servo_model_voltage_validated=True,
        servo_power_validated=True,
        output_enable_validated=True,
    )
    for index, joint in enumerate(record["joints"].values()):
        joint.update(
            lower_deg=-40,
            upper_deg=40,
            home_deg=0,
            max_velocity_deg_s=30,
            pulse_at_lower_us=1100 + index,
            pulse_at_upper_us=1900 + index,
        )
    return record


def _camera_record():
    record = create_camera_calibration_template(_model())
    record.update(
        captured_at="2026-09-19T18:15:00-04:00",
        resolution_px=[640, 480],
        camera_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        distortion_coefficients=[0, 0, 0, 0, 0],
        camera_to_base_matrix=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )
    return record


def test_camera_calibration_template_is_unset_and_validator_rejects_it():
    value = create_camera_calibration_template(_model())
    assert value["camera_id"] == "co6-usb"
    assert value["camera_matrix"] is None
    with pytest.raises(ValueError, match="captured_at"):
        validate_camera_calibration(value, RobotModel.from_mapping(_model()))


def test_episode_dataset_requires_complete_lineage_and_publishes_atomically(tmp_path):
    servo_path = tmp_path / "servo.json"
    camera_path = tmp_path / "camera.json"
    servo_path.write_text(json.dumps(_servo_record()), encoding="utf-8")
    camera_path.write_text(json.dumps(_camera_record()), encoding="utf-8")
    root = tmp_path / "dataset"
    manifest = initialize_episode_dataset(
        root,
        MODEL,
        servo_path,
        camera_path,
        created_at="2026-09-19T18:30:00-04:00",
    )
    assert manifest["robot_model_id"] == "three-actuator-desktop-arm-v2"

    with EpisodeRecorder(
        root,
        "episode-0001",
        "pick up the cube",
        "train",
        started_at="2026-09-19T18:31:00-04:00",
    ) as recorder:
        recorder.add_frame("co6-usb", 100, b"jpeg")
        recorder.add_transition(
            101,
            {"joint_positions_rad": [0, 0], "gripper_width_m": 0.04},
            {"target_joint_positions_rad": [0.1, 0]},
        )
        final = recorder.complete(
            completed_at="2026-09-19T18:31:02-04:00",
            success=True,
            outcome={"object_in_target": True},
        )
    assert final.is_dir()
    episode = json.loads((final / "episode.json").read_text(encoding="utf-8"))
    assert episode["servo_calibration_sha256"] == manifest["servo_calibration_sha256"]
    assert episode["frames"][0]["sha256"]
    assert not any((root / ".staging").iterdir())


def test_episode_recorder_rejects_stale_order_and_aborts_partial_episode(tmp_path):
    servo_path = tmp_path / "servo.json"
    camera_path = tmp_path / "camera.json"
    servo_path.write_text(json.dumps(_servo_record()), encoding="utf-8")
    camera_path.write_text(json.dumps(_camera_record()), encoding="utf-8")
    root = tmp_path / "dataset"
    initialize_episode_dataset(
        root,
        MODEL,
        servo_path,
        camera_path,
        created_at="2026-09-19T18:30:00-04:00",
    )
    with EpisodeRecorder(
        root,
        "episode-0002",
        "move the cube",
        "validation",
        started_at="2026-09-19T18:31:00-04:00",
    ) as recorder:
        recorder.add_frame("co6-usb", 100, b"first")
        with pytest.raises(ValueError, match="increase strictly"):
            recorder.add_frame("co6-usb", 100, b"duplicate")
    assert not (root / "episodes" / "episode-0002").exists()
    assert not any((root / ".staging").iterdir())
