import json
from pathlib import Path

import pytest

from moira.calibration import (
    apply_servo_calibration,
    apply_servo_calibration_file,
    create_servo_calibration_template,
)
from moira.robot_config import RobotModel

MODEL = Path("robot_models/four_dof_desktop_arm/physical_three_actuator_model.json")


def _model():
    return json.loads(MODEL.read_text(encoding="utf-8"))


def _complete_record():
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
            lower_deg=-45,
            upper_deg=45,
            home_deg=0,
            max_velocity_deg_s=40 + index,
            pulse_at_lower_us=1100 + index * 10,
            pulse_at_upper_us=1900 + index * 10,
        )
    return record


def test_template_uses_confirmed_arm_and_channels_without_inventing_measurements():
    record = create_servo_calibration_template(_model(), "left")

    assert record["robot_model_id"] == "three-actuator-desktop-arm-v2"
    assert record["physical_id"] == "arm_1"
    assert {name: item["channel"] for name, item in record["joints"].items()} == {
        "J1_BASE_YAW": 0,
        "J2_SHOULDER": 1,
        "J3_GRIPPER": 2,
    }
    assert all(item["lower_deg"] is None for item in record["joints"].values())
    with pytest.raises(ValueError, match="not marked installed"):
        create_servo_calibration_template(_model(), "right")


def test_apply_rejects_incomplete_or_wrong_hardware_calibration():
    record = create_servo_calibration_template(_model(), "left")
    with pytest.raises(ValueError, match="captured_at"):
        apply_servo_calibration(_model(), record)

    record = _complete_record()
    record["joints"]["J1_BASE_YAW"]["channel"] = 9
    with pytest.raises(ValueError, match="does not match configured channel 0"):
        apply_servo_calibration(_model(), record)


def test_apply_updates_only_validated_servo_fields_and_preserves_motion_gate(tmp_path):
    updated = apply_servo_calibration(_model(), _complete_record())
    parsed = RobotModel.from_mapping(updated)

    assert parsed.servo_controller.i2c_address == 0x40
    assert parsed.servo_controller.pwm_frequency_hz == 50
    assert parsed.actuator_mapping_validated
    assert parsed.joints[0].lower_rad is not None
    assert parsed.servo_controller.actuators["left"]["J1_BASE_YAW"].pulse_at_lower_us == 1100
    assert not parsed.motion_ready
    assert "collision-geometry validation" in parsed.readiness_issues

    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(_complete_record()), encoding="utf-8")
    output = tmp_path / "model.json"
    packaged = tmp_path / "packaged.json"
    apply_servo_calibration_file(MODEL, calibration_path, output, packaged)
    assert output.read_bytes() == packaged.read_bytes()
