"""Strict, file-backed servo calibration records for physical robot profiles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from .robot_config import RobotModel

SCHEMA_VERSION = 1


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _true(value: object, name: str) -> None:
    if value is not True:
        raise ValueError(f"{name} must be true after physical validation")


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON document without exposing a partially written destination."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def create_servo_calibration_template(
    robot_model: dict[str, Any], arm: str
) -> dict[str, Any]:
    model = RobotModel.from_mapping(robot_model)
    if arm not in model.servo_controller.arm_installations:
        raise ValueError(f"Unknown arm control slot: {arm}")
    installation = model.servo_controller.arm_installations[arm]
    if not installation.installed:
        raise ValueError(f"{installation.physical_id} is not marked installed")
    return {
        "schema_version": SCHEMA_VERSION,
        "robot_model_id": model.model_id,
        "robot_source_sha256": model.source_sha256,
        "arm": arm,
        "physical_id": installation.physical_id,
        "captured_at": None,
        "controller": {
            "i2c_address": model.servo_controller.i2c_address,
            "reference_clock_hz": model.servo_controller.reference_clock_hz,
            "pwm_frequency_hz": model.servo_controller.pwm_frequency_hz,
            "servo_model_voltage_validated": False,
            "servo_power_validated": model.servo_controller.servo_power_validated,
            "output_enable_validated": model.servo_controller.output_enable_validated,
        },
        "joints": {
            joint.name: {
                "channel": model.servo_controller.actuators[arm][joint.name].channel,
                "lower_deg": None,
                "upper_deg": None,
                "home_deg": None,
                "max_velocity_deg_s": None,
                "pulse_at_lower_us": None,
                "pulse_at_upper_us": None,
            }
            for joint in model.joints
        },
        "notes": [],
    }


def _validated_record(
    robot_model: dict[str, Any], calibration: dict[str, Any]
) -> tuple[RobotModel, str, dict[str, Any], dict[str, dict[str, float | int]]]:
    model = RobotModel.from_mapping(robot_model)
    if calibration.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Servo calibration schema_version must be {SCHEMA_VERSION}")
    if calibration.get("robot_model_id") != model.model_id:
        raise ValueError("Servo calibration robot_model_id does not match the robot model")
    if calibration.get("robot_source_sha256") != model.source_sha256:
        raise ValueError("Servo calibration was captured against different robot geometry")
    arm = calibration.get("arm")
    if not isinstance(arm, str) or arm not in model.servo_controller.arm_installations:
        raise ValueError("Servo calibration arm is invalid")
    installation = model.servo_controller.arm_installations[arm]
    if not installation.installed:
        raise ValueError(f"{installation.physical_id} is not marked installed")
    if calibration.get("physical_id") != installation.physical_id:
        raise ValueError("Servo calibration physical_id does not match the installed arm")
    captured_at = calibration.get("captured_at")
    if not isinstance(captured_at, str) or not captured_at.strip():
        raise ValueError("Servo calibration captured_at must be a non-empty timestamp")

    controller = _mapping(calibration.get("controller"), "controller")
    expected_controller_fields = {
        "i2c_address",
        "reference_clock_hz",
        "pwm_frequency_hz",
        "servo_model_voltage_validated",
        "servo_power_validated",
        "output_enable_validated",
    }
    if set(controller) != expected_controller_fields:
        raise ValueError("Servo calibration controller fields do not match the schema")
    address = _integer(controller["i2c_address"], "controller.i2c_address")
    if not 0x03 <= address <= 0x77:
        raise ValueError("controller.i2c_address must be a 7-bit I2C address")
    reference_clock = _number(
        controller["reference_clock_hz"], "controller.reference_clock_hz"
    )
    frequency = _number(controller["pwm_frequency_hz"], "controller.pwm_frequency_hz")
    if reference_clock <= 0:
        raise ValueError("controller.reference_clock_hz must be positive")
    if not 24 <= frequency <= 1526:
        raise ValueError("controller.pwm_frequency_hz must be in [24, 1526]")
    _true(controller["servo_power_validated"], "controller.servo_power_validated")
    _true(
        controller["servo_model_voltage_validated"],
        "controller.servo_model_voltage_validated",
    )
    _true(controller["output_enable_validated"], "controller.output_enable_validated")

    raw_joints = _mapping(calibration.get("joints"), "joints")
    expected_joint_names = {joint.name for joint in model.joints}
    if set(raw_joints) != expected_joint_names:
        raise ValueError("Servo calibration joints must exactly match the robot model")
    validated_joints: dict[str, dict[str, float | int]] = {}
    period_us = 1_000_000.0 / frequency
    expected_joint_fields = {
        "channel",
        "lower_deg",
        "upper_deg",
        "home_deg",
        "max_velocity_deg_s",
        "pulse_at_lower_us",
        "pulse_at_upper_us",
    }
    for joint in model.joints:
        raw = _mapping(raw_joints[joint.name], f"joints.{joint.name}")
        if set(raw) != expected_joint_fields:
            raise ValueError(f"Servo calibration fields are invalid for {joint.name}")
        channel = _integer(raw["channel"], f"joints.{joint.name}.channel")
        configured_channel = model.servo_controller.actuators[arm][joint.name].channel
        if channel != configured_channel:
            raise ValueError(
                f"{joint.name} channel {channel} does not match configured channel "
                f"{configured_channel}"
            )
        lower = _number(raw["lower_deg"], f"joints.{joint.name}.lower_deg")
        upper = _number(raw["upper_deg"], f"joints.{joint.name}.upper_deg")
        home = _number(raw["home_deg"], f"joints.{joint.name}.home_deg")
        velocity = _number(
            raw["max_velocity_deg_s"], f"joints.{joint.name}.max_velocity_deg_s"
        )
        pulse_lower = _number(
            raw["pulse_at_lower_us"], f"joints.{joint.name}.pulse_at_lower_us"
        )
        pulse_upper = _number(
            raw["pulse_at_upper_us"], f"joints.{joint.name}.pulse_at_upper_us"
        )
        if lower >= upper:
            raise ValueError(f"{joint.name} lower_deg must be below upper_deg")
        if not lower <= home <= upper:
            raise ValueError(f"{joint.name} home_deg must be inside its limits")
        if velocity <= 0:
            raise ValueError(f"{joint.name} max_velocity_deg_s must be positive")
        if pulse_lower <= 0 or pulse_upper <= 0 or pulse_lower == pulse_upper:
            raise ValueError(f"{joint.name} pulse endpoints must be positive and distinct")
        if pulse_lower >= period_us or pulse_upper >= period_us:
            raise ValueError(f"{joint.name} pulse endpoints must be below the PWM period")
        validated_joints[joint.name] = {
            "channel": channel,
            "lower_deg": lower,
            "upper_deg": upper,
            "home_deg": home,
            "max_velocity_deg_s": velocity,
            "pulse_at_lower_us": pulse_lower,
            "pulse_at_upper_us": pulse_upper,
        }
    return model, arm, controller, validated_joints


def apply_servo_calibration(
    robot_model: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    """Validate a complete calibration record and return an updated robot model."""
    _, arm, controller_record, joint_records = _validated_record(robot_model, calibration)
    result = deepcopy(robot_model)
    controller = result["servo_controller"]
    for name in (
        "i2c_address",
        "reference_clock_hz",
        "pwm_frequency_hz",
        "servo_power_validated",
        "output_enable_validated",
    ):
        controller[name] = controller_record[name]
    joint_by_name = {joint["name"]: joint for joint in result["joints"]}
    for name, record in joint_records.items():
        joint = joint_by_name[name]
        for field in ("lower_deg", "upper_deg", "home_deg", "max_velocity_deg_s"):
            joint[field] = record[field]
        actuator = controller["actuators"][arm][name]
        actuator["channel"] = record["channel"]
        actuator["pulse_at_lower_us"] = record["pulse_at_lower_us"]
        actuator["pulse_at_upper_us"] = record["pulse_at_upper_us"]
    result["actuator_mapping_validated"] = True
    serialized = json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode()
    records = result.setdefault("calibration_records", {})
    records[arm] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": calibration["captured_at"],
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }
    power_blocker = (
        "the SG90 supply-voltage rating must be confirmed before using the 6 V servo rail"
    )
    if controller_record["servo_model_voltage_validated"]:
        result["blockers"] = [
            item for item in result.get("blockers", []) if item != power_blocker
        ]
    RobotModel.from_mapping(result)
    return result


def write_servo_calibration_template(model_path: Path, arm: str, output_path: Path) -> None:
    write_json_atomic(
        output_path,
        create_servo_calibration_template(_load_mapping(model_path, "robot model"), arm),
    )


def apply_servo_calibration_file(
    model_path: Path,
    calibration_path: Path,
    output_path: Path,
    packaged_output: Path | None = None,
) -> dict[str, Any]:
    model = _load_mapping(model_path, "robot model")
    calibration = _load_mapping(calibration_path, "servo calibration")
    updated = apply_servo_calibration(model, calibration)
    write_json_atomic(output_path, updated)
    if packaged_output is not None:
        write_json_atomic(packaged_output, updated)
    return updated
