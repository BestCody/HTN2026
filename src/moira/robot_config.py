"""Validated robot-model metadata used by local motion and safety components."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    vector = tuple(_finite(item, name) for item in value)
    norm = math.sqrt(sum(item * item for item in vector))
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{name} must be a unit vector")
    return vector


@dataclass(frozen=True)
class RobotJoint:
    name: str
    role: str
    actuator_model: str
    axis: tuple[float, float, float] | None
    lower_rad: float | None
    upper_rad: float | None
    home_rad: float | None
    origin_m: tuple[float, float, float] | None = None
    max_velocity_rad_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Robot joint name must be non-empty")
        if self.role not in ("kinematic", "gripper"):
            raise ValueError("Robot joint role must be kinematic or gripper")
        if not isinstance(self.actuator_model, str) or not self.actuator_model.strip():
            raise ValueError(f"{self.name} actuator_model must be non-empty")
        if self.axis is not None:
            _vector3(self.axis, f"{self.name} axis")
        limits = (self.lower_rad, self.upper_rad, self.home_rad)
        if any(value is None for value in limits) and not all(value is None for value in limits):
            raise ValueError(f"{self.name} lower, upper, and home limits must be set together")
        if all(value is not None for value in limits):
            lower = _finite(self.lower_rad, f"{self.name} lower limit")
            upper = _finite(self.upper_rad, f"{self.name} upper limit")
            home = _finite(self.home_rad, f"{self.name} home")
            if lower >= upper:
                raise ValueError(f"{self.name} lower limit must be below its upper limit")
            if not lower <= home <= upper:
                raise ValueError(f"{self.name} home must be inside its limits")
        if self.origin_m is not None:
            if not isinstance(self.origin_m, tuple) or len(self.origin_m) != 3:
                raise ValueError(f"{self.name} origin must contain three values")
            for item in self.origin_m:
                _finite(item, f"{self.name} origin")
        if self.max_velocity_rad_s is not None and (
            _finite(self.max_velocity_rad_s, f"{self.name} max velocity") <= 0
        ):
            raise ValueError(f"{self.name} max velocity must be positive")


@dataclass(frozen=True)
class ServoCalibration:
    channel: int | None
    pulse_at_lower_us: float | None
    pulse_at_upper_us: float | None

    def __post_init__(self) -> None:
        if self.channel is not None and (
            not isinstance(self.channel, int)
            or isinstance(self.channel, bool)
            or not 0 <= self.channel < 16
        ):
            raise ValueError("PCA9685 channel must be an integer in [0, 15]")
        for value, name in (
            (self.pulse_at_lower_us, "pulse_at_lower_us"),
            (self.pulse_at_upper_us, "pulse_at_upper_us"),
        ):
            if value is not None and _finite(value, name) <= 0:
                raise ValueError(f"{name} must be positive when present")
        if (
            self.pulse_at_lower_us is not None
            and self.pulse_at_upper_us is not None
            and math.isclose(self.pulse_at_lower_us, self.pulse_at_upper_us)
        ):
            raise ValueError("Servo endpoint pulses must differ")

    @property
    def ready(self) -> bool:
        return (
            self.channel is not None
            and self.pulse_at_lower_us is not None
            and self.pulse_at_upper_us is not None
        )


@dataclass(frozen=True)
class ArmInstallation:
    physical_id: str
    installed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.physical_id, str) or not self.physical_id.strip():
            raise ValueError("Physical arm ID must be non-empty")
        if not isinstance(self.installed, bool):
            raise TypeError("Arm installed state must be boolean")


@dataclass(frozen=True)
class PCA9685Config:
    i2c_address: int | None
    reference_clock_hz: float | None
    pwm_frequency_hz: float | None
    servo_supply_voltage: float | None
    servo_supply_current_a: float | None
    servo_power_validated: bool
    output_enable_validated: bool
    arm_installations: Mapping[str, ArmInstallation]
    actuators: Mapping[str, Mapping[str, ServoCalibration]]

    def __post_init__(self) -> None:
        if self.i2c_address is not None and (
            not isinstance(self.i2c_address, int)
            or isinstance(self.i2c_address, bool)
            or not 0x03 <= self.i2c_address <= 0x77
        ):
            raise ValueError("PCA9685 I2C address must be a 7-bit device address")
        if self.reference_clock_hz is not None and (
            _finite(self.reference_clock_hz, "PCA9685 reference_clock_hz") <= 0
        ):
            raise ValueError("PCA9685 reference_clock_hz must be positive")
        if self.pwm_frequency_hz is not None and not 24 <= _finite(
            self.pwm_frequency_hz, "PCA9685 pwm_frequency_hz"
        ) <= 1526:
            raise ValueError("PCA9685 pwm_frequency_hz must be in [24, 1526]")
        for value, name in (
            (self.servo_supply_voltage, "servo_supply_voltage"),
            (self.servo_supply_current_a, "servo_supply_current_a"),
        ):
            if value is not None and _finite(value, name) <= 0:
                raise ValueError(f"{name} must be positive when present")
        if not isinstance(self.servo_power_validated, bool):
            raise TypeError("servo_power_validated must be boolean")
        if not isinstance(self.output_enable_validated, bool):
            raise TypeError("output_enable_validated must be boolean")
        if set(self.arm_installations) != {"left", "right"}:
            raise ValueError("PCA9685 arm installations require left and right control slots")
        if any(
            not isinstance(installation, ArmInstallation)
            for installation in self.arm_installations.values()
        ):
            raise TypeError("PCA9685 arm installations must contain ArmInstallation values")
        physical_ids = [item.physical_id for item in self.arm_installations.values()]
        if len(physical_ids) != len(set(physical_ids)):
            raise ValueError("Physical arm IDs must be unique")
        if set(self.actuators) != {"left", "right"}:
            raise ValueError("PCA9685 actuator mapping requires left and right arms")
        if any(not isinstance(values, Mapping) for values in self.actuators.values()):
            raise TypeError("PCA9685 per-arm actuator mappings must be mappings")
        channels = [
            calibration.channel
            for values in self.actuators.values()
            for calibration in values.values()
            if calibration.channel is not None
        ]
        if len(channels) != len(set(channels)):
            raise ValueError("PCA9685 channels must be unique across both arms")
        for arm, installation in self.arm_installations.items():
            if not installation.installed and any(
                calibration.channel is not None
                or calibration.pulse_at_lower_us is not None
                or calibration.pulse_at_upper_us is not None
                for calibration in self.actuators[arm].values()
            ):
                raise ValueError("An unbuilt physical arm cannot have actuator calibration")

    @property
    def installed_arms(self) -> tuple[str, ...]:
        return tuple(
            arm for arm, installation in self.arm_installations.items() if installation.installed
        )

    @property
    def ready(self) -> bool:
        return (
            self.i2c_address is not None
            and self.reference_clock_hz is not None
            and self.pwm_frequency_hz is not None
            and self.servo_supply_voltage is not None
            and self.servo_supply_current_a is not None
            and self.servo_power_validated
            and self.output_enable_validated
            and bool(self.installed_arms)
            and all(
                calibration.ready
                for arm in self.installed_arms
                for calibration in self.actuators[arm].values()
            )
        )

    @property
    def bimanual_ready(self) -> bool:
        return self.ready and set(self.installed_arms) == {"left", "right"}


@dataclass(frozen=True)
class RobotModel:
    model_id: str
    source_file: str
    source_sha256: str
    source_manifest: str | None
    print_file: str | None
    up_axis: str | None
    components: tuple[str, ...]
    payload_limit_kg: float | None
    joints: tuple[RobotJoint, ...]
    upper_arm_m: float | None
    forearm_m: float | None
    shoulder_height_m: float | None
    shoulder_offset_m: float | None
    trajectory_frequency_hz: float | None
    required_clearance_m: float | None
    max_gripper_width_m: float | None
    max_gripper_velocity_m_s: float | None
    max_gripper_force_n: float | None
    max_slip_probability: float | None
    max_collision_probability: float | None
    min_grasp_stability_score: float | None
    controller_timeout_margin_s: float | None
    servo_controller: PCA9685Config
    meshes: Mapping[str, str]
    mesh_scale_to_m: float | None
    calibration_complete: bool
    kinematics_validated: bool
    collision_geometry_validated: bool
    actuator_mapping_validated: bool
    payload_validated: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.model_id, "model_id"),
            (self.source_file, "source_file"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for value, name in (
            (self.source_manifest, "source_manifest"),
            (self.print_file, "print_file"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be null or a non-empty path")
        if self.up_axis is not None and self.up_axis not in ("x", "y", "z"):
            raise ValueError("Robot up_axis must be x, y, z, or null")
        if (
            len(self.components) < 2
            or len(set(self.components)) != len(self.components)
            or any(not isinstance(item, str) or not item.strip() for item in self.components)
        ):
            raise ValueError("Robot model requires at least two unique named components")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
        if self.payload_limit_kg is not None and (
            _finite(self.payload_limit_kg, "payload_limit_kg") <= 0
        ):
            raise ValueError("payload_limit_kg must be positive when present")
        if not self.joints or len({joint.name for joint in self.joints}) != len(self.joints):
            raise ValueError("Robot configuration requires unique joints")
        if not any(joint.role == "kinematic" for joint in self.joints):
            raise ValueError("Robot configuration requires at least one kinematic joint")
        if len([joint for joint in self.joints if joint.role == "gripper"]) != 1:
            raise ValueError("Robot configuration requires one gripper joint")
        if not isinstance(self.servo_controller, PCA9685Config):
            raise TypeError("Robot configuration requires a PCA9685 servo controller")
        joint_names = {joint.name for joint in self.joints}
        if any(
            set(mapping) != joint_names for mapping in self.servo_controller.actuators.values()
        ):
            raise ValueError("Each arm's PCA9685 mapping must cover every configured joint")
        if self.servo_controller.pwm_frequency_hz is not None:
            period_us = 1_000_000.0 / self.servo_controller.pwm_frequency_hz
            if any(
                pulse is not None and pulse >= period_us
                for mapping in self.servo_controller.actuators.values()
                for calibration in mapping.values()
                for pulse in (
                    calibration.pulse_at_lower_us,
                    calibration.pulse_at_upper_us,
                )
            ):
                raise ValueError("Servo pulse endpoints must be below the PCA9685 PWM period")
        for value, name in (
            (self.upper_arm_m, "upper_arm_m"),
            (self.forearm_m, "forearm_m"),
            (self.shoulder_height_m, "shoulder_height_m"),
            (self.shoulder_offset_m, "shoulder_offset_m"),
            (self.trajectory_frequency_hz, "trajectory_frequency_hz"),
            (self.required_clearance_m, "required_clearance_m"),
            (self.max_gripper_width_m, "max_gripper_width_m"),
            (self.max_gripper_velocity_m_s, "max_gripper_velocity_m_s"),
            (self.max_gripper_force_n, "max_gripper_force_n"),
            (self.controller_timeout_margin_s, "controller_timeout_margin_s"),
        ):
            if value is not None and _finite(value, name) <= 0:
                raise ValueError(f"{name} must be positive when present")
        if self.max_gripper_width_m is not None and self.max_gripper_width_m > 0.5:
            raise ValueError("max_gripper_width_m cannot exceed 0.5 metres")
        for value, name in (
            (self.max_slip_probability, "max_slip_probability"),
            (self.max_collision_probability, "max_collision_probability"),
            (self.min_grasp_stability_score, "min_grasp_stability_score"),
        ):
            if value is not None and not 0 < _finite(value, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1] when present")
        if not isinstance(self.meshes, Mapping):
            raise TypeError("meshes must be a mapping")
        if any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(path, str)
            or not path.strip()
            for name, path in self.meshes.items()
        ):
            raise ValueError("meshes must map non-empty component names to paths")
        if not set(self.meshes) <= set(self.components):
            raise ValueError("meshes contain components absent from the robot model")
        if self.mesh_scale_to_m is not None and (
            _finite(self.mesh_scale_to_m, "mesh_scale_to_m") <= 0
        ):
            raise ValueError("mesh_scale_to_m must be positive when present")
        for field_name in (
            "calibration_complete",
            "kinematics_validated",
            "collision_geometry_validated",
            "actuator_mapping_validated",
            "payload_validated",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean")
        if any(not isinstance(item, str) or not item.strip() for item in self.blockers):
            raise ValueError("blockers must contain non-empty strings")

    @property
    def motion_ready(self) -> bool:
        return (
            self.calibration_complete
            and self.kinematics_validated
            and self.collision_geometry_validated
            and self.actuator_mapping_validated
            and self.payload_validated
            and self.payload_limit_kg is not None
            and self.up_axis is not None
            and len(self.kinematic_joints) == 3
            and not self.blockers
            and self.upper_arm_m is not None
            and self.forearm_m is not None
            and self.shoulder_height_m is not None
            and self.trajectory_frequency_hz is not None
            and self.required_clearance_m is not None
            and self.max_gripper_width_m is not None
            and self.max_gripper_velocity_m_s is not None
            and self.max_gripper_force_n is not None
            and self.max_slip_probability is not None
            and self.max_collision_probability is not None
            and self.min_grasp_stability_score is not None
            and self.controller_timeout_margin_s is not None
            and self.servo_controller.ready
            and set(self.meshes) == set(self.components)
            and self.mesh_scale_to_m is not None
            and all(joint.axis is not None for joint in self.joints)
            and all(
                joint.lower_rad is not None
                and joint.upper_rad is not None
                and joint.home_rad is not None
                for joint in self.joints
            )
            and all(joint.origin_m is not None for joint in self.joints)
            and all(joint.max_velocity_rad_s is not None for joint in self.joints)
        )

    def require_motion_ready(self) -> None:
        missing = self.readiness_issues
        if missing:
            details = "; ".join(missing)
            raise RuntimeError(f"Robot model is not motion-ready: {details}")

    @property
    def bimanual_motion_ready(self) -> bool:
        return (
            self.motion_ready
            and self.servo_controller.bimanual_ready
            and self.shoulder_offset_m is not None
        )

    def require_bimanual_motion_ready(self) -> None:
        self.require_motion_ready()
        missing = self.bimanual_readiness_issues
        if missing:
            raise RuntimeError(
                "Robot model is not bimanual-motion-ready: " + "; ".join(missing)
            )

    @property
    def bimanual_readiness_issues(self) -> tuple[str, ...]:
        missing = [
            f"{installation.physical_id} is not marked installed"
            for installation in self.servo_controller.arm_installations.values()
            if not installation.installed
        ]
        if self.shoulder_offset_m is None:
            missing.append("bimanual shoulder offset")
        return tuple(missing)

    @property
    def readiness_issues(self) -> tuple[str, ...]:
        missing = []
        if self.up_axis is None:
            missing.append("coordinate-frame up axis")
        if self.payload_limit_kg is None:
            missing.append("validated payload limit")
        if len(self.kinematic_joints) != 3:
            missing.append("three-joint planar motion layout")
        if not self.calibration_complete:
            missing.append("physical calibration")
        for validated, label in (
            (self.kinematics_validated, "kinematics validation"),
            (self.collision_geometry_validated, "collision-geometry validation"),
            (self.actuator_mapping_validated, "actuator mapping validation"),
            (self.payload_validated, "payload validation"),
        ):
            if not validated:
                missing.append(label)
        if self.blockers:
            missing.extend(self.blockers)
        for name, value in (
            ("upper-arm length", self.upper_arm_m),
            ("forearm length", self.forearm_m),
            ("shoulder height", self.shoulder_height_m),
            ("trajectory control frequency", self.trajectory_frequency_hz),
            ("required collision clearance", self.required_clearance_m),
            ("calibrated maximum gripper width", self.max_gripper_width_m),
            ("calibrated gripper velocity limit", self.max_gripper_velocity_m_s),
            ("calibrated gripper force limit", self.max_gripper_force_n),
            ("calibrated slip probability limit", self.max_slip_probability),
            ("calibrated collision probability limit", self.max_collision_probability),
            ("calibrated minimum grasp stability", self.min_grasp_stability_score),
            ("controller timeout margin", self.controller_timeout_margin_s),
        ):
            if value is None:
                missing.append(name)
        if set(self.meshes) != set(self.components):
            missing.append("exported collision meshes for every configured link")
        if self.mesh_scale_to_m is None:
            missing.append("validated mesh scale")
        if any(joint.origin_m is None for joint in self.joints):
            missing.append("joint origins")
        if any(joint.axis is None for joint in self.joints):
            missing.append("joint axes")
        if any(
            joint.lower_rad is None or joint.upper_rad is None or joint.home_rad is None
            for joint in self.joints
        ):
            missing.append("joint limits and home positions")
        if any(joint.max_velocity_rad_s is None for joint in self.joints):
            missing.append("calibrated joint velocity limits")
        controller = self.servo_controller
        if not controller.installed_arms:
            missing.append("at least one installed physical arm")
        if controller.i2c_address is None:
            missing.append("PCA9685 I2C address")
        if controller.reference_clock_hz is None:
            missing.append("PCA9685 calibrated reference clock")
        if controller.pwm_frequency_hz is None:
            missing.append("PCA9685 servo PWM frequency")
        if controller.servo_supply_voltage is None or controller.servo_supply_current_a is None:
            missing.append("servo power-supply rating")
        if any(
            not calibration.ready
            for arm in controller.installed_arms
            for calibration in controller.actuators[arm].values()
        ):
            missing.append("PCA9685 channel and pulse endpoint mapping")
        if not controller.servo_power_validated:
            missing.append("separate servo power validation")
        if not controller.output_enable_validated:
            missing.append("PCA9685 output-enable stop validation")
        return tuple(dict.fromkeys(missing))

    @property
    def kinematic_joints(self) -> tuple[RobotJoint, ...]:
        return tuple(joint for joint in self.joints if joint.role == "kinematic")

    @property
    def gripper_joint(self) -> RobotJoint:
        return next(joint for joint in self.joints if joint.role == "gripper")

    @property
    def servo_models(self) -> Mapping[str, str]:
        """Return the physical servo model selected for every configured actuator."""

        return {joint.name: joint.actuator_model for joint in self.joints}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RobotModel:
        if not isinstance(value, Mapping):
            raise TypeError("Robot model must be a mapping")
        if value.get("schema_version") != 5:
            raise ValueError("Robot model schema_version must be 5")
        geometry = value.get("geometry", {})
        if not isinstance(geometry, Mapping):
            raise TypeError("Robot geometry must be a mapping")
        coordinate_frame = value.get("coordinate_frame", {})
        if not isinstance(coordinate_frame, Mapping):
            raise TypeError("Robot coordinate_frame must be a mapping")
        bimanual_mount = value.get("bimanual_mount", {})
        if not isinstance(bimanual_mount, Mapping):
            raise TypeError("Robot bimanual_mount must be a mapping")
        control_limits = value.get("control_limits", {})
        if not isinstance(control_limits, Mapping):
            raise TypeError("Robot control_limits must be a mapping")
        safety_limits = value.get("safety_limits", {})
        if not isinstance(safety_limits, Mapping):
            raise TypeError("Robot safety_limits must be a mapping")
        controller_value = value.get("servo_controller")
        if not isinstance(controller_value, Mapping):
            raise TypeError("Robot servo_controller must be a mapping")
        if controller_value.get("type") != "pca9685":
            raise ValueError("Robot servo_controller type must be pca9685")
        joints = []
        raw_joints = value.get("joints")
        if not isinstance(raw_joints, list):
            raise TypeError("Robot joints must be a list")
        for joint in raw_joints:
            if not isinstance(joint, Mapping):
                raise TypeError("Each robot joint must be a mapping")
            origin = joint.get("origin_m")
            joints.append(
                RobotJoint(
                    str(joint.get("name", "")),
                    str(joint.get("role", "")),
                    str(joint.get("actuator_model", "")),
                    None
                    if joint.get("axis") is None
                    else _vector3(joint.get("axis"), "joint axis"),
                    None
                    if joint.get("lower_deg") is None
                    else math.radians(_finite(joint.get("lower_deg"), "joint lower_deg")),
                    None
                    if joint.get("upper_deg") is None
                    else math.radians(_finite(joint.get("upper_deg"), "joint upper_deg")),
                    None
                    if joint.get("home_deg") is None
                    else math.radians(_finite(joint.get("home_deg"), "joint home_deg")),
                    None
                    if origin is None
                    else tuple(_finite(item, "joint origin") for item in origin),
                    None
                    if joint.get("max_velocity_deg_s") is None
                    else math.radians(
                        _finite(joint.get("max_velocity_deg_s"), "joint max_velocity_deg_s")
                    ),
                )
            )
        servo_count = value.get("servo_count")
        if (
            not isinstance(servo_count, int)
            or isinstance(servo_count, bool)
            or servo_count != len(joints)
        ):
            raise ValueError("servo_count must equal the number of configured joints")
        components = value.get("components")
        if not isinstance(components, list):
            raise TypeError("Robot components must be a list")
        blockers = value.get("blockers", [])
        if not isinstance(blockers, list):
            raise TypeError("Robot blockers must be a list")
        raw_actuators = controller_value.get("actuators")
        if not isinstance(raw_actuators, Mapping):
            raise TypeError("PCA9685 actuators must be a mapping")
        raw_installations = controller_value.get("arm_installations")
        if not isinstance(raw_installations, Mapping):
            raise TypeError("PCA9685 arm_installations must be a mapping")
        arm_installations = {}
        actuator_mappings = {}
        for arm in ("left", "right"):
            raw_installation = raw_installations.get(arm)
            if not isinstance(raw_installation, Mapping):
                raise TypeError(f"PCA9685 {arm} arm installation must be a mapping")
            arm_installations[arm] = ArmInstallation(
                str(raw_installation.get("physical_id", "")),
                raw_installation.get("installed"),
            )
            raw_mapping = raw_actuators.get(arm)
            if not isinstance(raw_mapping, Mapping):
                raise TypeError(f"PCA9685 {arm} actuator mapping must be a mapping")
            actuator_mappings[arm] = {}
            for joint_name, calibration in raw_mapping.items():
                if not isinstance(calibration, Mapping):
                    raise TypeError("Each PCA9685 actuator calibration must be a mapping")
                actuator_mappings[arm][str(joint_name)] = ServoCalibration(
                    calibration.get("channel"),
                    calibration.get("pulse_at_lower_us"),
                    calibration.get("pulse_at_upper_us"),
                )
        servo_controller = PCA9685Config(
            controller_value.get("i2c_address"),
            controller_value.get("reference_clock_hz"),
            controller_value.get("pwm_frequency_hz"),
            controller_value.get("servo_supply_voltage"),
            controller_value.get("servo_supply_current_a"),
            controller_value.get("servo_power_validated", False),
            controller_value.get("output_enable_validated", False),
            arm_installations,
            actuator_mappings,
        )
        return cls(
            str(value.get("model_id", "")),
            str(value.get("source_file", "")),
            str(value.get("source_sha256", "")),
            None
            if value.get("source_manifest") is None
            else str(value.get("source_manifest")),
            None if value.get("print_file") is None else str(value.get("print_file")),
            None
            if coordinate_frame.get("up_axis") is None
            else str(coordinate_frame.get("up_axis")).casefold(),
            tuple(components),
            None
            if value.get("payload_limit_kg") is None
            else _finite(value.get("payload_limit_kg"), "payload_limit_kg"),
            tuple(joints),
            geometry.get("upper_arm_m"),
            geometry.get("forearm_m"),
            geometry.get("shoulder_height_m"),
            bimanual_mount.get("shoulder_offset_m"),
            control_limits.get("trajectory_frequency_hz"),
            control_limits.get("required_clearance_m"),
            control_limits.get("max_gripper_width_m"),
            control_limits.get("max_gripper_velocity_m_s"),
            control_limits.get("max_gripper_force_n"),
            safety_limits.get("max_slip_probability"),
            safety_limits.get("max_collision_probability"),
            safety_limits.get("min_grasp_stability_score"),
            control_limits.get("controller_timeout_margin_s"),
            servo_controller,
            value.get("meshes", {}),
            value.get("mesh_scale_to_m"),
            value.get("calibration_complete", False),
            value.get("kinematics_validated", False),
            value.get("collision_geometry_validated", False),
            value.get("actuator_mapping_validated", False),
            value.get("payload_validated", False),
            tuple(blockers),
        )


def load_robot_model(path: str | Path, *, verify_source: bool = False) -> RobotModel:
    config_path = Path(path)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    model = RobotModel.from_mapping(value)
    if verify_source:
        source = (config_path.parent / model.source_file).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Robot source model does not exist: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest.casefold() != model.source_sha256.casefold():
            raise ValueError("Robot source model does not match source_sha256")
        if model.source_manifest is not None:
            manifest_path = (config_path.parent / model.source_manifest).resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1:
                raise ValueError("Robot source manifest schema_version must be 1")
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError("Robot source manifest must contain artifacts")
            verified_artifacts: set[Path] = set()
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    raise TypeError("Robot source manifest artifacts must be mappings")
                relative = artifact.get("path")
                expected_digest = artifact.get("sha256")
                expected_bytes = artifact.get("bytes")
                if not isinstance(relative, str) or not relative.strip():
                    raise ValueError("Robot source manifest artifact path must be non-empty")
                artifact_path = (manifest_path.parent / relative).resolve()
                if (
                    not isinstance(expected_bytes, int)
                    or isinstance(expected_bytes, bool)
                    or expected_bytes < 0
                ):
                    raise ValueError("Robot source artifact bytes must be a non-negative integer")
                if (
                    not isinstance(expected_digest, str)
                    or len(expected_digest) != 64
                    or any(
                        character not in "0123456789abcdefABCDEF"
                        for character in expected_digest
                    )
                ):
                    raise ValueError("Robot source artifact sha256 must be a hexadecimal digest")
                if not artifact_path.is_file():
                    raise FileNotFoundError(
                        f"Robot source artifact does not exist: {artifact_path}"
                    )
                if artifact_path.stat().st_size != expected_bytes:
                    raise ValueError(f"Robot source artifact size mismatch: {artifact_path}")
                artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                if artifact_digest.casefold() != str(expected_digest).casefold():
                    raise ValueError(f"Robot source artifact digest mismatch: {artifact_path}")
                verified_artifacts.add(artifact_path)
            required_artifacts = {source}
            if model.print_file is not None:
                required_artifacts.add((config_path.parent / model.print_file).resolve())
            if not required_artifacts <= verified_artifacts:
                raise ValueError("Robot source manifest omits a configured source artifact")
    if model.motion_ready:
        model_dir = config_path.parent.resolve()
        for component, relative in model.meshes.items():
            mesh = (model_dir / relative).resolve()
            if model_dir not in mesh.parents or not mesh.is_file():
                raise FileNotFoundError(
                    f"Robot mesh for {component} is missing or outside the model directory: {mesh}"
                )
    return model


def load_bundled_robot_model() -> RobotModel:
    """Load the packaged active robot metadata without assuming a repository checkout."""

    resource = importlib.resources.files("moira").joinpath(
        "data/four_dof_desktop_arm.json"
    )
    return RobotModel.from_mapping(json.loads(resource.read_text(encoding="utf-8")))
