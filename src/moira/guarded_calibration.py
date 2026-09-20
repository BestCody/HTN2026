"""Restricted command envelope for pre-production PCA9685 calibration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GuardedServo:
    name: str
    channel: int
    lower_command_deg: int
    upper_command_deg: int
    home_command_deg: int
    maximum_rate_deg_s: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("guarded servo name must be non-empty")
        if not 0 <= self.channel < 16:
            raise ValueError(f"{self.name} channel must be in [0, 15]")
        if self.lower_command_deg >= self.upper_command_deg:
            raise ValueError(f"{self.name} command envelope is invalid")
        if not self.lower_command_deg <= self.home_command_deg <= self.upper_command_deg:
            raise ValueError(f"{self.name} home command is outside its envelope")
        if not math.isfinite(self.maximum_rate_deg_s) or self.maximum_rate_deg_s <= 0:
            raise ValueError(f"{self.name} maximum rate must be finite and positive")

    def validate_command(self, command_deg: int) -> int:
        if not isinstance(command_deg, int) or isinstance(command_deg, bool):
            raise TypeError(f"{self.name} command must be an integer Arduino angle")
        if not self.lower_command_deg <= command_deg <= self.upper_command_deg:
            raise ValueError(
                f"{self.name} command {command_deg} is outside "
                f"[{self.lower_command_deg}, {self.upper_command_deg}]"
            )
        return command_deg


@dataclass(frozen=True)
class GuardedCalibrationEnvelope:
    robot_model_id: str
    i2c_address: int
    reference_clock_hz: int
    pwm_frequency_hz: int
    arduino_angle_range_deg: tuple[int, int]
    pca9685_tick_range: tuple[int, int]
    servos: dict[str, GuardedServo]
    disabled_channels: tuple[int, ...]
    physical_velocity_validated: bool
    stop_validated: bool

    def __post_init__(self) -> None:
        if not self.robot_model_id.strip():
            raise ValueError("guarded envelope robot_model_id must be non-empty")
        if not 0x03 <= self.i2c_address <= 0x77:
            raise ValueError("guarded envelope I2C address must be a 7-bit address")
        if self.reference_clock_hz <= 0 or not 24 <= self.pwm_frequency_hz <= 1526:
            raise ValueError("guarded envelope PCA9685 timing is invalid")
        angle_low, angle_high = self.arduino_angle_range_deg
        tick_low, tick_high = self.pca9685_tick_range
        if angle_low >= angle_high or tick_low >= tick_high:
            raise ValueError("guarded envelope mapping ranges must increase")
        if set(self.servos) != {"J1_BASE_YAW", "J2_SHOULDER", "J3_GRIPPER"}:
            raise ValueError("guarded envelope must contain exactly the three installed servos")
        active = [servo.channel for servo in self.servos.values()]
        if len(active) != len(set(active)):
            raise ValueError("guarded envelope servo channels must be unique")
        if len(self.disabled_channels) != len(set(self.disabled_channels)):
            raise ValueError("guarded envelope disabled channels must be unique")
        if set(active) & set(self.disabled_channels):
            raise ValueError("an active guarded channel cannot also be disabled")
        if 3 not in self.disabled_channels:
            raise ValueError("unused PCA9685 channel 3 must remain disabled")
        if (
            self.physical_velocity_validated is not False
            or self.stop_validated is not False
        ):
            raise ValueError(
                "pre-calibration envelope cannot claim velocity or stop validation"
            )

    def command_to_tick(self, joint_name: str, command_deg: int) -> int:
        try:
            servo = self.servos[joint_name]
        except KeyError as exc:
            raise ValueError(f"unknown guarded servo: {joint_name}") from exc
        command = servo.validate_command(command_deg)
        angle_low, angle_high = self.arduino_angle_range_deg
        tick_low, tick_high = self.pca9685_tick_range
        # Reproduce Arduino map() integer arithmetic used for the supplied
        # physical observations. All configured angles are non-negative.
        return tick_low + (command - angle_low) * (tick_high - tick_low) // (
            angle_high - angle_low
        )

    def ramp(
        self,
        joint_name: str,
        start_command_deg: int,
        target_command_deg: int,
        rate_deg_s: float,
    ) -> tuple[tuple[int, int], ...]:
        try:
            servo = self.servos[joint_name]
        except KeyError as exc:
            raise ValueError(f"unknown guarded servo: {joint_name}") from exc
        start = servo.validate_command(start_command_deg)
        target = servo.validate_command(target_command_deg)
        if (
            not isinstance(rate_deg_s, (int, float))
            or isinstance(rate_deg_s, bool)
            or not math.isfinite(rate_deg_s)
            or rate_deg_s <= 0
        ):
            raise ValueError("guarded ramp rate must be finite and positive")
        if rate_deg_s > servo.maximum_rate_deg_s:
            raise ValueError(
                f"{joint_name} rate {rate_deg_s:g} exceeds guarded maximum "
                f"{servo.maximum_rate_deg_s:g} deg/s"
            )
        direction = 1 if target >= start else -1
        commands = range(start, target + direction, direction)
        return tuple((command, self.command_to_tick(joint_name, command)) for command in commands)


def load_guarded_calibration_envelope(
    path: str | Path,
) -> GuardedCalibrationEnvelope:
    value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("guarded calibration schema_version must be 1")
    controller = value.get("controller")
    mapping = value.get("arduino_mapping")
    raw_servos = value.get("servos")
    validation = value.get("validation")
    if not all(isinstance(item, dict) for item in (controller, mapping, raw_servos, validation)):
        raise TypeError("guarded calibration sections must be objects")
    servos = {
        name: GuardedServo(
            name,
            servo["channel"],
            servo["lower_command_deg"],
            servo["upper_command_deg"],
            servo["home_command_deg"],
            float(servo["maximum_rate_deg_s"]),
        )
        for name, servo in raw_servos.items()
        if isinstance(name, str) and isinstance(servo, dict)
    }
    if len(servos) != len(raw_servos):
        raise TypeError("guarded calibration servos must be named objects")
    return GuardedCalibrationEnvelope(
        value.get("robot_model_id", ""),
        controller["i2c_address"],
        controller["reference_clock_hz"],
        controller["pwm_frequency_hz"],
        tuple(mapping["angle_range_deg"]),
        tuple(mapping["tick_range"]),
        servos,
        tuple(value.get("disabled_channels", ())),
        validation.get("physical_velocity_validated"),
        validation.get("stop_validated"),
    )
