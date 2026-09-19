"""PCA9685 hardware driver for calibrated dual MG996R arms on Raspberry Pi."""

from __future__ import annotations

import math
from collections.abc import Mapping
from threading import Event, Lock
from time import monotonic
from typing import Any, Protocol

from .physical import ActionChunk, ArmTelemetry, MotionTrajectory, PlanStep, WorldState
from .robot_config import PCA9685Config, RobotModel, ServoCalibration


class PCA9685PulseDevice(Protocol):
    frequency_hz: float

    def set_pulses_us(self, pulses: Mapping[int, float]) -> None: ...

    def disable_channels(self, channels: tuple[int, ...]) -> None: ...


class AdafruitPCA9685Device:
    """Thin, locked adapter over Adafruit's Raspberry Pi PCA9685 driver."""

    def __init__(self, pca: Any, i2c: Any, *, frequency_hz: float) -> None:
        if not 24 <= frequency_hz <= 1526 or not math.isfinite(frequency_hz):
            raise ValueError("PCA9685 frequency must be finite and in [24, 1526] Hz")
        self._pca = pca
        self._i2c = i2c
        self.frequency_hz = float(frequency_hz)
        self._lock = Lock()

    @classmethod
    def from_config(cls, config: PCA9685Config) -> AdafruitPCA9685Device:
        if not config.ready:
            raise RuntimeError("PCA9685 configuration is not calibrated and validated")
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
        except ImportError as exc:
            raise RuntimeError(
                "Install the Raspberry Pi hardware dependencies with 'pip install .[hardware]'"
            ) from exc
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(
            i2c,
            address=config.i2c_address,
            reference_clock_speed=int(config.reference_clock_hz),
        )
        pca.frequency = round(config.pwm_frequency_hz)
        return cls(pca, i2c, frequency_hz=float(pca.frequency))

    def set_pulses_us(self, pulses: Mapping[int, float]) -> None:
        period_us = 1_000_000.0 / self.frequency_hz
        duty_cycles = {}
        for channel, pulse_us in pulses.items():
            if not isinstance(channel, int) or isinstance(channel, bool) or not 0 <= channel < 16:
                raise ValueError("PCA9685 channel must be in [0, 15]")
            if (
                not isinstance(pulse_us, (int, float))
                or isinstance(pulse_us, bool)
                or not math.isfinite(pulse_us)
                or not 0 < pulse_us < period_us
            ):
                raise ValueError("Servo pulse must be finite, positive, and below the PWM period")
            duty_cycles[channel] = round(float(pulse_us) / period_us * 0xFFFF)
        with self._lock:
            for channel, duty_cycle in duty_cycles.items():
                self._pca.channels[channel].duty_cycle = duty_cycle

    def disable_channels(self, channels: tuple[int, ...]) -> None:
        if any(
            not isinstance(channel, int)
            or isinstance(channel, bool)
            or not 0 <= channel < 16
            for channel in channels
        ):
            raise ValueError("PCA9685 channel must be in [0, 15]")
        with self._lock:
            for channel in channels:
                self._pca.channels[channel].duty_cycle = 0

    def close(self) -> None:
        with self._lock:
            for channel in range(16):
                self._pca.channels[channel].duty_cycle = 0
            self._pca.deinit()
            deinit = getattr(self._i2c, "deinit", None)
            if callable(deinit):
                deinit()


class PCA9685ArmDriver:
    """Execute time-bounded joint and gripper trajectories on one calibrated arm."""

    def __init__(
        self,
        arm: str,
        model: RobotModel,
        device: PCA9685PulseDevice,
    ) -> None:
        if arm not in ("left", "right"):
            raise ValueError("PCA9685 arm must be left or right")
        model.require_motion_ready()
        installation = model.servo_controller.arm_installations[arm]
        if not installation.installed:
            raise RuntimeError(f"{installation.physical_id} is not built")
        for method in ("set_pulses_us", "disable_channels"):
            if not callable(getattr(device, method, None)):
                raise TypeError(f"PCA9685 device must implement {method}()")
        if (
            not isinstance(device.frequency_hz, (int, float))
            or not math.isfinite(device.frequency_hz)
            or not math.isclose(
                device.frequency_hz,
                model.servo_controller.pwm_frequency_hz,
                rel_tol=0.01,
                abs_tol=0.01,
            )
        ):
            raise ValueError("PCA9685 device frequency does not match the robot configuration")
        self.arm = arm
        self.physical_arm_id = installation.physical_id
        self.model = model
        self.device = device
        self.calibrations = model.servo_controller.actuators[arm]
        self.channels = tuple(
            calibration.channel for calibration in self.calibrations.values()
        )
        self._stop = Event()

    @staticmethod
    def _interpolate_pulse(
        calibration: ServoCalibration,
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        if not lower <= value <= upper:
            raise ValueError("Servo command exceeds its calibrated input range")
        fraction = (value - lower) / (upper - lower)
        return calibration.pulse_at_lower_us + fraction * (
            calibration.pulse_at_upper_us - calibration.pulse_at_lower_us
        )

    def execute(self, step: PlanStep, world: WorldState) -> ArmTelemetry:
        del step, world
        raise RuntimeError("PCA9685 hardware execution requires a validated motion trajectory")

    def execute_chunk(
        self,
        chunk: ActionChunk,
        trajectory: MotionTrajectory,
        world: WorldState,
    ) -> ArmTelemetry:
        del world
        if self.arm not in chunk.arms:
            raise ValueError(f"Action chunk {chunk.id} does not command the {self.arm} arm")
        if trajectory.chunk_ids != (chunk.id,):
            raise ValueError("Hardware trajectory must contain exactly the commanded chunk")
        if trajectory.duration_seconds > chunk.duration_seconds + 1e-6:
            raise ValueError("Hardware trajectory exceeds the action chunk duration")
        if self._stop.is_set():
            raise RuntimeError(f"{self.arm} arm stop is latched; rearm it before motion")
        started = monotonic()
        for point in trajectory.points:
            joints = point.joint_positions.get(self.arm)
            gripper_width = point.gripper_widths_m.get(self.arm)
            if joints is None or gripper_width is None:
                raise ValueError("Hardware trajectory is missing arm or gripper commands")
            if len(joints) != len(self.model.kinematic_joints):
                raise ValueError("Hardware trajectory joint count does not match the robot")
            remaining = started + point.time_s - monotonic()
            if remaining > 0 and self._stop.wait(remaining):
                raise RuntimeError(f"{self.arm} arm motion was stopped")
            if self._stop.is_set():
                raise RuntimeError(f"{self.arm} arm motion was stopped")
            pulses = {
                self.calibrations[joint.name].channel: self._interpolate_pulse(
                    self.calibrations[joint.name],
                    value,
                    joint.lower_rad,
                    joint.upper_rad,
                )
                for joint, value in zip(self.model.kinematic_joints, joints, strict=True)
            }
            gripper = self.model.gripper_joint
            gripper_calibration = self.calibrations[gripper.name]
            pulses[gripper_calibration.channel] = self._interpolate_pulse(
                gripper_calibration,
                gripper_width,
                0.0,
                self.model.max_gripper_width_m,
            )
            self.device.set_pulses_us(pulses)
        return ArmTelemetry(self.arm, chunk.step_id, True)

    def stop(self) -> None:
        self._stop.set()
        self.device.disable_channels(self.channels)

    def rearm(self) -> None:
        """Clear a latched stop after the physical workspace has been checked."""

        self.device.disable_channels(self.channels)
        self._stop.clear()


def pca9685_installed_arm_drivers(
    model: RobotModel,
    device: PCA9685PulseDevice | None = None,
) -> tuple[dict[str, PCA9685ArmDriver], PCA9685PulseDevice]:
    """Create drivers only for physical arms marked installed in the robot model."""

    model.require_motion_ready()
    shared = device or AdafruitPCA9685Device.from_config(model.servo_controller)
    return (
        {
            arm: PCA9685ArmDriver(arm, model, shared)
            for arm in model.servo_controller.installed_arms
        },
        shared,
    )


def pca9685_arm_drivers(
    model: RobotModel,
    device: PCA9685PulseDevice | None = None,
) -> tuple[PCA9685ArmDriver, PCA9685ArmDriver, PCA9685PulseDevice]:
    """Create left/right drivers that share one configured PCA9685 device."""

    model.require_bimanual_motion_ready()
    drivers, shared = pca9685_installed_arm_drivers(model, device)
    return drivers["left"], drivers["right"], shared
