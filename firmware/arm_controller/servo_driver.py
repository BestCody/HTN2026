from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)


class ServoDriver(Protocol):
    hardware: bool

    def write_us(self, channel: int, microseconds: float) -> None: ...
    def release(self, channel: int) -> None: ...
    def shutdown(self) -> None: ...


class MockDriver:
    """Used when running off-Pi (no I²C bus). Records the last commanded pulse."""

    hardware = False

    def __init__(self) -> None:
        self.last_us: dict[int, float] = {}

    def write_us(self, channel: int, microseconds: float) -> None:
        self.last_us[channel] = microseconds

    def release(self, channel: int) -> None:
        self.last_us.pop(channel, None)

    def shutdown(self) -> None:
        self.last_us.clear()


class PCA9685Driver:
    """Thin wrapper around adafruit_pca9685 that speaks raw microseconds."""

    hardware = True

    def __init__(self, i2c_address: int, frequency_hz: int) -> None:
        import board  # type: ignore[import-not-found]
        import busio  # type: ignore[import-not-found]
        from adafruit_pca9685 import PCA9685  # type: ignore[import-not-found]

        self._i2c = busio.I2C(board.SCL, board.SDA)
        self._pca = PCA9685(self._i2c, address=i2c_address)
        self._pca.frequency = frequency_hz
        self._period_us = 1_000_000.0 / frequency_hz

    def write_us(self, channel: int, microseconds: float) -> None:
        duty = int(microseconds / self._period_us * 0xFFFF)
        duty = max(0, min(0xFFFF, duty))
        self._pca.channels[channel].duty_cycle = duty

    def release(self, channel: int) -> None:
        self._pca.channels[channel].duty_cycle = 0

    def shutdown(self) -> None:
        for ch in range(16):
            self.release(ch)
        self._pca.deinit()


def make_driver(i2c_address: int, frequency_hz: int) -> ServoDriver:
    try:
        driver = PCA9685Driver(i2c_address, frequency_hz)
        log.info("PCA9685 attached at 0x%02X @ %d Hz", i2c_address, frequency_hz)
        return driver
    except Exception as exc:
        log.warning("Hardware driver unavailable (%s); using mock driver", exc)
        return MockDriver()
