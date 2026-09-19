from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .config import ArmConfig, JointConfig
from .servo_driver import ServoDriver


def _angle_to_us(angle_deg: float, cfg: JointConfig) -> float:
    span_deg = cfg.max_deg - cfg.min_deg
    frac = (angle_deg - cfg.min_deg) / span_deg
    if cfg.invert:
        frac = 1.0 - frac
    return cfg.min_us + frac * (cfg.max_us - cfg.min_us)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class ArmSnapshot:
    enabled: bool
    hardware: bool
    current: dict[str, float]
    target: dict[str, float]
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    modes: dict[str, str] = field(default_factory=dict)
    speeds: dict[str, float] = field(default_factory=dict)


class ArmState:
    """Owns the joint targets, smooths motion, and writes to the driver."""

    def __init__(self, config: ArmConfig, driver: ServoDriver) -> None:
        self._config = config
        self._driver = driver
        self._enabled = False
        self._current: dict[str, float] = {
            name: joint.home_deg for name, joint in config.joints.items()
        }
        self._target: dict[str, float] = dict(self._current)
        self._speed: dict[str, float] = {name: 0.0 for name in config.joints}
        self._lock = asyncio.Lock()
        self._last_tick = time.monotonic()
        self._dirty = True
        # A PCA9685 can retain its last duty cycle across a controller restart.
        # Start safely: no joint is driven until the user explicitly enables it.
        for joint in config.joints.values():
            self._driver.release(joint.channel)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def set_enabled(self, enabled: bool) -> None:
        async with self._lock:
            if enabled == self._enabled:
                return
            self._enabled = enabled
            if not enabled:
                for joint in self._config.joints.values():
                    self._driver.release(joint.channel)
                for name in self._speed:
                    self._speed[name] = 0.0
            else:
                self._dirty = True

    async def set_speed(self, joint_name: str, speed_percent: float) -> None:
        """Set a continuous servo's signed speed from -100 to 100 percent."""
        cfg = self._config.joints.get(joint_name)
        if cfg is None:
            raise KeyError(f"unknown joint: {joint_name}")
        if cfg.mode != "continuous":
            raise ValueError(f"joint is not continuous: {joint_name}")
        async with self._lock:
            self._speed[joint_name] = _clamp(speed_percent, -100.0, 100.0)
            self._dirty = True

    async def set_target(self, joint_name: str, degrees: float) -> None:
        cfg = self._config.joints.get(joint_name)
        if cfg is None:
            raise KeyError(f"unknown joint: {joint_name}")
        clamped = _clamp(degrees, cfg.min_deg, cfg.max_deg)
        async with self._lock:
            self._target[joint_name] = clamped
            self._dirty = True

    async def set_targets(self, targets: dict[str, float]) -> None:
        async with self._lock:
            for name, deg in targets.items():
                cfg = self._config.joints.get(name)
                if cfg is None:
                    continue
                self._target[name] = _clamp(deg, cfg.min_deg, cfg.max_deg)
            self._dirty = True

    async def home(self) -> None:
        homes = {name: cfg.home_deg for name, cfg in self._config.joints.items()}
        await self.set_targets(homes)

    def snapshot(self) -> ArmSnapshot:
        return ArmSnapshot(
            enabled=self._enabled,
            hardware=self._driver.hardware,
            current=dict(self._current),
            target=dict(self._target),
            limits={n: (c.min_deg, c.max_deg) for n, c in self._config.joints.items()},
            modes={n: c.mode for n, c in self._config.joints.items()},
            speeds=dict(self._speed),
        )

    async def tick(self) -> None:
        """Advance current toward target by the rate-limited step, write servos."""
        now = time.monotonic()
        dt = max(0.0, now - self._last_tick)
        self._last_tick = now
        if not self._enabled:
            return
        step = self._config.max_deg_per_sec * dt
        async with self._lock:
            moved = False
            for name, cfg in self._config.joints.items():
                if cfg.mode == "continuous":
                    speed = self._speed[name]
                    if cfg.invert:
                        speed = -speed
                    pulse = cfg.neutral_us + speed / 100 * cfg.max_speed_us
                    self._driver.write_us(cfg.channel, pulse)
                    moved = True
                    continue
                cur = self._current[name]
                tgt = self._target[name]
                delta = tgt - cur
                if abs(delta) <= step:
                    new = tgt
                else:
                    new = cur + step * (1 if delta > 0 else -1)
                if new != cur or self._dirty:
                    self._current[name] = new
                    self._driver.write_us(cfg.channel, _angle_to_us(new, cfg))
                    moved = True
            if not moved and not self._dirty:
                return
            self._dirty = False

    def shutdown(self) -> None:
        self._driver.shutdown()
