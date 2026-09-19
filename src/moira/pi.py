"""Raspberry Pi 4B runtime limits and host inspection."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PiRuntimeProfile:
    component_ram_budget_mb: int
    simulation_workers: int = 2
    simulation_horizon_seconds: float = 2.5
    simulation_step_seconds: float = 0.1
    max_candidate_plans: int = 3
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 10
    cloud_timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        integers = (
            self.component_ram_budget_mb,
            self.simulation_workers,
            self.max_candidate_plans,
            self.camera_width,
            self.camera_height,
            self.camera_fps,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers
        ):
            raise ValueError("Pi integer limits must be positive")
        if self.simulation_workers > 2:
            raise ValueError("Pi 4B simulation_workers is capped at 2")
        if self.max_candidate_plans > 4:
            raise ValueError("Pi 4B max_candidate_plans is capped at 4")
        if not 2 <= self.simulation_horizon_seconds <= 3:
            raise ValueError("Simulation horizon must be between 2 and 3 seconds")
        for value in (self.simulation_step_seconds, self.cloud_timeout_seconds):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Pi timing limits must be finite and positive")
        if self.simulation_step_seconds > self.simulation_horizon_seconds:
            raise ValueError("Simulation step cannot exceed its horizon")
        if self.camera_width * self.camera_height > 1280 * 720 or self.camera_fps > 30:
            raise ValueError("Default Pi camera processing is capped at 720p30")

    @classmethod
    def for_pi4(cls, total_memory_mb: int | None = None) -> PiRuntimeProfile:
        total_memory_mb = detected_memory_mb() if total_memory_mb is None else total_memory_mb
        if not isinstance(total_memory_mb, int) or isinstance(total_memory_mb, bool):
            raise TypeError("total_memory_mb must be an integer")
        if total_memory_mb < 768:
            raise ValueError("Raspberry Pi runtime requires at least 768 MB detected RAM")
        # Leave the majority of RAM to Raspberry Pi OS, camera buffers, and I/O.
        budget = max(192, min(1536, int(total_memory_mb * 0.35)))
        return cls(component_ram_budget_mb=budget)


@dataclass(frozen=True)
class PiHost:
    model: str
    total_memory_mb: int
    cpu_count: int
    temperature_c: float | None

    @property
    def is_pi4_model_b(self) -> bool:
        return "Raspberry Pi 4 Model B" in self.model

    def require_pi4(self) -> None:
        if not self.is_pi4_model_b:
            raise RuntimeError(f"Expected Raspberry Pi 4 Model B, detected: {self.model}")
        if self.cpu_count < 4:
            raise RuntimeError("Raspberry Pi 4B runtime requires all four CPU cores")


def detected_memory_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 1024


def inspect_host() -> PiHost:
    try:
        model = Path("/proc/device-tree/model").read_bytes().rstrip(b"\0").decode("utf-8")
    except OSError:
        model = os.environ.get("MOIRA_DEVICE_MODEL", "non-Raspberry-Pi development host")
    temperature = None
    try:
        temperature = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        pass
    return PiHost(model, detected_memory_mb(), os.cpu_count() or 1, temperature)
