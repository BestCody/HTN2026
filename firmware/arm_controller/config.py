from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class JointConfig(BaseModel):
    channel: int = Field(ge=0, le=15)
    min_deg: float
    max_deg: float
    home_deg: float
    min_us: int = Field(ge=100, le=3000)
    max_us: int = Field(ge=100, le=3000)
    invert: bool = False

    @field_validator("max_deg")
    @classmethod
    def _check_range(cls, v: float, info) -> float:
        min_deg = info.data.get("min_deg")
        if min_deg is not None and v <= min_deg:
            raise ValueError("max_deg must exceed min_deg")
        return v


class ArmConfig(BaseModel):
    i2c_address: int = 0x40
    pwm_frequency: int = 50
    control_rate_hz: float = 50.0
    broadcast_rate_hz: float = 20.0
    max_deg_per_sec: float = 120.0
    joints: dict[str, JointConfig]

    @classmethod
    def load(cls, path: Path | str) -> "ArmConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"
