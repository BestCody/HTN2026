"""Validated configuration and readiness checks for robot-specific simulation training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .calibration import apply_servo_calibration
from .episode_dataset import validate_camera_calibration
from .robot_config import RobotModel

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, name: str, *, allow_zero: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _vector(value: object, size: int, name: str, *, allow_zero: bool) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    return [_number(item, name, allow_zero=allow_zero) for item in value]


def _load(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def create_simulation_training_template(
    robot_model: dict[str, Any], mujoco_path: Path
) -> dict[str, Any]:
    model = RobotModel.from_mapping(robot_model)
    if not model.kinematics_validated:
        raise ValueError("Robot kinematics must be validated before simulation setup")
    root = ET.parse(mujoco_path).getroot()
    mesh_names = [item.attrib["name"] for item in root.findall("asset/mesh")]
    joint_names = [item.attrib["name"] for item in root.findall("worldbody//joint")]
    if not mesh_names or not joint_names:
        raise ValueError("MuJoCo model must contain meshes and joints")
    return {
        "schema_version": SCHEMA_VERSION,
        "robot_model_id": model.model_id,
        "robot_source_sha256": model.source_sha256,
        "mujoco_model_sha256": _sha256(mujoco_path),
        "timestep_s": None,
        "rollout_horizon_s": None,
        "parallel_candidates": None,
        "link_dynamics": {
            name: {
                "mass_kg": None,
                "center_of_mass_m": None,
                "diaginertia_kg_m2": None,
                "friction": None,
            }
            for name in mesh_names
        },
        "joint_dynamics": {
            name: {
                "damping_nms_rad": None,
                "armature_kg_m2": None,
                "backlash_deg": None,
                "max_torque_nm": None,
                "time_constant_s": None,
            }
            for name in joint_names
        },
        "domain_randomization": {
            "mass_fraction": None,
            "friction_fraction": None,
            "actuator_delay_s": None,
            "camera_translation_m": None,
            "camera_rotation_deg": None,
        },
    }


def validate_simulation_training_config(
    value: dict[str, Any], model: RobotModel, mujoco_path: Path
) -> dict[str, Any]:
    root = ET.parse(mujoco_path).getroot()
    expected_meshes = {item.attrib["name"] for item in root.findall("asset/mesh")}
    expected_joints = {item.attrib["name"] for item in root.findall("worldbody//joint")}
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Simulation config schema_version must be {SCHEMA_VERSION}")
    if value.get("robot_model_id") != model.model_id:
        raise ValueError("Simulation config robot_model_id does not match")
    if value.get("robot_source_sha256") != model.source_sha256:
        raise ValueError("Simulation config was created for different robot geometry")
    if value.get("mujoco_model_sha256") != _sha256(mujoco_path):
        raise ValueError("Simulation config was created for a different MuJoCo model")
    timestep = _number(value.get("timestep_s"), "timestep_s")
    horizon = _number(value.get("rollout_horizon_s"), "rollout_horizon_s")
    if not 2 <= horizon <= 3:
        raise ValueError("rollout_horizon_s must be between 2 and 3 seconds")
    candidates = value.get("parallel_candidates")
    if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates < 2:
        raise ValueError("parallel_candidates must be an integer of at least 2")
    links = value.get("link_dynamics")
    if not isinstance(links, dict) or set(links) != expected_meshes:
        raise ValueError("link_dynamics must exactly match the MuJoCo mesh assets")
    for name, raw in links.items():
        if not isinstance(raw, dict):
            raise TypeError(f"link_dynamics.{name} must be an object")
        _number(raw.get("mass_kg"), f"link_dynamics.{name}.mass_kg")
        center = raw.get("center_of_mass_m")
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError(f"link_dynamics.{name}.center_of_mass_m must contain 3 values")
        if any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(item)
            for item in center
        ):
            raise ValueError(f"link_dynamics.{name}.center_of_mass_m must be finite")
        _vector(
            raw.get("diaginertia_kg_m2"),
            3,
            f"link_dynamics.{name}.diaginertia_kg_m2",
            allow_zero=False,
        )
        _vector(
            raw.get("friction"),
            3,
            f"link_dynamics.{name}.friction",
            allow_zero=True,
        )
    joints = value.get("joint_dynamics")
    if not isinstance(joints, dict) or set(joints) != expected_joints:
        raise ValueError("joint_dynamics must exactly match the MuJoCo joints")
    for name, raw in joints.items():
        if not isinstance(raw, dict):
            raise TypeError(f"joint_dynamics.{name} must be an object")
        for field, allow_zero in (
            ("damping_nms_rad", True),
            ("armature_kg_m2", True),
            ("backlash_deg", True),
            ("max_torque_nm", False),
            ("time_constant_s", False),
        ):
            _number(raw.get(field), f"joint_dynamics.{name}.{field}", allow_zero=allow_zero)
    randomization = value.get("domain_randomization")
    if not isinstance(randomization, dict):
        raise TypeError("domain_randomization must be an object")
    for field in (
        "mass_fraction",
        "friction_fraction",
        "actuator_delay_s",
        "camera_translation_m",
        "camera_rotation_deg",
    ):
        _number(randomization.get(field), f"domain_randomization.{field}", allow_zero=True)
    result = dict(value)
    result["timestep_s"] = timestep
    result["rollout_horizon_s"] = horizon
    return result


def simulation_training_preflight(
    robot_model_path: Path,
    mujoco_path: Path,
    servo_calibration_path: Path,
    camera_calibration_path: Path,
    simulation_config_path: Path,
) -> dict[str, Any]:
    issues = []
    model_value = _load(robot_model_path, "robot model")
    model = RobotModel.from_mapping(model_value)
    if not model.kinematics_validated:
        issues.append("CAD-derived kinematics validation")
    if not model.collision_geometry_validated:
        issues.append("physical collision-geometry validation")
    if not model.payload_validated or model.payload_limit_kg is None:
        issues.append("physical payload validation")
    checks = (
        (
            "complete servo calibration",
            lambda: apply_servo_calibration(
                model_value, _load(servo_calibration_path, "servo calibration")
            ),
        ),
        (
            "complete camera calibration",
            lambda: validate_camera_calibration(
                _load(camera_calibration_path, "camera calibration"), model
            ),
        ),
        (
            "complete simulation dynamics configuration",
            lambda: validate_simulation_training_config(
                _load(simulation_config_path, "simulation config"), model, mujoco_path
            ),
        ),
    )
    details = {}
    for label, check in checks:
        try:
            check()
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            issues.append(label)
            details[label] = str(exc)
    return {
        "robot_model_id": model.model_id,
        "ready": not issues,
        "issues": issues,
        "details": details,
        "mujoco_model": str(mujoco_path),
    }
