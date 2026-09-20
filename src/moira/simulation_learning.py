"""Simulation-only learning data for the fixed-elbow physical arm."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POLICY_FEATURES = (
    "current_base_command_deg",
    "current_shoulder_command_deg",
    "current_gripper_command_deg",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "target_gripper_open",
)
POLICY_TARGETS = (
    "goal_base_command_deg",
    "goal_shoulder_command_deg",
    "goal_gripper_command_deg",
)
DYNAMICS_FEATURES = (
    "current_base_command_deg",
    "current_shoulder_command_deg",
    "current_gripper_command_deg",
    "goal_base_command_deg",
    "goal_shoulder_command_deg",
    "goal_gripper_command_deg",
    "payload_kg",
    "horizon_s",
)
DYNAMICS_TARGETS = (
    "future_base_command_deg",
    "future_shoulder_command_deg",
    "future_gripper_command_deg",
    "future_x_m",
    "future_y_m",
    "future_z_m",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two numbers")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in pair) or pair[0] >= pair[1]:
        raise ValueError(f"{name} must be a finite increasing interval")
    return pair


def _anchors(value: object, name: str, count: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} must contain {count} numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class SimulationLearningConfig:
    path: Path
    robot_model: Path
    mujoco_model: Path
    servo_observations: Path
    dataset: Path
    metadata: Path
    checkpoint_dir: Path
    seed: int
    sample_count: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    base_commands: tuple[float, float, float]
    base_joint_degrees: tuple[float, float, float]
    shoulder_commands: tuple[float, float, float]
    shoulder_joint_degrees: tuple[float, float, float]
    gripper_commands: tuple[float, float]
    gripper_joint_radians: tuple[float, float]
    time_constant_s: tuple[float, float]
    actuator_delay_s: tuple[float, float]
    backlash_command_deg: tuple[float, float]
    payload_kg: tuple[float, float]
    payload_shoulder_slowdown: tuple[float, float]
    horizon_s: tuple[float, float]
    training: dict[str, float | int]


def load_simulation_learning_config(path: str | Path) -> SimulationLearningConfig:
    config_path = Path(path).resolve()
    raw = _mapping(json.loads(config_path.read_text(encoding="utf-8")), "config")
    if raw.get("schema_version") != 1:
        raise ValueError("simulation learning schema_version must be 1")
    root = config_path.parent

    def resolved(name: str) -> Path:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a path")
        return (root / value).resolve()

    robot_model = resolved("robot_model")
    mujoco_model = resolved("mujoco_model")
    servo_observations = resolved("servo_observations")
    for name, value in (
        ("robot_model", robot_model),
        ("mujoco_model", mujoco_model),
        ("servo_observations", servo_observations),
    ):
        if not value.is_file():
            raise FileNotFoundError(f"{name} does not exist: {value}")
    seed = raw.get("seed")
    sample_count = raw.get("sample_count")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1000:
        raise ValueError("sample_count must be an integer of at least 1000")
    splits = _mapping(raw.get("split_fractions"), "split_fractions")
    fractions = tuple(
        float(splits.get(name, math.nan))
        for name in ("train", "validation", "test")
    )
    if not all(
        math.isfinite(item) and 0 < item < 1 for item in fractions
    ) or not math.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must be positive and sum to one")
    command = _mapping(raw.get("command_to_joint"), "command_to_joint")
    base = _mapping(command.get("base"), "command_to_joint.base")
    shoulder = _mapping(command.get("shoulder"), "command_to_joint.shoulder")
    gripper = _mapping(command.get("gripper"), "command_to_joint.gripper")
    base_commands = _anchors(base.get("command_deg"), "base command anchors", 3)
    base_joint_degrees = _anchors(base.get("joint_deg"), "base joint anchors", 3)
    shoulder_commands = _anchors(shoulder.get("command_deg"), "shoulder command anchors", 3)
    shoulder_joint_degrees = _anchors(shoulder.get("joint_deg"), "shoulder joint anchors", 3)
    if tuple(sorted(base_commands)) != base_commands:
        raise ValueError("base command anchors must be increasing")
    if tuple(sorted(base_joint_degrees)) != base_joint_degrees:
        raise ValueError("base joint anchors must be increasing")
    if tuple(sorted(shoulder_commands)) != shoulder_commands:
        raise ValueError("shoulder command anchors must be increasing")
    if tuple(sorted(shoulder_joint_degrees, reverse=True)) != shoulder_joint_degrees:
        raise ValueError("shoulder joint anchors must be decreasing")
    gripper_commands = _pair(gripper.get("command_deg"), "gripper command anchors")
    gripper_joint_radians = _anchors(
        gripper.get("simulation_joint_rad"), "gripper joint anchors", 2
    )
    observations = _mapping(
        json.loads(servo_observations.read_text(encoding="utf-8")),
        "servo observations",
    )
    observed = _mapping(observations.get("observations"), "observations")
    if (
        _mapping(observed.get("J1_BASE_YAW"), "J1_BASE_YAW").get("front_command_deg")
        != base_commands[0]
        or _mapping(observed.get("J1_BASE_YAW"), "J1_BASE_YAW").get("back_facing_command_deg")
        != base_commands[-1]
        or _mapping(observed.get("J1_BASE_YAW"), "J1_BASE_YAW").get("left_command_deg")
        != base_commands[1]
    ):
        raise ValueError("base simulation anchors disagree with recorded calibration")
    if (
        _mapping(observed.get("J3_GRIPPER"), "J3_GRIPPER").get("first_closed_command_deg")
        != gripper_commands[0]
        or _mapping(observed.get("J3_GRIPPER"), "J3_GRIPPER").get("maximum_open_command_deg")
        != gripper_commands[1]
    ):
        raise ValueError("gripper simulation anchors disagree with recorded calibration")
    dynamics = _mapping(raw.get("dynamics_randomization"), "dynamics_randomization")
    training = _mapping(raw.get("training"), "training")
    required_training = {
        "hidden_width",
        "hidden_layers",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "policy_max_command_mae_deg",
        "dynamics_max_command_mae_deg",
        "dynamics_max_position_mae_m",
    }
    if set(training) != required_training:
        raise ValueError("training fields do not match the schema")
    numeric_training: dict[str, float | int] = {}
    for name, value in training.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"training.{name} must be finite and positive")
        numeric_training[name] = value
    return SimulationLearningConfig(
        config_path,
        robot_model,
        mujoco_model,
        servo_observations,
        resolved("dataset"),
        resolved("metadata"),
        resolved("checkpoint_dir"),
        seed,
        sample_count,
        *fractions,
        base_commands,
        base_joint_degrees,
        shoulder_commands,
        shoulder_joint_degrees,
        gripper_commands,
        gripper_joint_radians,
        _pair(dynamics.get("time_constant_s"), "time_constant_s"),
        _pair(dynamics.get("actuator_delay_s"), "actuator_delay_s"),
        _pair(dynamics.get("backlash_command_deg"), "backlash_command_deg"),
        _pair(dynamics.get("payload_kg"), "payload_kg"),
        _pair(dynamics.get("payload_shoulder_slowdown"), "payload_shoulder_slowdown"),
        _pair(dynamics.get("horizon_s"), "horizon_s"),
        numeric_training,
    )


def _interpolate(values: Any, source: tuple[float, ...], target: tuple[float, ...], np: Any) -> Any:
    return np.interp(values, np.asarray(source), np.asarray(target))


def command_to_qpos(commands: Any, config: SimulationLearningConfig, np: Any) -> Any:
    values = np.asarray(commands, dtype=np.float64)
    result = np.empty_like(values)
    result[..., 0] = np.deg2rad(
        _interpolate(values[..., 0], config.base_commands, config.base_joint_degrees, np)
    )
    result[..., 1] = np.deg2rad(
        _interpolate(
            values[..., 1],
            config.shoulder_commands,
            config.shoulder_joint_degrees,
            np,
        )
    )
    result[..., 2] = _interpolate(
        values[..., 2], config.gripper_commands, config.gripper_joint_radians, np
    )
    return result


def generate_simulation_dataset(config: SimulationLearningConfig) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("simulation data generation requires MuJoCo and NumPy") from exc
    model = mujoco.MjModel.from_xml_path(str(config.mujoco_model))
    data = mujoco.MjData(model)
    joint_names = ("J1_BASE_YAW", "J2_SHOULDER", "J3_GRIPPER", "J3_GRIPPER_MIRROR")
    addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
    if site_id < 0:
        raise ValueError("MuJoCo model is missing gripper_center")
    rng = np.random.default_rng(config.seed)
    count = config.sample_count

    def uniform(bounds: tuple[float, float], size: int | tuple[int, ...]) -> Any:
        return rng.uniform(bounds[0], bounds[1], size=size)

    lower = np.asarray(
        [config.base_commands[0], config.shoulder_commands[0], config.gripper_commands[0]]
    )
    upper = np.asarray(
        [config.base_commands[-1], config.shoulder_commands[-1], config.gripper_commands[1]]
    )
    current = rng.uniform(lower, upper, size=(count, 3))
    goal = rng.uniform(lower, upper, size=(count, 3))
    target_open = rng.integers(0, 2, size=count).astype(np.float32)
    goal[:, 2] = np.where(target_open > 0.5, config.gripper_commands[1], config.gripper_commands[0])

    def positions(commands: Any) -> Any:
        qpos = command_to_qpos(commands, config, np)
        output = np.empty((commands.shape[0], 3), dtype=np.float32)
        for index, values in enumerate(qpos):
            data.qpos[addresses[0]] = values[0]
            data.qpos[addresses[1]] = values[1]
            data.qpos[addresses[2]] = values[2]
            data.qpos[addresses[3]] = -values[2]
            mujoco.mj_forward(model, data)
            output[index] = data.site_xpos[site_id]
        return output

    goal_positions = positions(goal)
    policy_inputs = np.column_stack((current, goal_positions, target_open)).astype(np.float32)
    policy_targets = goal.astype(np.float32)

    payload = uniform(config.payload_kg, count)
    horizon = uniform(config.horizon_s, count)
    tau = uniform(config.time_constant_s, (count, 3))
    slowdown = uniform(config.payload_shoulder_slowdown, count)
    if config.payload_kg[1] > 0:
        tau[:, 1] *= 1.0 + slowdown * payload / config.payload_kg[1]
    delay = uniform(config.actuator_delay_s, (count, 3))
    backlash = uniform(config.backlash_command_deg, (count, 3))
    delta = goal - current
    effective_delta = np.sign(delta) * np.maximum(np.abs(delta) - backlash, 0.0)
    response_time = np.maximum(horizon[:, None] - delay, 0.0)
    alpha = 1.0 - np.exp(-response_time / tau)
    future = np.clip(current + effective_delta * alpha, lower, upper)
    future_positions = positions(future)
    dynamics_inputs = np.column_stack((current, goal, payload, horizon)).astype(np.float32)
    dynamics_targets = np.column_stack((future, future_positions)).astype(np.float32)

    order = rng.permutation(count)
    split = np.empty(count, dtype=np.uint8)
    train_end = round(count * config.train_fraction)
    validation_end = train_end + round(count * config.validation_fraction)
    split[order[:train_end]] = 0
    split[order[train_end:validation_end]] = 1
    split[order[validation_end:]] = 2
    config.dataset.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=config.dataset.parent, suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            policy_inputs=policy_inputs,
            policy_targets=policy_targets,
            dynamics_inputs=dynamics_inputs,
            dynamics_targets=dynamics_targets,
            split=split,
        )
        os.replace(temporary, config.dataset)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    metadata = {
        "schema_version": 1,
        "scope": "simulation-only fixed-elbow command-space learning",
        "physical_execution_validated": False,
        "robot_model_sha256": sha256_file(config.robot_model),
        "mujoco_model_sha256": sha256_file(config.mujoco_model),
        "servo_observations_sha256": sha256_file(config.servo_observations),
        "dataset_sha256": sha256_file(config.dataset),
        "seed": config.seed,
        "sample_count": count,
        "split_counts": {
            "train": int((split == 0).sum()),
            "validation": int((split == 1).sum()),
            "test": int((split == 2).sum()),
        },
        "policy_features": list(POLICY_FEATURES),
        "policy_targets": list(POLICY_TARGETS),
        "dynamics_features": list(DYNAMICS_FEATURES),
        "dynamics_targets": list(DYNAMICS_TARGETS),
        "limitations": [
            "shoulder command-zero physical angle is an estimate",
            "servo delay, backlash, and response time are domain-randomized assumptions",
            "contact, friction, slip, and force require physical data from the finished arm",
        ],
    }
    config.metadata.parent.mkdir(parents=True, exist_ok=True)
    config.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
