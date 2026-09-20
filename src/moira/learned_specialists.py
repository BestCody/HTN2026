"""Typed runtime adapters for accepted fixed-elbow simulation checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from threading import RLock
from typing import Any

from .physical import (
    KinematicsInput,
    KinematicsSolution,
    PredictedState,
    WorldModelInput,
    WorldModelPrediction,
)
from .simulation_learning import (
    SimulationLearningConfig,
    command_to_qpos,
    sha256_file,
)


class _NormalizedMLP:
    def __init__(self, path: Path, section: dict[str, Any]) -> None:
        try:
            import numpy as np
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "learned simulation specialists require NumPy, PyTorch, and safetensors"
            ) from exc
        architecture = section.get("architecture")
        preprocessing = section.get("preprocessing")
        features = section.get("features")
        targets = section.get("targets")
        if (
            not isinstance(architecture, dict)
            or not isinstance(preprocessing, dict)
            or not isinstance(features, list)
            or not isinstance(targets, list)
        ):
            raise ValueError("checkpoint model contract is incomplete")
        width = architecture.get("hidden_width")
        depth = architecture.get("hidden_layers")
        if not isinstance(width, int) or not isinstance(depth, int) or min(width, depth) < 1:
            raise ValueError("checkpoint MLP architecture is invalid")
        layers: list[Any] = []
        previous = len(features)
        for _ in range(depth):
            layers.extend((torch.nn.Linear(previous, width), torch.nn.SiLU()))
            previous = width
        layers.append(torch.nn.Linear(previous, len(targets)))
        self.model = torch.nn.Sequential(*layers)
        self.model.load_state_dict(load_file(path, device="cpu"), strict=True)
        self.model.eval()
        self.np = np
        self.torch = torch
        self.input_mean = self._array(preprocessing, "input_mean", len(features))
        self.input_std = self._array(preprocessing, "input_std", len(features))
        self.target_mean = self._array(preprocessing, "target_mean", len(targets))
        self.target_std = self._array(preprocessing, "target_std", len(targets))
        if (self.input_std <= 0).any() or (self.target_std <= 0).any():
            raise ValueError("checkpoint normalization scales must be positive")
        self._lock = RLock()

    def _array(self, value: dict[str, Any], name: str, size: int) -> Any:
        result = self.np.asarray(value.get(name), dtype=self.np.float32)
        if result.shape != (size,) or not self.np.isfinite(result).all():
            raise ValueError(f"checkpoint {name} is invalid")
        return result

    def predict(self, features: tuple[float, ...]) -> tuple[float, ...]:
        values = self.np.asarray(features, dtype=self.np.float32)
        if values.shape != self.input_mean.shape or not self.np.isfinite(values).all():
            raise ValueError("learned-specialist input does not match its feature contract")
        normalized = (values - self.input_mean) / self.input_std
        with self._lock, self.torch.inference_mode():
            output = self.model(self.torch.from_numpy(normalized).unsqueeze(0))[0].numpy()
        prediction = output * self.target_std + self.target_mean
        if not self.np.isfinite(prediction).all():
            raise RuntimeError("learned specialist returned non-finite values")
        return tuple(float(item) for item in prediction)


class AcceptedSimulationModels:
    """Load only an accepted, hash-bound simulation checkpoint."""

    def __init__(
        self,
        current_path: str | Path,
        config: SimulationLearningConfig,
    ) -> None:
        pointer_path = Path(current_path).resolve()
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        report_value = pointer.get("report")
        if not isinstance(report_value, str) or not report_value.strip():
            raise ValueError("simulation checkpoint pointer is missing its report")
        report_path = Path(report_value).resolve()
        root = pointer_path.parent.resolve()
        if root not in report_path.parents:
            raise ValueError("simulation checkpoint report escapes its checkpoint directory")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("accepted") is not True:
            raise ValueError("simulation checkpoint was not accepted")
        if report.get("physical_execution_validated") is not False:
            raise ValueError("simulation checkpoint physical-validation flag is invalid")
        expected = {
            "robot_model_sha256": sha256_file(config.robot_model),
            "mujoco_model_sha256": sha256_file(config.mujoco_model),
            "servo_observations_sha256": sha256_file(config.servo_observations),
            "dataset_sha256": sha256_file(config.dataset),
        }
        for name, digest in expected.items():
            if report.get(name) != digest:
                raise ValueError(f"simulation checkpoint {name} lineage does not match")
        run_dir = report_path.parent
        waypoint_path = run_dir / "waypoint_policy.safetensors"
        dynamics_path = run_dir / "short_horizon_dynamics.safetensors"
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("simulation checkpoint is missing artifact hashes")
        for path, field in (
            (waypoint_path, "waypoint_policy_sha256"),
            (dynamics_path, "short_horizon_dynamics_sha256"),
        ):
            if not path.is_file() or artifacts.get(field) != sha256_file(path):
                raise ValueError(f"simulation checkpoint artifact failed validation: {field}")
        self.config = config
        self.report = report
        self.run_id = str(report["run_id"])
        self.waypoint = _NormalizedMLP(waypoint_path, report["policy"])
        self.dynamics = _NormalizedMLP(dynamics_path, report["dynamics"])
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("learned waypoint validation requires MuJoCo") from exc
        self._mujoco = mujoco
        self._mujoco_model = mujoco.MjModel.from_xml_path(str(config.mujoco_model))
        self._mujoco_data = mujoco.MjData(self._mujoco_model)
        names = ("J1_BASE_YAW", "J2_SHOULDER", "J3_GRIPPER", "J3_GRIPPER_MIRROR")
        joint_ids = tuple(
            mujoco.mj_name2id(
                self._mujoco_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in names
        )
        if any(joint_id < 0 for joint_id in joint_ids):
            raise ValueError("simulation checkpoint MuJoCo joint contract is incomplete")
        self._joint_addresses = tuple(
            int(self._mujoco_model.jnt_qposadr[joint_id]) for joint_id in joint_ids
        )
        self._site_id = mujoco.mj_name2id(
            self._mujoco_model,
            mujoco.mjtObj.mjOBJ_SITE,
            "gripper_center",
        )
        if self._site_id < 0:
            raise ValueError("simulation checkpoint MuJoCo model lacks gripper_center")
        self._mujoco_lock = RLock()

    def qpos_to_commands(
        self, base_rad: float, shoulder_rad: float, gripper_command: float
    ) -> tuple[float, float, float]:
        np = self.waypoint.np
        config = self.config
        base = float(
            np.interp(
                math.degrees(base_rad),
                config.base_joint_degrees,
                config.base_commands,
            )
        )
        shoulder = float(
            np.interp(
                math.degrees(shoulder_rad),
                config.shoulder_joint_degrees[::-1],
                config.shoulder_commands[::-1],
            )
        )
        return base, shoulder, gripper_command

    def commands_to_qpos(self, commands: tuple[float, float, float]) -> tuple[float, ...]:
        values = command_to_qpos(
            self.waypoint.np.asarray([commands]), self.config, self.waypoint.np
        )[0]
        return tuple(float(item) for item in values)

    def gripper_position(
        self, commands: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        qpos = self.commands_to_qpos(commands)
        with self._mujoco_lock:
            return self._gripper_position_locked(qpos)

    def _gripper_position_locked(
        self, qpos: tuple[float, ...]
    ) -> tuple[float, float, float]:
        for address, value in zip(
            self._joint_addresses,
            (qpos[0], qpos[1], qpos[2], -qpos[2]),
            strict=True,
        ):
            self._mujoco_data.qpos[address] = value
        self._mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        return tuple(float(item) for item in self._mujoco_data.site_xpos[self._site_id])

    def project_commands_to_workspace(
        self,
        initial: tuple[float, float, float],
        target_position: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], float]:
        """Project a learned proposal onto the calibrated fixed-elbow workspace.

        The network remains the action proposer.  A deterministic search through
        the two installed actuator ranges then minimizes Cartesian error against
        the hash-bound MuJoCo twin.  This removes regression approximation error
        without allowing commands outside the recorded servo envelope.
        """

        if len(initial) != 3 or len(target_position) != 3:
            raise ValueError("workspace projection requires three command and target values")
        if not all(math.isfinite(value) for value in (*initial, *target_position)):
            raise ValueError("workspace projection values must be finite")
        base_low = self.config.base_commands[0]
        base_high = self.config.base_commands[-1]
        shoulder_low = self.config.shoulder_commands[0]
        shoulder_high = self.config.shoulder_commands[-1]
        gripper_low, gripper_high = self.config.gripper_commands
        gripper = min(max(initial[2], gripper_low), gripper_high)
        seed = (
            min(max(initial[0], base_low), base_high),
            min(max(initial[1], shoulder_low), shoulder_high),
            gripper,
        )

        def command_grid(low: float, high: float, intervals: int) -> tuple[float, ...]:
            step = (high - low) / intervals
            return tuple(low + step * index for index in range(intervals + 1))

        # A small global grid prevents a poor neural seed from trapping the
        # projection on the wrong side of the yaw arc.  The following local
        # levels recover sub-degree command precision.
        coarse_bases = command_grid(base_low, base_high, 32)
        coarse_shoulders = command_grid(shoulder_low, shoulder_high, 16)
        with self._mujoco_lock:
            best = seed
            best_error = math.inf

            def evaluate(base: float, shoulder: float) -> None:
                nonlocal best, best_error
                candidate = (base, shoulder, gripper)
                position = self._gripper_position_locked(self.commands_to_qpos(candidate))
                error = math.dist(position, target_position)
                if error < best_error:
                    best = candidate
                    best_error = error

            evaluate(seed[0], seed[1])
            for base in coarse_bases:
                for shoulder in coarse_shoulders:
                    evaluate(base, shoulder)
            for step in (2.0, 1.0, 0.5, 0.25):
                center_base, center_shoulder, _ = best
                for base_delta in (-step, 0.0, step):
                    for shoulder_delta in (-step, 0.0, step):
                        evaluate(
                            min(max(center_base + base_delta, base_low), base_high),
                            min(
                                max(center_shoulder + shoulder_delta, shoulder_low),
                                shoulder_high,
                            ),
                        )
        return best, best_error


class LearnedWaypointKinematics:
    """Use the accepted waypoint checkpoint as the fixed-elbow action policy."""

    def __init__(
        self,
        models: AcceptedSimulationModels,
        *,
        max_gripper_width_m: float,
        position_tolerance_m: float,
    ) -> None:
        if not 0 < max_gripper_width_m <= 0.5:
            raise ValueError("max_gripper_width_m must be in (0, 0.5]")
        self.models = models
        self.max_gripper_width_m = max_gripper_width_m
        if not 0 < position_tolerance_m <= 0.1:
            raise ValueError("position_tolerance_m must be in (0, 0.1]")
        self.position_tolerance_m = position_tolerance_m

    def _gripper_command(self, width_m: float) -> float:
        low, high = self.models.config.gripper_commands
        return high if width_m >= self.max_gripper_width_m / 2 else low

    def run(self, request: KinematicsInput) -> KinematicsSolution:
        if not isinstance(request, KinematicsInput):
            raise TypeError("learned waypoint policy expects KinematicsInput")
        if request.robot_state is None:
            raise ValueError("learned waypoint policy requires observed robot state")
        targets: dict[str, dict[str, tuple[float, ...]]] = {}
        reasons: list[str] = []
        current_by_arm: dict[str, tuple[float, float, float]] = {}
        for arm, joints in request.robot_state.joint_positions.items():
            if len(joints) != 2:
                reasons.append(f"{arm}: fixed-elbow state must contain two joints")
                continue
            width = request.robot_state.gripper_widths_m.get(arm)
            if width is None:
                reasons.append(f"{arm}: gripper state is missing")
                continue
            current_by_arm[arm] = self.models.qpos_to_commands(
                float(joints[0]),
                float(joints[1]),
                self._gripper_command(float(width)),
            )
        base_bounds = (
            self.models.config.base_commands[0],
            self.models.config.base_commands[-1],
        )
        shoulder_bounds = (
            self.models.config.shoulder_commands[0],
            self.models.config.shoulder_commands[-1],
        )
        gripper_bounds = self.models.config.gripper_commands
        for chunk in request.policy.chunks:
            per_arm: dict[str, tuple[float, ...]] = {}
            for arm in chunk.arms:
                current = current_by_arm.get(arm)
                if arm != "left":
                    reasons.append(f"{chunk.id}/{arm}: arm is not installed")
                    continue
                if current is None:
                    continue
                x, y, z = chunk.target_pose[:3]
                target_open = float(
                    chunk.gripper_width_m >= self.max_gripper_width_m / 2
                )
                raw_prediction = self.models.waypoint.predict(
                    (*current, float(x), float(y), float(z), target_open)
                )
                bounds = (base_bounds, shoulder_bounds, gripper_bounds)
                # A regression head is not guaranteed to land exactly on a hard
                # endpoint (the gripper targets are deliberately 83 or 180).
                # Project every learned action onto the calibrated actuator set
                # before converting it to joint space.  The subsequent MuJoCo
                # position-error check still rejects a projected command that
                # cannot reach the requested pose.
                bounded_prediction = tuple(
                    min(max(value, limits[0]), limits[1])
                    for value, limits in zip(raw_prediction, bounds, strict=True)
                )
                prediction, error = self.models.project_commands_to_workspace(
                    bounded_prediction,
                    (float(x), float(y), float(z)),
                )
                qpos = self.models.commands_to_qpos(prediction)
                if error > self.position_tolerance_m:
                    reasons.append(
                        f"{chunk.id}/{arm}: target is {error:.3f} m outside the learned arc"
                    )
                    continue
                per_arm[arm] = (qpos[0], qpos[1])
                current_by_arm[arm] = prediction
            targets[chunk.id] = per_arm
        return KinematicsSolution(
            request.policy.plan_id,
            targets,
            not reasons,
            tuple(dict.fromkeys(reasons)),
        )


class LearnedForwardDynamics:
    """Predict the 2–3 second fixed-elbow actuator state from a routed trajectory."""

    def __init__(
        self,
        models: AcceptedSimulationModels,
        *,
        max_gripper_width_m: float,
    ) -> None:
        self.models = models
        self.max_gripper_width_m = max_gripper_width_m

    def _gripper_command(self, width_m: float) -> float:
        low, high = self.models.config.gripper_commands
        return high if width_m >= self.max_gripper_width_m / 2 else low

    def run(self, request: WorldModelInput) -> WorldModelPrediction:
        if not isinstance(request, WorldModelInput):
            raise TypeError("learned dynamics expects WorldModelInput")
        arm = "left"
        first = request.trajectory.points[0]
        last = request.trajectory.points[-1]
        current_q = first.joint_positions.get(arm)
        goal_q = last.joint_positions.get(arm)
        if current_q is None or goal_q is None or len(current_q) != 2 or len(goal_q) != 2:
            raise ValueError("learned dynamics requires a two-joint left-arm trajectory")
        current_width = first.gripper_widths_m.get(arm)
        goal_width = last.gripper_widths_m.get(arm)
        if current_width is None or goal_width is None:
            raise ValueError("learned dynamics requires gripper widths")
        current = self.models.qpos_to_commands(
            current_q[0], current_q[1], self._gripper_command(current_width)
        )
        goal = self.models.qpos_to_commands(
            goal_q[0], goal_q[1], self._gripper_command(goal_width)
        )
        target_ids = {
            chunk.target_object_id
            for chunk in request.policy.chunks
            if chunk.target_object_id is not None
        }
        payload = max(
            (
                item.estimated_mass_kg or 0.0
                for item in request.world.objects
                if item.id in target_ids
            ),
            default=0.0,
        )
        payload = min(
            max(payload, self.models.config.payload_kg[0]),
            self.models.config.payload_kg[1],
        )
        prediction = self.models.dynamics.predict(
            (*current, *goal, payload, request.horizon_seconds)
        )
        future_commands = prediction[:3]
        future_q = self.models.commands_to_qpos(future_commands)
        spans = tuple(
            high - low
            for low, high in (
                (
                    self.models.config.base_commands[0],
                    self.models.config.base_commands[-1],
                ),
                (
                    self.models.config.shoulder_commands[0],
                    self.models.config.shoulder_commands[-1],
                ),
                self.models.config.gripper_commands,
            )
        )
        tracking_error = sum(
            abs(value - target) / span
            for value, target, span in zip(future_commands, goal, spans, strict=True)
        ) / len(spans)
        risk = min(1.0, tracking_error)
        risks = (
            ("forward-dynamics: predicted actuator tracking error",)
            if tracking_error > 0.10
            else ()
        )
        poses = {
            item.id: (*item.position_m, 0.0, 0.0, 0.0, 1.0)
            for item in request.world.objects
        }
        state = PredictedState(
            request.horizon_seconds,
            poses,
            {arm: (future_q[0], future_q[1])},
            {arm: 0.0},
            0.0,
            0.0,
        )
        return WorldModelPrediction(
            request.policy.plan_id,
            "forward-dynamics",
            (state,),
            max(0.0, 1.0 - risk),
            min(
                1.0,
                max(
                    self.models.report["dynamics"]["metrics"]["test_mae_per_target"][
                        3:
                    ]
                )
                / 0.01,
            ),
            risks,
        )
