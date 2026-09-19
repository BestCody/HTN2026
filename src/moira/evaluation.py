"""Routing classification, active-joint MSE, and environment rollouts."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from .routing import Router, RoutingDecision, RoutingError
from .runtime import MoIRA


@dataclass(frozen=True)
class RoutingSample:
    instruction: str
    expert_id: str

    def __post_init__(self) -> None:
        for field in ("instruction", "expert_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Routing sample {field} must be a non-empty string")


def load_samples(path: str | Path) -> list[RoutingSample]:
    samples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            sample = RoutingSample(**entry)
            samples.append(sample)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid routing sample on line {line_number}: {exc}") from exc
    return samples


def evaluate_routing(
    router: Router,
    samples: Sequence[RoutingSample],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Macro-F1 across explicitly supplied labels, with invalid outputs as misses.

    Model loading errors propagate. Classification-format errors count as
    incorrect predictions, contribute false negatives, and are reported.
    Timing includes the first model load and description embedding computation.
    """
    if not callable(getattr(router, "route", None)):
        raise TypeError("router must implement route(instruction)")
    if (
        not samples
        or not labels
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Provide samples and non-empty unique evaluation labels")
    if any(not isinstance(sample, RoutingSample) for sample in samples):
        raise TypeError("Every evaluation sample must be a RoutingSample")
    if any(sample.expert_id not in labels for sample in samples):
        raise ValueError("Every ground-truth label must be in the expert pool")
    confusion = {label: {prediction: 0 for prediction in labels} for label in labels}
    invalid_by_label = dict.fromkeys(labels, 0)
    strategy_counts: dict[str, int] = {}
    predictions = []
    started = perf_counter()
    for sample in samples:
        error = None
        decision = None
        try:
            decision = router.route(sample.instruction)
            if not isinstance(decision, RoutingDecision):
                raise TypeError("router must return a RoutingDecision")
            prediction = decision.expert_id
            strategy_counts[decision.strategy] = strategy_counts.get(decision.strategy, 0) + 1
            if prediction not in labels:
                raise RoutingError(f"Unknown predicted expert: {prediction}")
            confusion[sample.expert_id][prediction] += 1
        except RoutingError as exc:
            prediction, error = None, str(exc)
            invalid_by_label[sample.expert_id] += 1
        predictions.append(
            {
                **asdict(sample),
                "predicted": prediction,
                "decision": asdict(decision) if decision is not None else None,
                "error": error,
            }
        )
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        fn += invalid_by_label[label]
        per_class[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "count": len(samples),
        "accuracy": sum(confusion[label][label] for label in labels) / len(samples),
        "macro_f1": sum(per_class.values()) / len(labels),
        "per_class_f1": per_class,
        "confusion_matrix": confusion,
        "invalid_predictions": sum(invalid_by_label.values()),
        "strategy_counts": strategy_counts,
        "elapsed_seconds": perf_counter() - started,
        "predictions": predictions,
    }


def active_joint_mse(
    predicted: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    active_joints: Sequence[int] | None = None,
) -> float:
    """Mean squared error over equally weighted time steps and selected joints."""
    if len(predicted) == 0 or len(predicted) != len(target) or len(target[0]) == 0:
        raise ValueError("Trajectories must be non-empty and have equal shapes")
    width = len(target[0])
    joints = tuple(range(width)) if active_joints is None else tuple(active_joints)
    if (
        not joints
        or len(set(joints)) != len(joints)
        or any(
            not isinstance(joint, int) or isinstance(joint, bool) or not 0 <= joint < width
            for joint in joints
        )
    ):
        raise ValueError("Active joints must be unique valid integer column indices")
    errors = []
    for actual, expected in zip(predicted, target, strict=True):
        if len(actual) != width or len(expected) != width:
            raise ValueError("Trajectories must have identical rectangular shapes")
        for joint in joints:
            a, b = float(actual[joint]), float(expected[joint])
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Active joint values must be finite")
            errors.append((a - b) ** 2)
    return math.fsum(errors) / len(errors)


class Environment(Protocol):
    def reset(self, *, seed: int) -> tuple[Any, dict[str, Any]]: ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...


@dataclass(frozen=True)
class RolloutResult:
    expert_id: str
    seed: int
    steps: int
    total_reward: float
    success: bool
    terminated: bool
    truncated: bool
    actions: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.expert_id, str) or not self.expert_id.strip():
            raise ValueError("Rollout expert_id must be a non-empty string")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("Rollout seed must be an integer")
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 0:
            raise ValueError("Rollout steps must be a nonnegative integer")
        if not isinstance(self.total_reward, (int, float)) or isinstance(self.total_reward, bool):
            raise ValueError("Rollout total_reward must be numeric")
        if not math.isfinite(float(self.total_reward)):
            raise ValueError("Rollout total_reward must be finite")
        if any(
            not isinstance(value, bool) for value in (self.success, self.terminated, self.truncated)
        ):
            raise ValueError("Rollout status fields must be boolean")
        if not isinstance(self.actions, tuple) or len(self.actions) != self.steps:
            raise ValueError("Rollout actions must be a tuple with one entry per step")


def rollout(
    controller: MoIRA,
    environment: Environment,
    instruction: str,
    *,
    seed: int,
    max_steps: int,
    success_fn: Callable[[Environment, Mapping[str, Any]], Any] | None = None,
) -> RolloutResult:
    """Run a Gymnasium-style environment. The caller owns environment cleanup.

    Default success is bool(info['success']); termination alone is not success.
    For LIBERO, adapt its API and supply a task-specific success function.
    Policies must emit one env.step-compatible action; handle chunk queues in
    the policy wrapper and clear them in reset().
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be positive")
    if success_fn is not None and not callable(success_fn):
        raise TypeError("success_fn must be callable or null")
    observation, reset_info = environment.reset(seed=seed)
    if not isinstance(reset_info, Mapping):
        raise TypeError("Environment reset info must be a mapping")
    actions, total_reward = [], 0.0
    success = terminated = truncated = False
    with controller.episode(instruction) as episode:
        for _ in range(max_steps):
            action = episode.act(observation)
            observation, reward, terminated, truncated, info = environment.step(action)
            actions.append(action)
            reward = float(reward)
            if not math.isfinite(reward):
                raise ValueError("Environment rewards must be finite")
            if not isinstance(info, Mapping):
                raise TypeError("Environment step info must be a mapping")
            total_reward += reward
            success = bool(
                success_fn(environment, info) if success_fn else info.get("success", False)
            )
            if success or terminated or truncated:
                break
        else:
            truncated = True
        return RolloutResult(
            episode.decision.expert_id,
            seed,
            len(actions),
            total_reward,
            success,
            bool(terminated),
            bool(truncated),
            tuple(actions),
        )


def success_rate(results: Sequence[RolloutResult]) -> float:
    if not results:
        raise ValueError("At least one rollout is required")
    if any(not isinstance(result, RolloutResult) for result in results):
        raise TypeError("Every result must be a RolloutResult")
    return sum(result.success for result in results) / len(results)
