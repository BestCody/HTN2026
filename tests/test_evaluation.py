import json

import pytest

from moira import Expert, ExpertRegistry, InMemoryServer, MoIRA, RoutingDecision, RoutingError
from moira.cli import main
from moira.demo import MovePolicy
from moira.evaluation import (
    RolloutResult,
    RoutingSample,
    active_joint_mse,
    evaluate_routing,
    load_samples,
    rollout,
    success_rate,
)


class ScriptedRouter:
    def route(self, instruction):
        if instruction == "invalid":
            raise RoutingError("invalid response")
        return RoutingDecision(instruction, "test")


def test_macro_f1_counts_invalid_as_false_negative():
    report = evaluate_routing(
        ScriptedRouter(),
        [
            RoutingSample("a", "a"),
            RoutingSample("b", "a"),
            RoutingSample("b", "b"),
            RoutingSample("invalid", "b"),
        ],
        ["a", "b"],
    )
    assert report["accuracy"] == 0.5
    assert report["per_class_f1"] == pytest.approx({"a": 2 / 3, "b": 0.5})
    assert report["macro_f1"] == pytest.approx(7 / 12)
    assert report["invalid_predictions"] == 1
    assert report["strategy_counts"] == {"test": 3}
    assert report["predictions"][0]["decision"]["strategy"] == "test"


def test_active_joints_only():
    assert active_joint_mse([[2, 100], [4, 100]], [[1, 0], [2, 0]], [0]) == 2.5


@pytest.mark.parametrize(
    "predicted,target,joints",
    [
        ([], [], None),
        ([[1]], [[1], [2]], None),
        ([[1, 2]], [[1]], None),
        ([[1]], [[1]], []),
        ([[1]], [[1]], [-1]),
        ([[1]], [[1]], [0, 0]),
        ([[float("nan")]], [[1]], None),
    ],
)
def test_invalid_mse_shapes_and_masks(predicted, target, joints):
    with pytest.raises(ValueError):
        active_joint_mse(predicted, target, joints)


class NeverSuccessful:
    def reset(self, *, seed):
        return 0, {}

    def step(self, action):
        return 0, 1.0, False, False, {}


def test_rollout_enforces_limit_and_does_not_infer_success_from_reward():
    controller = MoIRA(
        ExpertRegistry([Expert("a", "s", "a")]),
        ScriptedRouter(),
        InMemoryServer({"a": MovePolicy(1)}),
    )
    result = rollout(controller, NeverSuccessful(), "a", seed=3, max_steps=2)
    assert result.truncated and not result.success
    assert result.steps == 2 and result.total_reward == 2
    assert success_rate([result]) == 0


def test_rollout_rejects_nonfinite_rewards_and_invalid_metadata():
    class InvalidReward(NeverSuccessful):
        def step(self, action):
            return 0, float("nan"), False, False, {}

    controller = MoIRA(
        ExpertRegistry([Expert("a", "s", "a")]),
        ScriptedRouter(),
        InMemoryServer({"a": MovePolicy(1)}),
    )
    with pytest.raises(ValueError, match="finite"):
        rollout(controller, InvalidReward(), "a", seed=3, max_steps=2)
    with pytest.raises(ValueError, match="actions"):
        RolloutResult("a", 1, 1, 0.0, False, False, True, ())
    with pytest.raises(TypeError, match="RolloutResult"):
        success_rate([object()])


def test_jsonl_validation_and_cli(tmp_path, capsys):
    assert main(["demo"]) == 0
    assert json.loads(capsys.readouterr().out)["success"]
    path = tmp_path / "samples.jsonl"
    path.write_text('{"instruction": "task", "expert_id": "a"}\n\n')
    assert len(load_samples(path)) == 1
    path.write_text("{}")
    with pytest.raises(ValueError, match="line 1"):
        load_samples(path)
