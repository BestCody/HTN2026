from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from moira.learned_specialists import AcceptedSimulationModels
from moira.simulation_learning import (
    DYNAMICS_FEATURES,
    POLICY_FEATURES,
    generate_simulation_dataset,
    load_simulation_learning_config,
)

CONFIG = Path("config/simulation_learning.json")


def test_learning_config_is_bound_to_recorded_servo_commands() -> None:
    config = load_simulation_learning_config(CONFIG)

    assert config.base_commands == (0.0, 60.0, 123.0)
    assert config.shoulder_commands == (0.0, 45.0, 90.0)
    assert config.gripper_commands == (83.0, 180.0)
    assert config.horizon_s == (2.0, 3.0)
    assert config.mujoco_model.name == "fixed_elbow_learning.xml"


def test_generated_learning_data_has_disjoint_splits_and_lineage(tmp_path: Path) -> None:
    config = load_simulation_learning_config(CONFIG)
    generated = replace(
        config,
        sample_count=1000,
        dataset=tmp_path / "episodes.npz",
        metadata=tmp_path / "episodes.json",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    metadata = generate_simulation_dataset(generated)

    assert metadata["sample_count"] == 1000
    assert metadata["physical_execution_validated"] is False
    assert sum(metadata["split_counts"].values()) == 1000
    assert metadata["policy_features"] == list(POLICY_FEATURES)
    assert metadata["dynamics_features"] == list(DYNAMICS_FEATURES)
    assert json.loads(generated.metadata.read_text())["dataset_sha256"]
    with np.load(generated.dataset, allow_pickle=False) as dataset:
        assert dataset["policy_inputs"].shape == (1000, len(POLICY_FEATURES))
        assert dataset["dynamics_inputs"].shape == (1000, len(DYNAMICS_FEATURES))
        assert set(dataset["split"].tolist()) == {0, 1, 2}


def test_accepted_policy_projection_recovers_reachable_fixed_elbow_pose() -> None:
    config = load_simulation_learning_config(CONFIG)
    models = AcceptedSimulationModels(
        "checkpoints/simulation-learning/current.json",
        config,
    )
    target = models.gripper_position((37.0, 23.0, 180.0))

    projected, error = models.project_commands_to_workspace(
        (123.0, 90.0, 180.0),
        target,
    )

    assert error < 0.002
    assert config.base_commands[0] <= projected[0] <= config.base_commands[-1]
    assert config.shoulder_commands[0] <= projected[1] <= config.shoulder_commands[-1]
