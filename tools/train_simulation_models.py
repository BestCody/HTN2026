"""Train fixed-elbow waypoint and short-horizon dynamics MLPs on the RTX GPU."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.simulation_learning import (  # noqa: E402
    DYNAMICS_FEATURES,
    DYNAMICS_TARGETS,
    POLICY_FEATURES,
    POLICY_TARGETS,
    load_simulation_learning_config,
    sha256_file,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _train_one(
    *,
    name: str,
    inputs: Any,
    targets: Any,
    split: Any,
    training: dict[str, float | int],
    device: Any,
    torch: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    train_mask = split == 0
    validation_mask = split == 1
    test_mask = split == 2
    input_mean = inputs[train_mask].mean(axis=0)
    input_std = inputs[train_mask].std(axis=0).clip(min=1e-6)
    target_mean = targets[train_mask].mean(axis=0)
    target_std = targets[train_mask].std(axis=0).clip(min=1e-6)
    normalized_inputs = (inputs - input_mean) / input_std
    normalized_targets = (targets - target_mean) / target_std

    layers: list[Any] = []
    width = int(training["hidden_width"])
    previous = inputs.shape[1]
    for _ in range(int(training["hidden_layers"])):
        layers.extend((nn.Linear(previous, width), nn.SiLU()))
        previous = width
    layers.append(nn.Linear(previous, targets.shape[1]))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_function = nn.MSELoss()
    x_train = torch.from_numpy(normalized_inputs[train_mask]).float()
    y_train = torch.from_numpy(normalized_targets[train_mask]).float()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(20260920),
    )
    x_validation = torch.from_numpy(normalized_inputs[validation_mask]).float().to(device)
    y_validation = torch.from_numpy(normalized_targets[validation_mask]).float().to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model(batch_x)
                loss = loss_function(prediction, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * batch_x.shape[0]
            seen += batch_x.shape[0]
        model.eval()
        with torch.inference_mode():
            validation_loss = float(loss_function(model(x_validation), y_validation))
        train_loss /= seen
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(
                json.dumps(
                    {
                        "model": name,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                    }
                ),
                flush=True,
            )
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    x_test = torch.from_numpy(normalized_inputs[test_mask]).float().to(device)
    with torch.inference_mode():
        normalized_prediction = model(x_test).float().cpu().numpy()
    prediction = normalized_prediction * target_std + target_mean
    absolute_error = abs(prediction - targets[test_mask])
    metrics = {
        "best_validation_mse_normalized": best_loss,
        "test_mae_per_target": absolute_error.mean(axis=0).tolist(),
        "test_rmse_per_target": ((absolute_error**2).mean(axis=0) ** 0.5).tolist(),
        "test_samples": int(test_mask.sum()),
        "epochs": int(training["epochs"]),
        "history": history,
    }
    preprocessing = {
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
    }
    return model, metrics, preprocessing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "simulation_learning.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        import numpy as np
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("training requires NumPy, PyTorch, and safetensors") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but PyTorch cannot access the GPU")
    device = torch.device(args.device)
    torch.manual_seed(20260920)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260920)
    config = load_simulation_learning_config(args.config)
    metadata = json.loads(config.metadata.read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != sha256_file(config.dataset):
        raise ValueError("dataset does not match its lineage metadata")
    with np.load(config.dataset, allow_pickle=False) as dataset:
        policy_inputs = dataset["policy_inputs"]
        policy_targets = dataset["policy_targets"]
        dynamics_inputs = dataset["dynamics_inputs"]
        dynamics_targets = dataset["dynamics_targets"]
        split = dataset["split"]
    policy, policy_metrics, policy_preprocessing = _train_one(
        name="waypoint_policy",
        inputs=policy_inputs,
        targets=policy_targets,
        split=split,
        training=config.training,
        device=device,
        torch=torch,
    )
    dynamics, dynamics_metrics, dynamics_preprocessing = _train_one(
        name="short_horizon_dynamics",
        inputs=dynamics_inputs,
        targets=dynamics_targets,
        split=split,
        training=config.training,
        device=device,
        torch=torch,
    )
    policy_mae = max(policy_metrics["test_mae_per_target"])
    dynamics_command_mae = max(dynamics_metrics["test_mae_per_target"][:3])
    dynamics_position_mae = max(dynamics_metrics["test_mae_per_target"][3:])
    accepted = (
        policy_mae <= float(config.training["policy_max_command_mae_deg"])
        and dynamics_command_mae
        <= float(config.training["dynamics_max_command_mae_deg"])
        and dynamics_position_mae
        <= float(config.training["dynamics_max_position_mae_m"])
    )
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    destination = config.checkpoint_dir / run_id
    staging = config.checkpoint_dir / f".{run_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    save_file(policy.state_dict(), staging / "waypoint_policy.safetensors")
    save_file(dynamics.state_dict(), staging / "short_horizon_dynamics.safetensors")
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "accepted": accepted,
        "scope": metadata["scope"],
        "physical_execution_validated": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset_sha256": metadata["dataset_sha256"],
        "robot_model_sha256": metadata["robot_model_sha256"],
        "mujoco_model_sha256": metadata["mujoco_model_sha256"],
        "servo_observations_sha256": metadata["servo_observations_sha256"],
        "artifacts": {
            "waypoint_policy_sha256": sha256_file(
                staging / "waypoint_policy.safetensors"
            ),
            "short_horizon_dynamics_sha256": sha256_file(
                staging / "short_horizon_dynamics.safetensors"
            ),
        },
        "policy": {
            "features": list(POLICY_FEATURES),
            "targets": list(POLICY_TARGETS),
            "architecture": {
                "hidden_width": int(config.training["hidden_width"]),
                "hidden_layers": int(config.training["hidden_layers"]),
            },
            "preprocessing": policy_preprocessing,
            "metrics": policy_metrics,
        },
        "dynamics": {
            "features": list(DYNAMICS_FEATURES),
            "targets": list(DYNAMICS_TARGETS),
            "architecture": {
                "hidden_width": int(config.training["hidden_width"]),
                "hidden_layers": int(config.training["hidden_layers"]),
            },
            "preprocessing": dynamics_preprocessing,
            "metrics": dynamics_metrics,
        },
        "acceptance": {
            "policy_max_command_mae_deg": policy_mae,
            "dynamics_max_command_mae_deg": dynamics_command_mae,
            "dynamics_max_position_mae_m": dynamics_position_mae,
            "thresholds": {
                key: config.training[key]
                for key in (
                    "policy_max_command_mae_deg",
                    "dynamics_max_command_mae_deg",
                    "dynamics_max_position_mae_m",
                )
            },
        },
        "limitations": metadata["limitations"],
    }
    _atomic_json(staging / "report.json", report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    if accepted:
        _atomic_json(
            config.checkpoint_dir / "current.json",
            {"run_id": run_id, "report": str((destination / "report.json").resolve())},
        )
    print(
        json.dumps(
            {
                "checkpoint": str(destination),
                **report["acceptance"],
                "accepted": accepted,
            },
            indent=2,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
