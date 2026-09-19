"""Independent LoRA specialist fine-tuning for PEFT-compatible backbones."""

from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoraTrainingConfig:
    # These are implementation defaults, NOT hyperparameters reported by MoIRA.
    target_modules: tuple[str, ...]
    steps: int
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_modules, tuple)
            or not self.target_modules
            or any(not isinstance(name, str) or not name.strip() for name in self.target_modules)
        ):
            raise ValueError("Specify the backbone's LoRA target module names")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target module names must be unique")
        integers = (self.steps, self.rank, self.alpha)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers
        ):
            raise ValueError("steps, rank, and alpha must be positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(self.dropout, (int, float))
            or isinstance(self.dropout, bool)
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")
        for value in (self.learning_rate, self.max_grad_norm):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("learning_rate and max_grad_norm must be finite and positive")
        if (
            not isinstance(self.weight_decay, (int, float))
            or isinstance(self.weight_decay, bool)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and nonnegative")


def train_specialist(
    base_model: Any,
    batches: Iterable[Any],
    loss_fn: Callable[[Any, Any], Any],
    output_dir: str | Path,
    config: LoraTrainingConfig,
) -> list[float]:
    """Train one fresh backbone's adapter and save it in PEFT format.

    The caller supplies device-ready batches and the native policy loss. A
    reiterable dataloader restarts at epoch boundaries; exhausted one-shot
    iterators fail clearly. Use a fresh base model for each specialist. This
    mutates that model by inserting LoRA modules. Seeds control adapter init,
    not data already shuffled or backbone weights already initialized.
    """
    if not isinstance(config, LoraTrainingConfig):
        raise TypeError("config must be a LoraTrainingConfig")
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    if not callable(getattr(base_model, "parameters", None)):
        raise TypeError("base_model must be a PyTorch module")
    if not isinstance(batches, Iterable):
        raise TypeError("batches must be iterable")

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("Training output path exists and is not a directory")
        if any(output_dir.iterdir()):
            raise ValueError(
                "Training output directory must be empty to avoid overwriting a specialist"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(base_model, PeftModel) or getattr(base_model, "peft_config", None):
        raise ValueError("Train each specialist from a fresh, unadapted backbone")
    # Keep this helper from perturbing the caller's global CPU/CUDA RNG streams.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed)
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules=list(config.target_modules),
                bias="none",
            ),
        )
        model.train()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("LoRA configuration produced no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        iterator = iter(batches)
        losses = []
        for _ in range(config.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                try:
                    batch = next(iterator)
                except StopIteration:
                    raise ValueError(
                        "Training batches are empty or a one-shot iterator was exhausted"
                    ) from None
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            if (
                not isinstance(loss, torch.Tensor)
                or loss.numel() != 1
                or not torch.is_floating_point(loss)
            ):
                raise ValueError("Policy loss must be a scalar floating-point tensor")
            if not bool(torch.isfinite(loss).all()):
                raise ValueError("Policy loss must be finite")
            if not loss.requires_grad:
                raise ValueError("Policy loss must require gradients")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
        try:
            model.save_pretrained(staging)
            if output_dir.exists():
                output_dir.rmdir()  # It was required to be empty above.
            staging.replace(output_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        model.eval()
        return losses
