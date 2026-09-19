"""PEFT adapter serving for compatible PyTorch policies.

GR00T and OpenPI may use their own adapter implementations. Implement the small
AdapterBackend protocol for those runtimes; no VLA-specific tensor schema is
assumed here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .experts import Expert


class PeftAdapterBackend:
    def __init__(
        self,
        base_model: Any,
        action_fn: Callable[[Any, Any, str], Any],
        *,
        reset_fn: Callable[[Any, str], None] | None = None,
    ) -> None:
        if base_model is None:
            raise ValueError("base_model is required")
        if getattr(base_model, "peft_config", None):
            raise ValueError("base_model must be fresh and must not already contain PEFT adapters")
        if not callable(action_fn):
            raise TypeError("action_fn must be callable")
        if reset_fn is not None and not callable(reset_fn):
            raise TypeError("reset_fn must be callable or null")
        self.model = base_model
        self.action_fn, self.reset_fn = action_fn, reset_fn
        self._wrapped = False
        self._names: dict[str, str] = {}
        self._next_name = 0
        self._active: str | None = None
        self._instruction: str | None = None

    @property
    def loaded_ids(self) -> tuple[str, ...]:
        return tuple(self._names)

    @property
    def active_id(self) -> str | None:
        return self._active

    def load_adapter(self, expert: Expert) -> None:
        if not expert.adapter_path:
            raise ValueError(f"Expert {expert.id} needs an adapter_path")
        if expert.id in self._names:
            raise ValueError(f"Adapter already loaded: {expert.id}")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError('Install PEFT support: pip install -e ".[adapters]"') from exc
        # External IDs may contain dots, which PyTorch module names cannot.
        name = f"moira_{self._next_name}"
        self._next_name += 1
        if not self._wrapped:
            self.model = PeftModel.from_pretrained(
                self.model,
                expert.adapter_path,
                adapter_name=name,
                is_trainable=False,
            )
            self._wrapped = True
        else:
            self.model.load_adapter(expert.adapter_path, adapter_name=name, is_trainable=False)
        self.model.eval()
        self.model.requires_grad_(False)
        self._names[expert.id] = name

    def unload_adapter(self, expert_id: str) -> None:
        if expert_id not in self._names:
            raise KeyError(f"Adapter is not loaded: {expert_id}")
        name = self._names[expert_id]
        was_active = self._active == expert_id
        if was_active:
            # Clear public episode state before changing PEFT's active adapter.
            self._active, self._instruction = None, None
        if len(self._names) == 1:
            # Return to a clean backbone when evicting the last adapter.
            self.model = self.model.unload()
            self._wrapped = False
        else:
            if was_active:
                replacement = next(
                    adapter_name
                    for loaded_id, adapter_name in self._names.items()
                    if loaded_id != expert_id
                )
                self.model.set_adapter(replacement)
                self.model.requires_grad_(False)
                self.model.eval()
            self.model.delete_adapter(name)
        del self._names[expert_id]

    def activate_adapter(self, expert_id: str) -> None:
        if expert_id not in self._names:
            raise KeyError(f"Adapter is not loaded: {expert_id}")
        # Fail closed if PEFT changes model state before raising.
        self._active, self._instruction = None, None
        self.model.set_adapter(self._names[expert_id])
        # Some PEFT versions mark selected parameters trainable on set_adapter.
        self.model.requires_grad_(False)
        self.model.eval()
        self._active, self._instruction = expert_id, None

    def reset(self, instruction: str) -> None:
        if self._active is None:
            raise RuntimeError("No adapter is active")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Task instruction must be a non-empty string")
        # A failed reset must never leave the backend ready to act using an old
        # or partially initialized episode state.
        self._instruction = None
        if self.reset_fn is not None:
            self.reset_fn(self.model, instruction)
        self._instruction = instruction

    def act(self, observation: Any) -> Any:
        if self._active is None or self._instruction is None:
            raise RuntimeError("Activate an adapter and reset the episode before inference")
        import torch

        with torch.inference_mode():
            return self.action_fn(self.model, observation, self._instruction)
