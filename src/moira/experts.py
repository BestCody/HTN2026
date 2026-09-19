"""Expert metadata is independent of the policy architecture and routing model."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal

DescriptionStyle = Literal["simple", "abstract"]


@dataclass(frozen=True)
class Expert:
    id: str
    simple: str
    abstract: str
    adapter_path: str | None = None
    interfaces: tuple[str, ...] = ("general",)
    routing_examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("id", "simple", "abstract"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Expert {field} must be a non-empty string")
        if self.adapter_path is not None and (
            not isinstance(self.adapter_path, str) or not self.adapter_path.strip()
        ):
            raise ValueError("adapter_path must be a non-empty string or null")
        if isinstance(self.interfaces, list):
            object.__setattr__(self, "interfaces", tuple(self.interfaces))
        if (
            not isinstance(self.interfaces, tuple)
            or not self.interfaces
            or any(not isinstance(value, str) or not value.strip() for value in self.interfaces)
            or len(set(self.interfaces)) != len(self.interfaces)
        ):
            raise ValueError("interfaces must contain unique non-empty strings")
        if isinstance(self.routing_examples, list):
            object.__setattr__(self, "routing_examples", tuple(self.routing_examples))
        if (
            not isinstance(self.routing_examples, tuple)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.routing_examples
            )
            or len(set(self.routing_examples)) != len(self.routing_examples)
            or len(self.routing_examples) > 16
        ):
            raise ValueError("routing_examples must contain at most 16 unique non-empty strings")

    def description(self, style: DescriptionStyle) -> str:
        if style not in ("simple", "abstract"):
            raise ValueError(f"Unknown description style: {style}")
        return getattr(self, style)

    def routing_texts(self, style: DescriptionStyle) -> tuple[str, ...]:
        """Return the description plus optional metadata examples."""
        return (self.description(style), *self.routing_examples)


class ExpertRegistry:
    """Thread-safe insertion-ordered expert metadata.

    A route sees one stable snapshot. Existing episodes retain the selected
    immutable Expert even when the registry changes between episodes.
    """

    def __init__(self, experts: list[Expert] | tuple[Expert, ...] = ()) -> None:
        self._experts: dict[str, Expert] = {}
        self._lock = RLock()
        for expert in experts:
            self.add(expert)

    def add(self, expert: Expert) -> None:
        if not isinstance(expert, Expert):
            raise TypeError("Registry entries must be Expert instances")
        with self._lock:
            if expert.id in self._experts:
                raise ValueError(f"Duplicate expert ID: {expert.id}")
            self._experts[expert.id] = expert

    def replace(self, expert: Expert) -> None:
        if not isinstance(expert, Expert):
            raise TypeError("Registry entries must be Expert instances")
        with self._lock:
            self.get(expert.id)
            self._experts[expert.id] = expert

    def remove(self, expert_id: str) -> Expert:
        with self._lock:
            expert = self.get(expert_id)
            del self._experts[expert_id]
            return expert

    def get(self, expert_id: str) -> Expert:
        with self._lock:
            try:
                return self._experts[expert_id]
            except KeyError:
                raise KeyError(f"Unknown expert ID: {expert_id}") from None

    def snapshot(self) -> tuple[Expert, ...]:
        with self._lock:
            return tuple(self._experts.values())

    def for_interface(self, interface: str) -> ExpertRegistry:
        """Return the experts that satisfy one caller-defined compatibility contract.

        Interfaces constrain schemas or execution roles; they do not encode a
        task-to-expert route. The semantic router still ranks every compatible
        expert from its natural-language description.
        """
        if not isinstance(interface, str) or not interface.strip():
            raise ValueError("interface must be a non-empty string")
        experts = tuple(expert for expert in self.snapshot() if interface in expert.interfaces)
        if not experts:
            raise LookupError(f"No experts provide interface: {interface}")
        return ExpertRegistry(experts)

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold metadata stable across a multi-stage route-and-lookup operation."""
        with self._lock:
            yield

    @classmethod
    def from_json(cls, path: str | Path) -> ExpertRegistry:
        """Load metadata; local adapter paths are relative to the manifest directory.

        Hub adapter IDs may be supplied as hf://organization/repository.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("experts"), list):
            raise ValueError("Manifest must be an object containing an 'experts' list")
        experts = []
        for index, entry in enumerate(data["experts"]):
            if not isinstance(entry, dict):
                raise ValueError(f"Expert at index {index} must be an object")
            entry = dict(entry)
            adapter = entry.get("adapter_path")
            if isinstance(adapter, str) and adapter.strip():
                if adapter.startswith("hf://"):
                    entry["adapter_path"] = adapter.removeprefix("hf://")
                else:
                    entry["adapter_path"] = str((path.parent / adapter).resolve())
            try:
                experts.append(Expert(**entry))
            except TypeError as exc:
                raise ValueError(f"Invalid expert schema at index {index}: {exc}") from exc
        return cls(experts)
