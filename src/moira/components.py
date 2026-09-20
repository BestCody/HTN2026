"""Exact-capability component routing for resource-constrained physical AI."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import BoundedSemaphore, RLock
from time import monotonic
from typing import Any, Literal, Protocol


class Layer(str, Enum):
    PERCEPTION = "perception"
    PERSONAL = "personal"
    GRASP = "grasp"
    MANIPULATION = "manipulation"
    KINEMATICS = "kinematics"
    MOTION = "motion"
    DYNAMICS = "dynamics"
    WORLD = "world"
    REWARD = "reward"
    SAFETY = "safety"
    TACTILE = "tactile"
    LOAD = "load"
    OUTCOME = "outcome"
    FAILURE = "failure"
    SIMULATION = "simulation"  # Legacy paper/demo compatibility.
    VOICE = "voice"
    PLANNING = "planning"
    CONTROL = "control"
    FEEDBACK = "feedback"


RuntimeKind = Literal["local", "remote", "hardware"]


class ModelComponent(Protocol):
    """One specialized component with a single typed request/response boundary."""

    def run(self, request: Any) -> Any: ...


@dataclass(frozen=True)
class ComponentSpec:
    id: str
    layer: Layer
    capabilities: tuple[str, ...]
    model: str
    runtime: RuntimeKind = "local"
    estimated_ram_mb: int = 0
    priority: int = 0
    max_concurrency: int = 1
    pi_compatible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Component id must be a non-empty string")
        if isinstance(self.layer, str):
            try:
                object.__setattr__(self, "layer", Layer(self.layer))
            except ValueError as exc:
                raise ValueError(f"Unknown component layer: {self.layer}") from exc
        if not isinstance(self.layer, Layer):
            raise ValueError("Component layer must be a Layer")
        if (
            not isinstance(self.capabilities, tuple)
            or not self.capabilities
            or any(not isinstance(item, str) or not item.strip() for item in self.capabilities)
        ):
            raise ValueError("Component capabilities must be a non-empty tuple of strings")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("Component capabilities must be unique")
        prefix = f"{self.layer.value}."
        if any(
            capability == "*" or not capability.startswith(prefix)
            for capability in self.capabilities
        ):
            raise ValueError(f"Every capability must be exact and start with '{prefix}'")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Component model must be a non-empty identifier")
        if self.runtime not in ("local", "remote", "hardware"):
            raise ValueError("Component runtime must be local, remote, or hardware")
        for name in ("estimated_ram_mb", "priority", "max_concurrency"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Component {name} must be an integer")
        if self.estimated_ram_mb < 0:
            raise ValueError("Component estimated_ram_mb must be nonnegative")
        if self.max_concurrency < 1:
            raise ValueError("Component max_concurrency must be positive")
        if not isinstance(self.pi_compatible, bool):
            raise ValueError("Component pi_compatible must be boolean")


@dataclass(frozen=True)
class ComponentDecision:
    capability: str
    component_id: str
    layer: Layer
    model: str
    runtime: RuntimeKind
    alternatives: tuple[str, ...]
    router: str = "local"

    def __post_init__(self) -> None:
        for value, name in (
            (self.capability, "Decision capability"),
            (self.component_id, "Decision component id"),
            (self.model, "Decision model"),
            (self.router, "Decision router"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.layer, Layer):
            raise TypeError("Decision layer must be a Layer")
        if not self.capability.startswith(f"{self.layer.value}."):
            raise ValueError("Decision capability does not match its layer")
        if self.runtime not in ("local", "remote", "hardware"):
            raise ValueError("Decision runtime must be local, remote, or hardware")
        if (
            not isinstance(self.alternatives, tuple)
            or any(not isinstance(item, str) or not item.strip() for item in self.alternatives)
            or len(set(self.alternatives)) != len(self.alternatives)
            or self.component_id in self.alternatives
        ):
            raise ValueError("Decision alternatives must be unique component IDs")


class ComponentRouter(Protocol):
    def decide(
        self,
        layer: Layer,
        capability: str,
        context: Mapping[str, Any] | None = None,
    ) -> ComponentDecision: ...


@dataclass
class _Entry:
    spec: ComponentSpec
    factory: Callable[[], ModelComponent]
    gate: BoundedSemaphore
    instance: ModelComponent | None = None
    active: int = 0
    last_used: float = 0.0


class ComponentLease:
    def __init__(self, component: ModelComponent) -> None:
        self._component = component
        self._active = True

    def run(self, request: Any) -> Any:
        if not self._active:
            raise RuntimeError("Component session is closed")
        return self._component.run(request)

    def close(self) -> None:
        self._active = False


class ComponentRegistry:
    """Route exact capabilities and keep a bounded LRU set of components loaded.

    RAM values are admission estimates supplied by component authors. They let a
    Pi deployment reject an impossible resident set before loading it; they are
    not a substitute for operating-system memory telemetry.
    """

    def __init__(self, *, ram_budget_mb: int, allow_remote: bool = True) -> None:
        if (
            not isinstance(ram_budget_mb, int)
            or isinstance(ram_budget_mb, bool)
            or ram_budget_mb < 1
        ):
            raise ValueError("ram_budget_mb must be positive")
        if not isinstance(allow_remote, bool):
            raise TypeError("allow_remote must be boolean")
        self.ram_budget_mb = ram_budget_mb
        self.allow_remote = allow_remote
        self._entries: dict[str, _Entry] = {}
        self._lock = RLock()

    def register(self, spec: ComponentSpec, factory: Callable[[], ModelComponent]) -> None:
        if not isinstance(spec, ComponentSpec):
            raise TypeError("spec must be a ComponentSpec")
        if not callable(factory):
            raise TypeError("component factory must be callable")
        if not spec.pi_compatible:
            raise ValueError(f"Component is not marked Raspberry Pi compatible: {spec.id}")
        if spec.estimated_ram_mb > self.ram_budget_mb:
            raise ValueError(
                f"Component {spec.id} needs {spec.estimated_ram_mb} MB, above the "
                f"{self.ram_budget_mb} MB component budget"
            )
        with self._lock:
            if spec.id in self._entries:
                raise ValueError(f"Duplicate component id: {spec.id}")
            self._entries[spec.id] = _Entry(
                spec,
                factory,
                BoundedSemaphore(spec.max_concurrency),
            )

    @property
    def specs(self) -> tuple[ComponentSpec, ...]:
        with self._lock:
            return tuple(entry.spec for entry in self._entries.values())

    @property
    def loaded_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                component_id
                for component_id, entry in self._entries.items()
                if entry.instance is not None
            )

    @property
    def estimated_loaded_ram_mb(self) -> int:
        with self._lock:
            return self._loaded_ram()

    def decide(
        self,
        layer: Layer,
        capability: str,
        context: Mapping[str, Any] | None = None,
    ) -> ComponentDecision:
        del context
        if isinstance(layer, str):
            try:
                layer = Layer(layer)
            except ValueError as exc:
                raise ValueError(f"Unknown component layer: {layer}") from exc
        if not isinstance(layer, Layer):
            raise TypeError("layer must be a Layer")
        if not isinstance(capability, str) or not capability.startswith(f"{layer.value}."):
            raise ValueError(f"Capability must start with '{layer.value}.'")
        with self._lock:
            candidates = [
                entry.spec
                for entry in self._entries.values()
                if entry.spec.layer is layer
                and capability in entry.spec.capabilities
                and (self.allow_remote or entry.spec.runtime != "remote")
            ]
        if not candidates:
            raise LookupError(f"No Raspberry Pi component provides exact capability: {capability}")
        runtime_rank = {"hardware": 0, "local": 1, "remote": 2}
        candidates.sort(
            key=lambda spec: (
                -spec.priority,
                runtime_rank[spec.runtime],
                spec.estimated_ram_mb,
                spec.id,
            )
        )
        selected = candidates[0]
        return ComponentDecision(
            capability,
            selected.id,
            selected.layer,
            selected.model,
            selected.runtime,
            tuple(spec.id for spec in candidates[1:]),
        )

    def decision_for(
        self,
        layer: Layer,
        capability: str,
        component_id: str,
        *,
        router: str,
    ) -> ComponentDecision:
        local = self.decide(layer, capability)
        with self._lock:
            try:
                selected = self._entries[component_id].spec
            except KeyError:
                raise LookupError(f"Router selected unknown component: {component_id}") from None
            valid = (
                selected.layer is local.layer
                and capability in selected.capabilities
                and (self.allow_remote or selected.runtime != "remote")
            )
            if not valid:
                raise LookupError(
                    f"Router selected {component_id}, which cannot provide exact "
                    f"capability {capability}"
                )
            alternatives = tuple(
                spec.id
                for spec in self.specs
                if spec.id != selected.id
                and spec.layer is local.layer
                and capability in spec.capabilities
                and (self.allow_remote or spec.runtime != "remote")
            )
        return ComponentDecision(
            capability,
            selected.id,
            selected.layer,
            selected.model,
            selected.runtime,
            alternatives,
            router,
        )

    @contextmanager
    def session(self, component_id: str) -> Iterator[ComponentLease]:
        with self._lock:
            try:
                entry = self._entries[component_id]
            except KeyError:
                raise KeyError(f"Unknown component id: {component_id}") from None
        entry.gate.acquire()
        lease: ComponentLease | None = None
        try:
            with self._lock:
                self._ensure_loaded(entry)
                entry.active += 1
                entry.last_used = monotonic()
                lease = ComponentLease(entry.instance)
            yield lease
        finally:
            if lease is not None:
                lease.close()
            with self._lock:
                if entry.active:
                    entry.active -= 1
                    entry.last_used = monotonic()
            entry.gate.release()

    def invoke(self, decision: ComponentDecision, request: Any) -> Any:
        if not isinstance(decision, ComponentDecision):
            raise TypeError("decision must be a ComponentDecision")
        with self._lock:
            try:
                spec = self._entries[decision.component_id].spec
            except KeyError:
                raise LookupError(
                    f"Decision selected unknown component: {decision.component_id}"
                ) from None
            if (
                spec.layer is not decision.layer
                or decision.capability not in spec.capabilities
                or spec.model != decision.model
                or spec.runtime != decision.runtime
                or (not self.allow_remote and spec.runtime == "remote")
            ):
                raise LookupError(
                    f"Decision metadata does not match registered component {spec.id}"
                )
        with self.session(decision.component_id) as component:
            return component.run(request)

    def unload_all(self) -> None:
        with self._lock:
            active = [entry.spec.id for entry in self._entries.values() if entry.active]
            if active:
                raise RuntimeError(f"Cannot unload active components: {', '.join(active)}")
            failures = []
            for entry in self._entries.values():
                if entry.instance is None:
                    continue
                try:
                    self._close(entry.instance)
                except Exception as exc:
                    failures.append((entry.spec.id, exc))
                else:
                    entry.instance = None
            if failures:
                ids = ", ".join(component_id for component_id, _ in failures)
                raise RuntimeError(f"Failed to unload components: {ids}") from failures[0][1]

    def emergency_stop(self) -> tuple[str, ...]:
        """Signal every loaded stoppable component without waiting for its lease.

        This method intentionally snapshots component instances under the registry
        lock and calls stop hooks after releasing it. A concurrent hardware lease
        must remain interruptible while an arm command is in progress.
        """

        with self._lock:
            loaded = tuple(
                (component_id, entry.instance)
                for component_id, entry in self._entries.items()
                if entry.instance is not None
            )
        issues: list[str] = []
        for component_id, instance in loaded:
            stop = getattr(instance, "emergency_stop", None)
            if not callable(stop):
                stop = getattr(instance, "stop", None)
            if not callable(stop):
                continue
            try:
                stop()
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                issues.append(f"{component_id}: {detail}")
        return tuple(issues)

    def _loaded_ram(self) -> int:
        return sum(
            entry.spec.estimated_ram_mb
            for entry in self._entries.values()
            if entry.instance is not None
        )

    def _ensure_loaded(self, entry: _Entry) -> None:
        if entry.instance is not None:
            return
        required = self._loaded_ram() + entry.spec.estimated_ram_mb - self.ram_budget_mb
        if required > 0:
            idle = sorted(
                (
                    candidate
                    for candidate in self._entries.values()
                    if candidate is not entry
                    and candidate.instance is not None
                    and candidate.active == 0
                ),
                key=lambda candidate: candidate.last_used,
            )
            for candidate in idle:
                self._close(candidate.instance)
                candidate.instance = None
                required -= candidate.spec.estimated_ram_mb
                if required <= 0:
                    break
        if required > 0:
            raise MemoryError(
                f"Cannot load {entry.spec.id} within the {self.ram_budget_mb} MB component budget"
            )
        instance = entry.factory()
        if not callable(getattr(instance, "run", None)):
            self._close(instance)
            raise TypeError(
                f"Component factory did not return a runnable component: {entry.spec.id}"
            )
        entry.instance = instance
        entry.last_used = monotonic()

    @staticmethod
    def _close(instance: Any) -> None:
        close = getattr(instance, "close", None)
        if callable(close):
            close()

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        factories: Mapping[str, Callable[[], ModelComponent]],
        *,
        ram_budget_mb: int,
        allow_remote: bool = True,
    ) -> ComponentRegistry:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("components"), list):
            raise ValueError("Component manifest must contain a components list")
        registry = cls(ram_budget_mb=ram_budget_mb, allow_remote=allow_remote)
        for index, value in enumerate(data["components"]):
            if not isinstance(value, dict):
                raise ValueError(f"Component at index {index} must be an object")
            value = dict(value)
            if isinstance(value.get("capabilities"), list):
                value["capabilities"] = tuple(value["capabilities"])
            try:
                spec = ComponentSpec(**value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid component at index {index}: {exc}") from exc
            if spec.id not in factories:
                raise ValueError(f"No factory supplied for component: {spec.id}")
            registry.register(spec, factories[spec.id])
        return registry
