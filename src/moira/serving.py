"""Expert serving with exclusive episode leases on mutable policy state."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from threading import Lock, RLock
from typing import Any, Literal, Protocol

from .experts import Expert


class Policy(Protocol):
    def reset(self, instruction: str) -> None:
        """Clear episode state and set the original language instruction."""
        ...

    def act(self, observation: Any) -> Any:
        """Return a robot-specific action (or action chunk) for this observation."""
        ...


class PolicyServer(Protocol):
    def session(self, expert: Expert) -> AbstractContextManager[Policy]: ...


class AdapterBackend(Policy, Protocol):
    """One shared backbone. Backends own device placement and preprocessing."""

    def load_adapter(self, expert: Expert) -> None: ...
    def unload_adapter(self, expert_id: str) -> None: ...
    def activate_adapter(self, expert_id: str) -> None: ...


class _PolicyLease:
    """Prevent a policy reference from being used after its session closes."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._active = True

    def reset(self, instruction: str) -> None:
        if not self._active:
            raise RuntimeError("Policy session is closed")
        self._policy.reset(instruction)

    def act(self, observation: Any) -> Any:
        if not self._active:
            raise RuntimeError("Policy session is closed")
        return self._policy.act(observation)

    def close(self) -> None:
        self._active = False


class InMemoryServer:
    """Keep independent full policies resident with one lease per expert.

    Different experts may run concurrently because they have independent model
    and episode state. Overlapping episodes for the same policy are rejected.
    The policy mapping is copied at construction and is immutable thereafter.
    """

    def __init__(self, policies: Mapping[str, Policy]) -> None:
        self._policies = dict(policies)
        if any(not isinstance(expert_id, str) or not expert_id.strip() for expert_id in policies):
            raise ValueError("Resident policy IDs must be non-empty strings")
        for expert_id, policy in self._policies.items():
            if not callable(getattr(policy, "reset", None)) or not callable(
                getattr(policy, "act", None)
            ):
                raise TypeError(f"Resident policy must implement reset() and act(): {expert_id}")
        locks_by_policy: dict[int, Lock] = {}
        self._locks = {
            expert_id: locks_by_policy.setdefault(id(policy), Lock())
            for expert_id, policy in self._policies.items()
        }

    @property
    def policy_ids(self) -> tuple[str, ...]:
        return tuple(self._policies)

    @contextmanager
    def session(self, expert: Expert) -> Iterator[Policy]:
        if not isinstance(expert, Expert):
            raise TypeError("expert must be an Expert")
        if expert.id not in self._policies:
            raise KeyError(f"No resident policy for expert: {expert.id}")
        lock = self._locks[expert.id]
        if not lock.acquire(blocking=False):
            raise RuntimeError(f"Resident policy already has an active episode: {expert.id}")
        lease = _PolicyLease(self._policies[expert.id])
        try:
            yield lease
        finally:
            lease.close()
            lock.release()


class AdapterServer:
    """Serve one backbone with disk swapping or multiple resident LoRA adapters.

    disk: evict the previous adapter before loading the next (one resident).
    multi: load each adapter once and retain it for subsequent hot switches.
    Use preload() to populate multi mode before measuring switching latency.
    Neither mode is a batched multi-tenant kernel implementation.
    """

    def __init__(self, backend: AdapterBackend, *, mode: Literal["disk", "multi"] = "disk"):
        if mode not in ("disk", "multi"):
            raise ValueError("Adapter mode must be 'disk' or 'multi'")
        methods = ("load_adapter", "unload_adapter", "activate_adapter", "reset", "act")
        if any(not callable(getattr(backend, method, None)) for method in methods):
            raise TypeError("backend must implement the AdapterBackend protocol")
        self.backend, self.mode = backend, mode
        self._loaded: dict[str, Expert] = {}
        self._lock = Lock()
        self._state_lock = RLock()

    @property
    def loaded_ids(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(self._loaded)

    def _ensure_loaded(self, expert: Expert) -> None:
        with self._state_lock:
            previous = self._loaded.get(expert.id)
            if previous is not None and previous.adapter_path != expert.adapter_path:
                self.backend.unload_adapter(expert.id)
                del self._loaded[expert.id]
            if self.mode == "disk":
                for expert_id in tuple(self._loaded):
                    if expert_id != expert.id:
                        self.backend.unload_adapter(expert_id)
                        del self._loaded[expert_id]
            if expert.id not in self._loaded:
                self.backend.load_adapter(expert)
                self._loaded[expert.id] = expert

    def preload(self, experts: Iterable[Expert]) -> None:
        if self.mode != "multi":
            raise ValueError("Preloading is only available for multi-adapter serving")
        try:
            experts = tuple(experts)
        except TypeError as exc:
            raise TypeError("experts must be iterable") from exc
        if any(not isinstance(expert, Expert) for expert in experts):
            raise TypeError("preload entries must be Expert instances")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot preload during an active episode")
        try:
            ids = [expert.id for expert in experts]
            if len(ids) != len(set(ids)):
                raise ValueError("Cannot preload duplicate expert IDs")
            with self._state_lock:
                conflicting = [
                    expert.id
                    for expert in experts
                    if expert.id in self._loaded
                    and self._loaded[expert.id].adapter_path != expert.adapter_path
                ]
                if conflicting:
                    joined = ", ".join(conflicting)
                    raise ValueError(f"Cannot replace loaded adapters during preload: {joined}")
                original_ids = set(self._loaded)
            try:
                for expert in experts:
                    self._ensure_loaded(expert)
            except Exception as load_error:
                rollback_error: Exception | None = None
                with self._state_lock:
                    added_ids = [
                        expert_id
                        for expert_id in reversed(tuple(self._loaded))
                        if expert_id not in original_ids
                    ]
                    for expert_id in added_ids:
                        try:
                            self.backend.unload_adapter(expert_id)
                        except Exception as error:
                            rollback_error = rollback_error or error
                        else:
                            del self._loaded[expert_id]
                if rollback_error is not None:
                    raise RuntimeError(
                        "Adapter preload failed and rollback could not unload every adapter"
                    ) from load_error
                raise
        finally:
            self._lock.release()

    @contextmanager
    def session(self, expert: Expert) -> Iterator[Policy]:
        if not isinstance(expert, Expert):
            raise TypeError("expert must be an Expert")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Policy server already has an active episode")
        lease: _PolicyLease | None = None
        try:
            self._ensure_loaded(expert)
            self.backend.activate_adapter(expert.id)
            lease = _PolicyLease(self.backend)
            yield lease
        finally:
            if lease is not None:
                lease.close()
            self._lock.release()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close during an active episode")
        try:
            failures: list[tuple[str, Exception]] = []
            with self._state_lock:
                for expert_id in tuple(self._loaded):
                    try:
                        self.backend.unload_adapter(expert_id)
                    except Exception as error:
                        failures.append((expert_id, error))
                    else:
                        del self._loaded[expert_id]
            if failures:
                ids = ", ".join(expert_id for expert_id, _ in failures)
                raise RuntimeError(f"Failed to unload adapters: {ids}") from failures[0][1]
        finally:
            self._lock.release()
