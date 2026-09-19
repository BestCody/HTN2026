"""Route once at an episode boundary, then forward observations to that expert."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .experts import ExpertRegistry
from .routing import Router, RoutingDecision
from .serving import Policy, PolicyServer


@dataclass
class Episode:
    decision: RoutingDecision
    _policy: Policy
    _active: bool = True

    @property
    def active(self) -> bool:
        return self._active

    def act(self, observation: Any) -> Any:
        if not self._active:
            raise RuntimeError("Episode is closed")
        return self._policy.act(observation)


class MoIRA:
    def __init__(self, registry: ExpertRegistry, router: Router, server: PolicyServer):
        if not isinstance(registry, ExpertRegistry):
            raise TypeError("registry must be an ExpertRegistry")
        if not callable(getattr(router, "route", None)):
            raise TypeError("router must implement route(instruction)")
        if not callable(getattr(server, "session", None)):
            raise TypeError("server must implement session(expert)")
        router_registry = getattr(router, "registry", registry)
        if router_registry is not registry:
            raise ValueError("Controller and router must share the same ExpertRegistry instance")
        self.registry, self.router, self.server = registry, router, server

    @contextmanager
    def episode(self, instruction: str) -> Iterator[Episode]:
        # Keep the selected immutable metadata alive across a concurrent registry
        # update; the long-running policy session does not hold the registry lock.
        with self.registry.locked():
            decision = self.router.route(instruction)
            if not isinstance(decision, RoutingDecision):
                raise TypeError("router must return a RoutingDecision")
            expert = self.registry.get(decision.expert_id)
        with self.server.session(expert) as policy:
            policy.reset(instruction)
            episode = Episode(decision, policy)
            try:
                yield episode
            finally:
                episode._active = False
