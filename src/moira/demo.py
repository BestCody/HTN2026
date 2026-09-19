"""Deterministic plumbing demo: no pretrained model and no robot required."""

from __future__ import annotations

from .evaluation import rollout
from .experts import Expert, ExpertRegistry
from .routing import EmbeddingRouter
from .runtime import MoIRA
from .serving import InMemoryServer


class ToyEncoder:
    """Two manually chosen axes; NOT a substitute for the MiniLM router."""

    def encode(self, texts):
        return [
            [float("left" in text.lower()), float("right" in text.lower()), 0.01] for text in texts
        ]


class MovePolicy:
    def __init__(self, direction: int):
        self.direction = direction

    def reset(self, instruction: str) -> None:
        self.instruction = instruction

    def act(self, observation: int) -> int:
        return self.direction


class LineWorld:
    def reset(self, *, seed: int):
        self.position = 0
        return self.position, {}

    def step(self, action: int):
        self.position += action
        success = self.position == 3
        return self.position, float(success), success, False, {"success": success}


def run_demo():
    registry = ExpertRegistry(
        [
            Expert("left", "move left", "decrease horizontal position"),
            Expert("right", "move right", "increase horizontal position"),
        ]
    )
    router = EmbeddingRouter(registry, ToyEncoder())
    controller = MoIRA(
        registry,
        router,
        InMemoryServer(
            {
                "left": MovePolicy(-1),
                "right": MovePolicy(1),
            }
        ),
    )
    return rollout(controller, LineWorld(), "move right", seed=0, max_steps=5)
