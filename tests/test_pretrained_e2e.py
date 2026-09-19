"""Real text models -> selected trained LoRA -> closed-loop synthetic environment.

Run explicitly with --run-e2e. This tests software integration, not robot
manipulation performance. No router responses or learned actions are mocked.
"""

import gc
import json
import math
import random
from dataclasses import asdict

import pytest

from moira import (
    AdapterServer,
    ExpertRegistry,
    HybridRouter,
    InMemoryServer,
    MoIRA,
    PromptExample,
    PromptRouter,
)
from moira.backends import SentenceTransformerEncoder, TransformersGenerator
from moira.evaluation import rollout, success_rate

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

# Pin real model weights so quality regressions remain reproducible.
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SMOLLM_REVISION = "31b70e2e869a7173562077fd711b654946d38674"


def new_backbone():
    import torch

    class PositionPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action = torch.nn.Linear(2, 1, bias=False)
            torch.nn.init.zeros_(self.action.weight)

        def forward(self, observation):
            return self.action(observation)

    return PositionPolicy()


@pytest.fixture(scope="module")
def trained_registry(tmp_path_factory):
    import torch

    from moira.training import LoraTrainingConfig, train_specialist

    directory = tmp_path_factory.mktemp("pretrained_e2e")
    observations = torch.tensor([[position / 3, 1.0] for position in range(-3, 4)])
    experts = []
    for expert_id, direction in [("left", -1.0), ("right", 1.0)]:
        targets = torch.full((len(observations), 1), direction)
        model = new_backbone()
        backbone_before = model.action.weight.detach().clone()
        losses = train_specialist(
            model,
            [(observations, targets)],
            lambda policy, batch: ((policy(batch[0]) - batch[1]) ** 2).mean(),
            directory / expert_id,
            LoraTrainingConfig(
                ("action",),
                steps=80,
                rank=2,
                alpha=2,
                learning_rate=0.1,
                seed=7,
            ),
        )
        assert losses[-1] < 0.01
        assert torch.equal(backbone_before, model.action.base_layer.weight)
        experts.append(
            {
                "id": expert_id,
                "simple": f"Move the robot to the {expert_id} along a horizontal line.",
                "abstract": (
                    "Decrease horizontal position to reach a lower coordinate."
                    if direction < 0
                    else "Increase horizontal position to reach a higher coordinate."
                ),
                "adapter_path": expert_id,
            }
        )
    manifest = directory / "experts.json"
    manifest.write_text(json.dumps({"experts": experts}), encoding="utf-8")
    # Exercise the same on-disk manifest and checkpoint resolution used by deployments.
    return ExpertRegistry.from_json(manifest)


@pytest.fixture(scope="module")
def hybrid_router(trained_registry):
    encoder = SentenceTransformerEncoder(revision=MINILM_REVISION)
    generator = TransformersGenerator(revision=SMOLLM_REVISION)
    yield HybridRouter(trained_registry, encoder, generator)
    # Share frozen weights between cases, then release both models.
    encoder._model = None
    generator._model = None
    generator._tokenizer = None
    gc.collect()


@pytest.fixture(scope="module", params=["hybrid", "prompt"])
def pretrained_router(request, hybrid_router):
    if request.param == "hybrid":
        return hybrid_router
    return PromptRouter(
        hybrid_router.embedding.registry,
        hybrid_router.prompt.generator,
        examples=[
            PromptExample("Move the robot to the left.", "left"),
            PromptExample("Move the robot to the right.", "right"),
        ],
    )


def test_hybrid_resolves_original_embedding_failure(hybrid_router, trained_registry):
    instruction = "Travel toward the right end of the line."
    baseline = hybrid_router.embedding.route(instruction)
    # Preserve the original failure as diagnostic evidence. The production
    # hybrid must override it; no expected-failure marker or altered instruction.
    assert baseline.expert_id == "left"
    assert 0 < baseline.margin < 0.002
    corrected = hybrid_router.route(instruction)
    assert corrected.expert_id == "right"
    assert corrected.embedding_expert_id == "left"
    assert corrected.strategy == "hybrid_prompt"
    assert corrected.scores == baseline.scores

    # Check additional wording without putting it into metadata or few-shot examples.
    for task, expected in [
        ("Head toward the right-hand end of the line.", "right"),
        ("Head toward the left-hand end of the line.", "left"),
        ("Do not go left; move right.", "right"),
        ("Do not go right; move left.", "left"),
    ]:
        assert hybrid_router.route(task).expert_id == expected


class CountingRouter:
    def __init__(self, router):
        self.router, self.calls = router, 0
        self.decisions = []

    def route(self, instruction):
        self.calls += 1
        decision = self.router.route(instruction)
        self.decisions.append(decision)
        return decision


class LineEnvironment:
    """Observation is position plus bias; the target is never exposed to the policy.

    Choosing the wrong specialist reaches the wrong boundary and fails. Actions
    come from the learned adapter; there is no task-to-action lookup in the env.
    """

    def __init__(self, direction):
        self.direction = direction

    def observation(self):
        import torch

        return torch.tensor([[self.position / 3, 1.0]])

    def reset(self, *, seed):
        self.position = float(random.Random(seed).choice([-1, 0, 1]))
        return self.observation(), {}

    def step(self, action):
        assert math.isfinite(action)
        assert -1.2 <= action <= 1.2
        self.position += action
        success = self.direction * self.position >= 2.9
        terminated = abs(self.position) >= 2.9
        return self.observation(), float(success), terminated, False, {"success": success}


def make_server(mode, registry):
    from moira.adapters import PeftAdapterBackend

    class TrackedBackend(PeftAdapterBackend):
        def __init__(self):
            self.loads, self.unloads, self.resets = [], [], []
            super().__init__(
                new_backbone(),
                lambda model, observation, instruction: float(model(observation).item()),
                reset_fn=lambda model, instruction: self.resets.append(instruction),
            )

        def load_adapter(self, expert):
            super().load_adapter(expert)
            self.loads.append(expert.id)

        def unload_adapter(self, expert_id):
            super().unload_adapter(expert_id)
            self.unloads.append(expert_id)

    if mode == "resident":
        policies = {}
        for expert in registry.snapshot():
            backend = TrackedBackend()
            backend.load_adapter(expert)
            backend.activate_adapter(expert.id)
            policies[expert.id] = backend
        return InMemoryServer(policies), list(policies.values())
    backend = TrackedBackend()
    server = AdapterServer(backend, mode=mode)
    if mode == "multi":
        server.preload(registry.snapshot())
    return server, [backend]


@pytest.mark.parametrize("mode", ["resident", "disk", "multi"])
@pytest.mark.parametrize("phrasing", ["literal", "paraphrased"])
def test_pretrained_router_to_trained_policy_episode(
    pretrained_router,
    trained_registry,
    mode,
    phrasing,
):
    router = CountingRouter(pretrained_router)
    server, backends = make_server(mode, trained_registry)
    controller = MoIRA(trained_registry, router, server)
    tasks = [
        ("Move left until you reach the target.", "left", -1, 11),
        ("Move right until you reach the target.", "right", 1, 22),
        ("Travel toward the left end of the line.", "left", -1, 33),
        ("Travel toward the right end of the line.", "right", 1, 44),
    ]
    if phrasing == "literal":
        tasks = [
            (trained_registry.get(expected).simple, expected, direction, seed)
            for _, expected, direction, seed in tasks
        ]
    results = []
    routing_matches = []
    try:
        for instruction, expected, direction, seed in tasks:
            before = router.calls
            result = rollout(
                controller,
                LineEnvironment(direction),
                instruction,
                seed=seed,
                max_steps=8,
            )
            assert router.calls == before + 1
            routing_matches.append(result.expert_id == expected)
            assert result.terminated and not result.truncated
            assert result.steps >= 2  # multiple real observations, not just a single prediction
            selected_direction = -1 if result.expert_id == "left" else 1
            assert all(selected_direction * action > 0.8 for action in result.actions)
            # A wrong route must genuinely fail the environment task.
            assert result.success == (result.expert_id == expected)
            results.append(result)
        assert sorted(text for b in backends for text in b.resets) == sorted(t[0] for t in tasks)
        switches = sum(
            previous.expert_id != current.expert_id
            for previous, current in zip(results, results[1:], strict=False)
        )
        assert sum(len(b.loads) for b in backends) == (1 + switches if mode == "disk" else 2)
        assert sum(len(b.unloads) for b in backends) == (switches if mode == "disk" else 0)
        print(
            json.dumps(
                {
                    "router": type(pretrained_router).__name__,
                    "serving": mode,
                    "phrasing": phrasing,
                    "instructions": [task[0] for task in tasks],
                    "routing_matches": routing_matches,
                    "decisions": [asdict(decision) for decision in router.decisions],
                    "success_rate": success_rate(results),
                    "episodes": [asdict(r) for r in results],
                }
            )
        )
        # Report every episode before asserting quality; never stop at the first
        # routing miss or silently weaken the success criterion after a failure.
        assert all(routing_matches), f"Incorrect routes: {routing_matches}"
        assert success_rate(results) == 1.0
    finally:
        if mode != "resident":
            server.close()
        else:
            for expert, backend in zip(trained_registry.snapshot(), backends, strict=True):
                backend.unload_adapter(expert.id)
