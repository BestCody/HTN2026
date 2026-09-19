import json
from dataclasses import replace

import pytest

from moira import (
    EmbeddingRouter,
    Expert,
    ExpertRegistry,
    PromptExample,
    PromptRouter,
    RoutingDecision,
    RoutingError,
)


@pytest.fixture
def registry():
    return ExpertRegistry(
        [
            Expert("a", "first description", "first abstract"),
            Expert("b", "second description", "second abstract"),
        ]
    )


class Encoder:
    def __init__(self, values):
        self.values, self.calls = values, []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [self.values[text] for text in texts]


def test_cosine_not_unnormalized_dot_product_and_cache(registry):
    encoder = Encoder({"first description": [100, 1], "second description": [1, 1], "task": [1, 1]})
    router = EmbeddingRouter(registry, encoder)
    decision = router.route("task")
    assert decision.expert_id == "b"
    assert dict(decision.scores)["b"] == pytest.approx(1)
    router.route("task")
    assert encoder.calls == [["first description", "second description"], ["task"], ["task"]]
    registry.replace(replace(registry.get("b"), simple="first description"))
    assert router.route("task").expert_id == "a"  # deterministic tie
    assert encoder.calls[-2] == ["first description", "first description"]


def test_adding_removing_experts_refreshes_cache(registry):
    encoder = Encoder({"first description": [1, 0], "second description": [0, 1], "task": [-1, 0]})
    router = EmbeddingRouter(registry, encoder)
    assert router.route("task").expert_id == "b"
    registry.add(Expert("c", "task", "abstract"))
    assert router.route("task").expert_id == "c"
    registry.remove("c")
    assert router.route("task").expert_id == "b"


def test_abstract_descriptions(registry):
    router = EmbeddingRouter(
        registry,
        Encoder(
            {
                "first abstract": [0, 1],
                "second abstract": [1, 0],
                "task": [1, 0],
            }
        ),
        style="abstract",
    )
    assert router.route("task").expert_id == "b"


@pytest.mark.parametrize("bad", [[0, 0], [float("nan"), 1], [float("inf"), 0], []])
def test_rejects_invalid_embeddings(registry, bad):
    with pytest.raises(RoutingError):
        EmbeddingRouter(
            registry,
            Encoder(
                {
                    "first description": bad,
                    "second description": [1, 1],
                    "task": [1, 1],
                }
            ),
        ).route("task")


def test_dimension_mismatch(registry):
    with pytest.raises(RoutingError, match="dimensions"):
        EmbeddingRouter(
            registry,
            Encoder(
                {
                    "first description": [1, 1],
                    "second description": [1, 1],
                    "task": [1],
                }
            ),
        ).route("task")


@pytest.mark.parametrize("instruction", ["", "   ", None])
def test_empty_instructions(registry, instruction):
    with pytest.raises(RoutingError):
        EmbeddingRouter(registry, Encoder({})).route(instruction)


def test_empty_registry():
    with pytest.raises(RoutingError, match="empty expert"):
        EmbeddingRouter(ExpertRegistry(), Encoder({})).route("task")


class Generator:
    def __init__(self, response):
        self.response = response

    def generate(self, messages):
        self.messages = messages
        return self.response


@pytest.mark.parametrize("response", ["1", "Output: 1", "Compare 2 capabilities.\nOutput: 1\n"])
def test_prompt_selection_and_few_shot_mapping(registry, response):
    generator = Generator(response)
    router = PromptRouter(registry, generator, examples=[PromptExample("example", "b")])
    assert router.route("new task").expert_id == "b"
    assert generator.messages[2] == {"role": "assistant", "content": "Output: 1"}
    assert json.loads(generator.messages[-1]["content"]) == {"task": "new task"}


@pytest.mark.parametrize(
    "response",
    [
        "2",
        "-1",
        "The answer is 1",
        "Output: 1 or 0",
        "Output: 0\nOutput: 1",
        "Output: 1\nextra text",
        "",
        "1.0",
        "Output: 999999",
    ],
)
def test_invalid_lm_output_never_falls_back(registry, response):
    with pytest.raises(RoutingError):
        PromptRouter(registry, Generator(response)).route("task")


@pytest.mark.parametrize("response", ["0" * 10, "x" * 4097])
def test_prompt_output_is_bounded_before_parsing(registry, response):
    with pytest.raises(RoutingError):
        PromptRouter(registry, Generator(response)).route("task")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expert_id": "", "strategy": "embedding"},
        {"expert_id": "a", "strategy": "", "scores": (("a", 1.0),)},
        {"expert_id": "a", "strategy": "embedding", "scores": (("b", 1.0),)},
        {
            "expert_id": "a",
            "strategy": "embedding",
            "scores": (("a", float("nan")),),
        },
        {
            "expert_id": "a",
            "strategy": "embedding",
            "scores": (("a", 1.0), ("a", 0.5)),
        },
        {"expert_id": "a", "strategy": "embedding", "margin": -0.1},
    ],
)
def test_routing_decision_rejects_inconsistent_metadata(kwargs):
    with pytest.raises(ValueError):
        RoutingDecision(**kwargs)


@pytest.mark.parametrize("instruction,expert_id", [("", "a"), ("task", "")])
def test_prompt_examples_require_complete_labels(instruction, expert_id):
    with pytest.raises(ValueError):
        PromptExample(instruction, expert_id)


def test_stale_example_is_reported(registry):
    with pytest.raises(RoutingError, match="missing expert"):
        PromptRouter(registry, Generator("0"), examples=[PromptExample("task", "gone")]).route(
            "task"
        )


def test_registry_rejects_duplicate_and_resolves_paths(tmp_path):
    manifest = tmp_path / "experts.json"
    manifest.write_text(
        json.dumps(
            {
                "experts": [
                    {"id": "a", "simple": "s", "abstract": "a", "adapter_path": "weights/a"},
                    {"id": "b", "simple": "s", "abstract": "a", "adapter_path": "hf://org/adapter"},
                ]
            }
        )
    )
    registry = ExpertRegistry.from_json(manifest)
    assert registry.get("a").adapter_path == str((tmp_path / "weights/a").resolve())
    assert registry.get("b").adapter_path == "org/adapter"
    with pytest.raises(ValueError, match="Duplicate"):
        registry.add(registry.get("a"))
