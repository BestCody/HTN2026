import json

import pytest

from moira import (
    Expert,
    ExpertRegistry,
    HybridRouter,
    InMemoryServer,
    MoIRA,
    PromptExample,
    RoutingError,
)
from moira.cli import main


class Encoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return [self.vectors[text] for text in texts]


class Generator:
    def __init__(self, response="Output: 1"):
        self.response, self.calls = response, []

    def generate(self, messages):
        self.calls.append(messages)
        return self.response


@pytest.fixture
def registry():
    return ExpertRegistry([Expert("a", "first", "one"), Expert("b", "second", "two")])


def test_small_margin_is_resolved_with_lm_and_original_instruction(registry):
    encoder = Encoder({"first": [1, 0], "second": [1, 0.01], "task": [1, 0]})
    generator = Generator()
    router = HybridRouter(registry, encoder, generator, examples=[PromptExample("example", "b")])
    decision = router.route("task")
    assert decision.expert_id == "b"
    assert decision.strategy == "hybrid_prompt"
    assert decision.embedding_expert_id == "a"
    assert 0 < decision.margin < router.min_margin
    assert dict(decision.scores)["a"] > dict(decision.scores)["b"]
    assert json.loads(generator.calls[0][-1]["content"]) == {"task": "task"}
    assert generator.calls[0][2]["content"] == "Output: 1"


def test_large_margin_does_not_load_or_call_lm(registry):
    generator = Generator("invalid")
    router = HybridRouter(
        registry,
        Encoder({"first": [1, 0], "second": [0, 1], "task": [0, 1]}),
        generator,
    )
    decision = router.route("task")
    assert decision.expert_id == "b" and decision.strategy == "hybrid_embedding"
    assert decision.margin == pytest.approx(1)
    assert not generator.calls


def test_exact_tie_uses_lm_instead_of_insertion_order(registry):
    generator = Generator()
    decision = HybridRouter(
        registry,
        Encoder({"first": [1], "second": [1], "task": [1]}),
        generator,
    ).route("task")
    assert decision.expert_id == "b"
    assert decision.margin == 0
    # With no user examples, metadata supplies one neutral demonstration per label.
    assert generator.calls[0][2] == {"role": "assistant", "content": "Output: 0"}
    assert generator.calls[0][4] == {"role": "assistant", "content": "Output: 1"}


def test_generated_examples_follow_registry_description_changes(registry):
    generator = Generator()
    router = HybridRouter(
        registry,
        Encoder({"first": [1], "second": [1], "updated": [1], "task": [1]}),
        generator,
    )
    router.route("task")
    assert json.loads(generator.calls[-1][1]["content"]) == {"task": "first"}
    registry.replace(Expert("a", "updated", "one"))
    router.route("task")
    assert json.loads(generator.calls[-1][1]["content"]) == {"task": "updated"}


def test_single_expert_needs_no_disambiguation(registry):
    registry.remove("b")
    generator = Generator()
    decision = HybridRouter(
        registry,
        Encoder({"first": [1, 0], "task": [1, 0]}),
        generator,
    ).route("task")
    assert decision.expert_id == "a" and decision.margin is None
    assert not generator.calls


def test_configurable_margin_boundary_and_description_changes(registry):
    encoder = Encoder({"first": [1, 0], "second": [0.8, 0.6], "task": [1, 0]})
    generator = Generator()
    probe = HybridRouter(registry, encoder, generator)
    margin = probe.embedding.route("task").margin
    router = HybridRouter(registry, encoder, generator, min_margin=margin)
    assert router.route("task").strategy == "hybrid_embedding"
    registry.replace(Expert("b", "first", "one"))
    assert router.route("task").strategy == "hybrid_prompt"


@pytest.mark.parametrize("margin", [0, -0.1, 2.01, float("nan"), float("inf")])
def test_invalid_margin_configuration(registry, margin):
    with pytest.raises(ValueError, match="min_margin"):
        HybridRouter(registry, min_margin=margin)


def test_invalid_disambiguation_does_not_execute_any_policy(registry):
    class Policy:
        resets = []

        def reset(self, instruction):
            self.resets.append(instruction)

        def act(self, observation):
            raise AssertionError("No action should be executed")

    policy = Policy()
    router = HybridRouter(
        registry,
        Encoder({"first": [1], "second": [1], "task": [1]}),
        Generator("nonsense"),
    )
    controller = MoIRA(registry, router, InMemoryServer({"a": policy, "b": policy}))
    with pytest.raises(RoutingError):
        with controller.episode("task"):
            pytest.fail("Invalid routing must not open an episode")
    assert policy.resets == []


def test_unavailable_fallback_does_not_use_ambiguous_embedding_guess(registry):
    class UnavailableGenerator:
        def generate(self, messages):
            raise OSError("Model is not cached")

    router = HybridRouter(
        registry,
        Encoder({"first": [1], "second": [1], "task": [1]}),
        UnavailableGenerator(),
    )
    with pytest.raises(OSError, match="not cached"):
        router.route("task")


def test_cli_defaults_to_hybrid_and_keeps_model_revisions_separate(tmp_path, monkeypatch, capsys):
    import moira.cli as cli

    manifest = tmp_path / "experts.json"
    manifest.write_text(
        json.dumps(
            {
                "experts": [
                    {"id": "a", "simple": "first", "abstract": "one"},
                    {"id": "b", "simple": "second", "abstract": "two"},
                ]
            }
        )
    )
    constructors = []

    def encoder(model, **options):
        constructors.append(("embedding", model, options))
        return Encoder({"first": [1], "second": [1], "task": [1]})

    def generator(model, **options):
        constructors.append(("prompt", model, options))
        return Generator()

    monkeypatch.setattr(cli, "SentenceTransformerEncoder", encoder)
    monkeypatch.setattr(cli, "TransformersGenerator", generator)
    arguments = [
        "route",
        "--experts",
        str(manifest),
        "--model",
        "embedding-model",
        "--revision",
        "embedding-revision",
        "--fallback-model",
        "prompt-model",
        "--fallback-revision",
        "prompt-revision",
        "--offline",
        "task",
    ]
    assert main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["expert_id"] == "b" and output["strategy"] == "hybrid_prompt"
    assert constructors == [
        (
            "embedding",
            "embedding-model",
            {
                "device": "cpu",
                "revision": "embedding-revision",
                "local_files_only": True,
            },
        ),
        (
            "prompt",
            "prompt-model",
            {
                "device": "cpu",
                "revision": "prompt-revision",
                "local_files_only": True,
                "max_new_tokens": 64,
            },
        ),
    ]
    constructors.clear()
    embedding_arguments = [
        "route",
        "--experts",
        str(manifest),
        "--router",
        "embedding",
        "--model",
        "embedding-model",
        "--revision",
        "embedding-revision",
        "--offline",
        "task",
    ]
    assert main(embedding_arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["expert_id"] == "a" and output["strategy"] == "embedding"
    assert len(constructors) == 1  # explicit paper baseline never loads the LM


@pytest.mark.parametrize(
    "option",
    [
        ["--fallback-model", "lm"],
        ["--fallback-revision", "commit"],
        ["--min-margin", "0.1"],
        ["--max-new-tokens", "4"],
    ],
)
def test_cli_rejects_options_that_embedding_mode_cannot_use(tmp_path, option):
    manifest = tmp_path / "experts.json"
    manifest.write_text(
        json.dumps({"experts": [{"id": "a", "simple": "first", "abstract": "one"}]})
    )
    with pytest.raises(SystemExit) as error:
        main(
            [
                "route",
                "--experts",
                str(manifest),
                "--router",
                "embedding",
                *option,
                "task",
            ]
        )
    assert error.value.code == 2
