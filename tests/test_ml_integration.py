"""Exercise real training/loading/switching and generation without downloads."""

import pytest

pytestmark = pytest.mark.integration


def tiny_policy():
    import torch

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1, bias=False)
            torch.nn.init.zeros_(self.linear.weight)

        def forward(self, x):
            return self.linear(x)

    return Policy()


@pytest.mark.parametrize("mode", ["disk", "multi"])
def test_real_lora_training_and_switching(tmp_path, mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from moira import AdapterServer, Expert
    from moira.adapters import PeftAdapterBackend
    from moira.training import LoraTrainingConfig, train_specialist

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    experts = []
    for name, target in [("positive.v1", 1.0), ("negative.v1", -1.0)]:
        base = tiny_policy()
        original = base.linear.weight.detach().clone()
        path = tmp_path / name
        rng_state = torch.random.get_rng_state()
        losses = train_specialist(
            base,
            [(x, torch.full((2, 1), target))],
            lambda model, batch: ((model(batch[0]) - batch[1]) ** 2).mean(),
            path,
            LoraTrainingConfig(("linear",), steps=40, rank=2, alpha=2, learning_rate=0.1),
        )
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        assert losses[-1] < losses[0] / 10
        assert torch.equal(base.linear.base_layer.weight, original)
        assert (path / "adapter_config.json").exists()
        experts.append(Expert(name, "simple", "abstract", str(path)))
    backend = PeftAdapterBackend(tiny_policy(), lambda model, observation, _: model(observation))
    server = AdapterServer(backend, mode=mode)
    if mode == "multi":
        server.preload(tuple(experts))
    for index in (0, 1, 0):
        with server.session(experts[index]) as policy:
            policy.reset("instruction")
            output = policy.act(x)
            assert not output.requires_grad
            assert not any(p.requires_grad for p in backend.model.parameters())
            assert float(output.mean()) * (1 if index == 0 else -1) > 0.7
    server.close()
    assert not server.loaded_ids


def test_local_transformers_generation(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from moira.backends import TransformersGenerator

    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[EOS]": 1, "task": 2, "Output": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}Output "
    tokenizer.save_pretrained(tmp_path)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            eos_token_id=1,
            pad_token_id=1,
        )
    )
    model.save_pretrained(tmp_path)
    generator = TransformersGenerator(str(tmp_path), local_files_only=True, max_new_tokens=3)
    messages = [{"role": "user", "content": "task task task task task"}]
    first = generator.generate(messages)
    assert first == generator.generate(messages)
    assert len(first.split()) <= 3  # prompt was removed from the generated text
    with pytest.raises(ValueError, match="context window"):
        generator.generate([{"role": "user", "content": "task " * 64}])


def test_training_rejects_nonfinite_loss_without_publishing_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from moira.training import LoraTrainingConfig, train_specialist

    x = torch.tensor([[1.0, 0.0]])
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="finite"):
        train_specialist(
            tiny_policy(),
            [x],
            lambda model, batch: model(batch).sum() * float("nan"),
            output,
            LoraTrainingConfig(("linear",), steps=1),
        )
    assert not output.exists()
