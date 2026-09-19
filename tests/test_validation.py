import pytest

from moira.adapters import PeftAdapterBackend
from moira.backends import SentenceTransformerEncoder, TransformersGenerator
from moira.training import LoraTrainingConfig, train_specialist


class FailingActivationModel:
    def set_adapter(self, name):
        raise RuntimeError("activation failed")


def test_peft_backend_fails_closed_after_activation_or_reset_error():
    backend = PeftAdapterBackend(FailingActivationModel(), lambda model, observation, text: text)
    backend._names["new"] = "adapter"
    backend._active = "old"
    backend._instruction = "stale instruction"
    with pytest.raises(RuntimeError, match="activation failed"):
        backend.activate_adapter("new")
    assert backend.active_id is None
    with pytest.raises(RuntimeError, match="Activate an adapter"):
        backend.act(None)

    def fail_reset(model, instruction):
        raise RuntimeError("reset failed")

    backend = PeftAdapterBackend(
        object(), lambda model, observation, text: text, reset_fn=fail_reset
    )
    backend._active = "active"
    backend._instruction = "stale instruction"
    with pytest.raises(RuntimeError, match="reset failed"):
        backend.reset("new instruction")
    with pytest.raises(RuntimeError, match="reset the episode"):
        backend.act(None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"steps": True},
        {"rank": 0},
        {"dropout": float("nan")},
        {"learning_rate": float("inf")},
        {"weight_decay": -1},
        {"max_grad_norm": 0},
        {"seed": False},
        {"target_modules": ("linear", "linear")},
    ],
)
def test_training_config_rejects_unsafe_values(overrides):
    values = {"target_modules": ("linear",), "steps": 1, **overrides}
    with pytest.raises(ValueError):
        LoraTrainingConfig(**values)


def test_training_rejects_invalid_collaborators_before_loading_ml_dependencies(tmp_path):
    config = LoraTrainingConfig(("linear",), steps=1)
    with pytest.raises(TypeError, match="base_model"):
        train_specialist(object(), (), lambda model, batch: None, tmp_path / "out", config)
    with pytest.raises(TypeError, match="loss_fn"):
        train_specialist(
            type("Model", (), {"parameters": lambda self: ()})(), (), None, tmp_path / "out", config
        )


def test_optional_backend_inputs_are_validated_without_loading_models():
    with pytest.raises(TypeError, match="local_files_only"):
        SentenceTransformerEncoder(local_files_only=1)
    with pytest.raises(ValueError, match="non-empty sequence"):
        SentenceTransformerEncoder().encode("one string is not a batch")
    with pytest.raises(ValueError, match="max_new_tokens"):
        TransformersGenerator(max_new_tokens=True)
    with pytest.raises(ValueError, match="valid role"):
        TransformersGenerator().generate([{"role": "tool", "content": "result"}])
