import base64
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class FakeWhisperModel:
    init_args = None
    transcribe_options = None

    def __init__(self, *args, **kwargs):
        type(self).init_args = (args, kwargs)

    def transcribe(self, audio_path, **options):
        assert Path(audio_path).read_bytes() == b"fake-wav"
        type(self).transcribe_options = options
        segments = iter(
            [
                SimpleNamespace(text=" Pick up", start=0.1, end=0.5),
                SimpleNamespace(text=" the red block. ", start=0.5, end=1.2),
            ]
        )
        info = SimpleNamespace(language="en", language_probability=0.99, duration=1.2)
        return segments, info


def load_deployment_module(monkeypatch):
    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)

    spec = importlib.util.spec_from_file_location(
        "test_baseten_whisper_module", "deploy/baseten_whisper/model/model.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_whisper_deployment_uses_l4_and_turbo_checkpoint():
    config = Path("deploy/baseten_whisper/config.yaml").read_text(encoding="utf-8")
    assert "instance_type: L4:4x16" in config
    assert "faster-whisper-large-v3-turbo" in config
    assert "RTX-PRO-6000" not in config


def test_whisper_deployment_matches_moira_speech_contract(monkeypatch):
    module = load_deployment_module(monkeypatch)
    model = module.Model(
        config={"model_metadata": {"model_id": "/models/faster-whisper-large-v3-turbo"}}
    )
    model.load()

    result = model.predict(
        {"audio": {"base64": base64.b64encode(b"fake-wav").decode("ascii")}}
    )

    assert result["text"] == "Pick up the red block."
    assert result["language"] == "en"
    assert FakeWhisperModel.init_args == (
        ("/models/faster-whisper-large-v3-turbo",),
        {"device": "cuda", "compute_type": "float16", "local_files_only": True},
    )
    assert FakeWhisperModel.transcribe_options["language"] == "en"
    assert FakeWhisperModel.transcribe_options["vad_filter"] is True
    assert FakeWhisperModel.transcribe_options["condition_on_previous_text"] is False


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "audio.base64"),
        ({"audio": {"base64": "%%%"}}, "valid base64"),
        ({"audio": {"base64": ""}}, "empty file"),
    ],
)
def test_whisper_deployment_rejects_invalid_audio(monkeypatch, payload, message):
    module = load_deployment_module(monkeypatch)
    with pytest.raises(ValueError, match=message):
        module.Model._decode_audio(payload)
