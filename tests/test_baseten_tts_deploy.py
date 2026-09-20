import base64
import importlib.util
import io
import wave
from pathlib import Path

import numpy as np


def load_module():
    spec = importlib.util.spec_from_file_location(
        "test_baseten_tts_module", "deploy/baseten_tts/model/model.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakePipeline:
    def __call__(self, text, *, voice):
        assert text == "Task complete."
        assert voice.endswith("voices\\af_heart.pt") or voice.endswith("voices/af_heart.pt")
        yield text, "phonemes", np.asarray([0.0, 0.5, -0.5], dtype=np.float32)


def test_tts_deployment_is_pinned_to_kokoro_and_gpu():
    config = Path("deploy/baseten_tts/config.yaml").read_text(encoding="utf-8")
    assert "hexgrad/Kokoro-82M@f3ff357" in config
    assert "kokoro==0.9.4" in config
    assert "instance_type: L4:4x16" in config


def test_tts_returns_a_real_wav_contract():
    module = load_module()
    model = module.Model(
        config={
            "model_metadata": {
                "model_id": "/models/kokoro-82m",
                "default_voice": "af_heart",
            }
        }
    )
    model._pipeline = FakePipeline()
    result = model.predict({"text": "Task complete.", "user_id": "demo-user"})
    raw = base64.b64decode(result["audio_b64"], validate=True)
    with wave.open(io.BytesIO(raw), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 3
