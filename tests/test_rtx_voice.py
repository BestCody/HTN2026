import base64
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from moira.rtx_voice import FasterWhisperRTX, KokoroRTX, _base64_payload


def test_audio_payload_is_strict_and_bounded():
    assert _base64_payload({"base64": base64.b64encode(b"wav").decode()}, "audio") == b"wav"
    with pytest.raises(ValueError, match="invalid"):
        _base64_payload({"base64": "%%%"}, "audio")
    with pytest.raises(ValueError, match="1 byte"):
        _base64_payload({"base64": ""}, "audio")


def test_whisper_adapter_joins_segments_without_loading_checkpoint():
    class Model:
        def transcribe(self, path, **options):
            assert path.is_file()
            assert options["beam_size"] == 5
            return (
                [
                    SimpleNamespace(text=" Pick up the red block", start=0.0, end=1.0),
                    SimpleNamespace(text=" and place it.", start=1.0, end=2.0),
                ],
                SimpleNamespace(language="en", language_probability=0.99, duration=2.0),
            )

    whisper = FasterWhisperRTX()
    whisper._model = Model()
    result = whisper.predict(
        {"audio": {"base64": base64.b64encode(b"encoded audio").decode()}}
    )
    assert result["text"] == "Pick up the red block and place it."
    assert result["language"] == "en"
    assert len(result["segments"]) == 2


def test_kokoro_adapter_returns_valid_pcm_wav_without_loading_checkpoint():
    class Pipeline:
        def __call__(self, text, *, voice):
            assert text == "Ready."
            assert voice == "voice.pt"
            return [(None, None, np.array([0.0, 0.5, -0.5], dtype=np.float32))]

    kokoro = KokoroRTX()
    kokoro._pipeline = Pipeline()
    kokoro._voice_path = "voice.pt"
    result = kokoro.predict({"text": "Ready.", "user_id": "demo"})
    raw = base64.b64decode(result["audio_b64"])
    with wave.open(io.BytesIO(raw), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 3
