"""Kokoro speech synthesis endpoint matching MoIRA's TTS contract."""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from typing import Any

MAX_TEXT_CHARS = 1_500
SAMPLE_RATE = 24_000


class Model:
    def __init__(self, **kwargs: Any) -> None:
        metadata = kwargs["config"]["model_metadata"]
        self._model_path = Path(metadata["model_id"])
        self._voice = metadata["default_voice"]
        self._pipeline: Any = None

    def load(self) -> None:
        from kokoro import KModel, KPipeline

        model = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(self._model_path / "config.json"),
            model=str(self._model_path / "kokoro-v1_0.pth"),
        ).to("cuda").eval()
        self._pipeline = KPipeline(
            lang_code="a",
            repo_id="hexgrad/Kokoro-82M",
            model=model,
            device="cuda",
        )

    @staticmethod
    def _wav_bytes(chunks: list[Any]) -> bytes:
        import numpy as np

        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        waveform = np.concatenate(
            [
                chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else chunk
                for chunk in chunks
            ]
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise RuntimeError("Kokoro produced an invalid waveform")
        pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
        return output.getvalue()

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._pipeline is None:
            raise RuntimeError("TTS model is not loaded")
        text = request.get("text")
        user_id = request.get("user_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError("text exceeds the 1,500-character speech limit")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be non-empty")
        voice_path = str(self._model_path / "voices" / f"{self._voice}.pt")
        chunks = [audio for _, _, audio in self._pipeline(text.strip(), voice=voice_path)]
        audio = self._wav_bytes(chunks)
        return {
            "audio_b64": base64.b64encode(audio).decode("ascii"),
            "media_type": "audio/wav",
            "sample_rate_hz": SAMPLE_RATE,
            "voice": self._voice,
        }
