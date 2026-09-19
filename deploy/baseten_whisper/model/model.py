"""Whisper Large V3 Turbo endpoint matching MoIRA's typed speech contract."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from faster_whisper import WhisperModel

MAX_AUDIO_BYTES = 8 * 1024 * 1024


class Model:
    def __init__(self, **kwargs: Any) -> None:
        config = kwargs["config"]
        self._model_path = config["model_metadata"]["model_id"]
        self._model: WhisperModel | None = None

    def load(self) -> None:
        self._model = WhisperModel(
            self._model_path,
            device="cuda",
            compute_type="float16",
            local_files_only=True,
        )

    @staticmethod
    def _decode_audio(request: dict[str, Any]) -> bytes:
        audio = request.get("audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("base64"), str):
            raise ValueError("audio.base64 must contain a base64-encoded audio file")
        try:
            payload = base64.b64decode(audio["base64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("audio.base64 is not valid base64") from exc
        if not payload:
            raise ValueError("audio.base64 decoded to an empty file")
        if len(payload) > MAX_AUDIO_BYTES:
            raise ValueError("audio exceeds the 8 MiB command limit")
        return payload

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Whisper model is not loaded")

        audio = self._decode_audio(request)
        with NamedTemporaryFile(suffix=".audio", delete=False) as temporary:
            temporary.write(audio)
            audio_path = Path(temporary.name)

        try:
            segments, info = self._model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                word_timestamps=False,
            )
            result_segments = [
                {"text": segment.text.strip(), "start": segment.start, "end": segment.end}
                for segment in segments
                if segment.text.strip()
            ]
        finally:
            audio_path.unlink(missing_ok=True)

        text = " ".join(segment["text"] for segment in result_segments).strip()
        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": result_segments,
        }
