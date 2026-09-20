"""Pinned RTX speech specialists used by the LAN inference service."""

from __future__ import annotations

import base64
import binascii
import io
import os
import tempfile
import wave
from pathlib import Path
from threading import Lock
from typing import Any

WHISPER_REPOSITORY = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
WHISPER_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
KOKORO_REPOSITORY = "hexgrad/Kokoro-82M"
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 1_500
SAMPLE_RATE = 24_000
_DLL_HANDLES: list[Any] = []


def configure_windows_cuda_dlls() -> None:
    """Expose CUDA 12/cuDNN 9 DLLs bundled with the verified PyTorch wheel."""

    if os.name != "nt":
        return
    import torch

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    required = ("cublas64_12.dll", "cudnn64_9.dll")
    missing = [name for name in required if not (torch_lib / name).is_file()]
    if missing:
        raise RuntimeError("PyTorch CUDA runtime is missing: " + ", ".join(missing))
    _DLL_HANDLES.append(os.add_dll_directory(str(torch_lib)))
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


def _base64_payload(value: Any, name: str) -> bytes:
    if not isinstance(value, dict) or not isinstance(value.get("base64"), str):
        raise ValueError(f"{name}.base64 must contain a base64 payload")
    try:
        payload = base64.b64decode(value["base64"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{name}.base64 is invalid") from exc
    if not payload or len(payload) > MAX_AUDIO_BYTES:
        raise ValueError(f"{name} must contain 1 byte to 8 MiB")
    return payload


class FasterWhisperRTX:
    """Whisper Large V3 Turbo with a pinned checkpoint and mandatory CUDA."""

    def __init__(
        self,
        *,
        repository: str = WHISPER_REPOSITORY,
        revision: str = WHISPER_REVISION,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.repository = repository
        self.revision = revision
        self.cache_dir = str(Path(cache_dir).resolve()) if cache_dir else None
        self._model: Any = None
        self._lock = Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            configure_windows_cuda_dlls()
            import ctranslate2
            from faster_whisper import WhisperModel
            from huggingface_hub import snapshot_download

            if ctranslate2.get_cuda_device_count() < 1:
                raise RuntimeError("Faster-Whisper requires the RTX CUDA device")
            model_source = self.repository
            if self.cache_dir:
                snapshot = Path(self.cache_dir) / self.revision
                snapshot_download(
                    self.repository,
                    revision=self.revision,
                    local_dir=snapshot,
                    allow_patterns=(
                        "config.json",
                        "model.bin",
                        "preprocessor_config.json",
                        "tokenizer.json",
                        "vocabulary.*",
                    ),
                )
                model_source = str(snapshot)
            self._model = WhisperModel(
                model_source,
                device="cuda",
                compute_type="float16",
                local_files_only=bool(self.cache_dir),
            )

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        audio = _base64_payload(request.get("audio"), "audio")
        self.load()
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as temporary:
            temporary.write(audio)
            path = Path(temporary.name)
        try:
            segments, info = self._model.transcribe(
                path,
                language="en",
                task="transcribe",
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                word_timestamps=False,
            )
            values = [
                {"text": item.text.strip(), "start": item.start, "end": item.end}
                for item in segments
                if item.text.strip()
            ]
        finally:
            path.unlink(missing_ok=True)
        return {
            "text": " ".join(item["text"] for item in values).strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": values,
        }


class KokoroRTX:
    """Kokoro-82M speech synthesis with a pinned checkpoint and voice."""

    def __init__(
        self,
        *,
        repository: str = KOKORO_REPOSITORY,
        revision: str = KOKORO_REVISION,
        voice: str = "af_heart",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.repository = repository
        self.revision = revision
        self.voice = voice
        self.cache_dir = str(Path(cache_dir).resolve()) if cache_dir else None
        self._pipeline: Any = None
        self._voice_path: str | None = None
        self._lock = Lock()

    def load(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                return
            configure_windows_cuda_dlls()
            import torch
            from huggingface_hub import snapshot_download
            from kokoro import KModel, KPipeline

            if not torch.cuda.is_available():
                raise RuntimeError("Kokoro requires the RTX CUDA device")
            local_dir = (
                Path(self.cache_dir) / self.revision if self.cache_dir else None
            )
            snapshot = Path(
                snapshot_download(
                    self.repository,
                    revision=self.revision,
                    local_dir=local_dir,
                )
            )
            model = KModel(
                repo_id=self.repository,
                config=str(snapshot / "config.json"),
                model=str(snapshot / "kokoro-v1_0.pth"),
            ).to("cuda").eval()
            self._voice_path = str(snapshot / "voices" / f"{self.voice}.pt")
            if not Path(self._voice_path).is_file():
                raise RuntimeError(f"Kokoro voice is absent from the pinned snapshot: {self.voice}")
            self._pipeline = KPipeline(
                lang_code="a",
                repo_id=self.repository,
                model=model,
                device="cuda",
            )

    @staticmethod
    def _wav_bytes(chunks: list[Any]) -> bytes:
        import numpy as np

        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        waveform = np.concatenate(
            [item.detach().cpu().numpy() if hasattr(item, "detach") else item for item in chunks]
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise RuntimeError("Kokoro produced an invalid waveform")
        pcm = (np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
        return output.getvalue()

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        text = request.get("text")
        user_id = request.get("user_id")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            raise ValueError("text must contain 1 to 1,500 characters")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be non-empty")
        self.load()
        chunks = [audio for _, _, audio in self._pipeline(text.strip(), voice=self._voice_path)]
        audio = self._wav_bytes(chunks)
        return {
            "audio_b64": base64.b64encode(audio).decode("ascii"),
            "media_type": "audio/wav",
            "sample_rate_hz": SAMPLE_RATE,
            "voice": self.voice,
        }
