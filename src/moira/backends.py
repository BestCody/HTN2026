"""Optional pretrained text backends; importing the core never downloads models."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Any

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SMOLLM_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 64


class SentenceTransformerEncoder:
    def __init__(
        self,
        model_name: str = MINILM_MODEL,
        *,
        device: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Embedding model name must be a non-empty string")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("Embedding device must be a non-empty string or null")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("Embedding revision must be a non-empty string or null")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be boolean")
        self.model_name, self.device = model_name, device
        self.revision, self.local_files_only = revision, local_files_only
        self._model: Any = None
        self._lock = RLock()

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or len(texts) == 0:
            raise ValueError("Embedding input must be a non-empty sequence of strings")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding texts must be non-empty strings")
        with self._lock:
            return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    'Install the embedding backend: pip install -e ".[embeddings]"'
                ) from exc
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                revision=self.revision,
                local_files_only=self.local_files_only,
                trust_remote_code=False,
            )
            self._model.eval()
        return self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()


class TransformersGenerator:
    def __init__(
        self,
        model_name: str = SMOLLM_MODEL,
        *,
        device: str = "cpu",
        revision: str | None = None,
        local_files_only: bool = False,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Language model name must be a non-empty string")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("Language model device must be a non-empty string")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("Language model revision must be a non-empty string or null")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be boolean")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be positive")
        self.model_name, self.device = model_name, device
        self.revision, self.local_files_only = revision, local_files_only
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = RLock()

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not isinstance(messages, list) or not messages:
            raise ValueError("Messages must be a non-empty list")
        if any(
            not isinstance(message, dict)
            or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            for message in messages
        ):
            raise ValueError("Each message needs a valid role and non-empty string content")
        with self._lock:
            return self._generate(messages)

    def _generate(self, messages: list[dict[str, str]]) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError('Install the prompt backend: pip install -e ".[prompt]"') from exc
        if self._model is None:
            options = dict(
                revision=self.revision,
                local_files_only=self.local_files_only,
                trust_remote_code=False,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **options)
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype="auto",
                    **options,
                )
                .to(self.device)
                .eval()
            )
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_length = inputs["input_ids"].shape[-1]
        limits = [getattr(self._model.config, "max_position_embeddings", None)]
        tokenizer_limit = getattr(self._tokenizer, "model_max_length", None)
        # Tokenizers use enormous sentinel values to mean "unknown".
        if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000_000:
            limits.append(tokenizer_limit)
        valid_limits = [limit for limit in limits if isinstance(limit, int) and limit > 0]
        context_length = min(valid_limits) if valid_limits else None
        if context_length and input_length + self.max_new_tokens > context_length:
            raise ValueError("Expert descriptions and examples exceed the LM context window")
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self._model.config, "pad_token_id", None)
        generation_options = {}
        if pad_token_id is not None:
            generation_options["pad_token_id"] = pad_token_id
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                **generation_options,
            )
        return self._tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
