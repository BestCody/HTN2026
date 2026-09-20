"""Cloud routing and Baseten inference clients with no Pi-side SDK dependency."""

from __future__ import annotations

import base64
import ipaddress
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import sleep
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from .components import ComponentDecision, ComponentRegistry, Layer


def load_runtime_environment() -> Path:
    """Load the explicit runtime env file, or `.env` in the current directory."""

    configured_path = os.environ.get("MOIRA_ENV_FILE")
    env_path = Path(configured_path).expanduser() if configured_path else Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    return env_path


def _environment_value(name: str) -> str | None:
    """Read process configuration after loading the canonical runtime env file."""

    load_runtime_environment()
    return os.environ.get(name)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


class JsonHttpClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        max_request_bytes: int = 8 * 1024 * 1024,
        max_response_bytes: int = 16 * 1024 * 1024,
        attempts: int = 2,
        allow_private_http: bool = False,
    ) -> None:
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        for name, value in (
            ("max_request_bytes", max_request_bytes),
            ("max_response_bytes", max_response_bytes),
            ("attempts", attempts),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.timeout_seconds = float(timeout_seconds)
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.attempts = attempts
        if not isinstance(allow_private_http, bool):
            raise TypeError("allow_private_http must be boolean")
        self.allow_private_http = allow_private_http

    @staticmethod
    def _is_private_http(url: str) -> bool:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
            return False
        hostname = parsed.hostname.casefold()
        if hostname in {"localhost", "host.docker.internal"} or hostname.endswith(".local"):
            return True
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return False
        return address.is_private or address.is_loopback

    def post(self, url: str, payload: Any, headers: Mapping[str, str]) -> Any:
        if not isinstance(url, str) or not (
            url.startswith("https://")
            or (self.allow_private_http and self._is_private_http(url))
        ):
            raise ValueError("Endpoints must use HTTPS or an explicitly enabled private HTTP URL")
        body = json.dumps(_jsonable(payload), separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        if len(body) > self.max_request_bytes:
            raise ValueError("Cloud request exceeds the configured byte limit")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **dict(headers)},
            method="POST",
        )
        for attempt in range(self.attempts):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read(self.max_response_bytes + 1)
                if len(raw) > self.max_response_bytes:
                    raise ValueError("Cloud response exceeds the configured byte limit")
                return json.loads(raw.decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read(2048).decode("utf-8", errors="replace")
                if (exc.code < 500 and exc.code not in (408, 429)) or attempt + 1 == self.attempts:
                    raise RuntimeError(
                        f"Cloud endpoint returned HTTP {exc.code}: {detail}"
                    ) from exc
            except (TimeoutError, URLError) as exc:
                if attempt + 1 == self.attempts:
                    raise ConnectionError(f"Cloud endpoint is unavailable: {url}") from exc
            sleep(0.25 * (2**attempt))
        raise AssertionError("unreachable")


class BasetenEndpoint:
    """Invoke a production Baseten model or Chain environment."""

    def __init__(
        self,
        entity_id: str,
        *,
        entity: Literal["model", "chain"] = "model",
        environment: str = "production",
        api_key: str | None = None,
        http: JsonHttpClient | None = None,
    ) -> None:
        identifier = re.compile(r"^[A-Za-z0-9_-]+$")
        if not isinstance(entity_id, str) or not identifier.fullmatch(entity_id):
            raise ValueError("Baseten entity_id contains unsupported characters")
        if entity not in ("model", "chain"):
            raise ValueError("Baseten entity must be model or chain")
        if not isinstance(environment, str) or not identifier.fullmatch(environment):
            raise ValueError("Baseten environment contains unsupported characters")
        api_key = api_key or _environment_value("BASETEN_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Set BASETEN_API_KEY or pass a Baseten API key")
        method = "predict" if entity == "model" else "run_remote"
        deployment_path = (
            "development" if environment == "development" else f"environments/{environment}"
        )
        self.url = f"https://{entity}-{entity_id}.api.baseten.co/{deployment_path}/{method}"
        self._api_key = api_key
        self._http = http or JsonHttpClient()

    def predict(self, payload: Any) -> Any:
        return self._http.post(
            self.url,
            payload,
            {"Authorization": f"Api-Key {self._api_key}"},
        )


class JsonEndpoint:
    """Invoke a typed JSON specialist on a configured HTTPS or private LAN URL."""

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        http: JsonHttpClient | None = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("JSON specialist URL must be non-empty")
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise ValueError("JSON specialist token must be non-empty or null")
        self.url = url
        self.token = token
        self._http = http or JsonHttpClient(allow_private_http=True)

    def predict(self, payload: Any) -> Any:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        return self._http.post(self.url, payload, headers)


class BasetenModelAPI:
    """Call a shared Baseten Model API through its OpenAI-compatible endpoint.

    Model APIs are billed from Model API credits and do not create a dedicated
    deployment.  This client deliberately exposes only strict JSON completion,
    which keeps every physical-AI boundary typed and independently validated.
    """

    def __init__(
        self,
        model: str = "zai-org/GLM-5.3-Flash",
        *,
        api_key: str | None = None,
        http: JsonHttpClient | None = None,
        reasoning_effort: Literal["low", "high", "max"] = "low",
    ) -> None:
        if (
            not isinstance(model, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{1,199}", model)
            or ".." in model
        ):
            raise ValueError("Baseten Model API model name is invalid")
        if reasoning_effort not in ("low", "high", "max"):
            raise ValueError("reasoning_effort must be low, high, or max")
        api_key = api_key or _environment_value("BASETEN_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Set BASETEN_API_KEY or pass a Baseten API key")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.url = "https://inference.baseten.co/v1/chat/completions"
        self._api_key = api_key
        self._http = http or JsonHttpClient(timeout_seconds=60, attempts=2)

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: Mapping[str, Any],
        name: str,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("Model API messages must be a non-empty list")
        if not isinstance(schema, Mapping) or schema.get("type") != "object":
            raise ValueError("Model API response schema must describe an object")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
            raise ValueError("Model API schema name is invalid")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        response = self._http.post(
            self.url,
            {
                "model": self.model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": True,
                        "schema": dict(schema),
                    },
                },
                "temperature": 0,
                "max_tokens": max_tokens,
                "reasoning_effort": self.reasoning_effort,
            },
            {"Authorization": f"Bearer {self._api_key}"},
        )
        if not isinstance(response, dict):
            raise TypeError("Baseten Model API response must be a JSON object")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("Baseten Model API response needs exactly one choice")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("Baseten Model API choice needs textual JSON content")
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Baseten Model API returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Baseten Model API JSON result must be an object")
        return value


class BasetenComponent:
    """Adapt a specialized Baseten deployment to the component `run` contract."""

    def __init__(
        self,
        endpoint: BasetenEndpoint,
        *,
        encode: Callable[[Any], Any] = _jsonable,
        decode: Callable[[Any, Any], Any] = lambda value, request: value,
    ) -> None:
        if not isinstance(endpoint, BasetenEndpoint):
            raise TypeError("endpoint must be a BasetenEndpoint")
        if not callable(encode) or not callable(decode):
            raise TypeError("Baseten component codecs must be callable")
        self.endpoint, self.encode, self.decode = endpoint, encode, decode

    def run(self, request: Any) -> Any:
        return self.decode(self.endpoint.predict(self.encode(request)), request)


class JsonComponent:
    """Adapt a LAN or HTTPS JSON endpoint to the component `run` contract."""

    def __init__(
        self,
        endpoint: JsonEndpoint,
        *,
        encode: Callable[[Any], Any] = _jsonable,
        decode: Callable[[Any, Any], Any] = lambda value, request: value,
    ) -> None:
        if not isinstance(endpoint, JsonEndpoint):
            raise TypeError("endpoint must be a JsonEndpoint")
        if not callable(encode) or not callable(decode):
            raise TypeError("JSON component codecs must be callable")
        self.endpoint, self.encode, self.decode = endpoint, encode, decode

    def run(self, request: Any) -> Any:
        return self.decode(self.endpoint.predict(self.encode(request)), request)


class RemoteComponentRouter:
    """Ask a cloud router for one allow-listed component ID."""

    def __init__(
        self,
        registry: ComponentRegistry,
        url: str,
        *,
        token: str | None = None,
        auth_scheme: Literal["Bearer", "Api-Key"] = "Bearer",
        http: JsonHttpClient | None = None,
    ) -> None:
        if not isinstance(registry, ComponentRegistry):
            raise TypeError("registry must be a ComponentRegistry")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Router URL must use HTTPS")
        if auth_scheme not in ("Bearer", "Api-Key"):
            raise ValueError("Router auth_scheme must be Bearer or Api-Key")
        environment_key = "BASETEN_API_KEY" if auth_scheme == "Api-Key" else "PHYSICAL_ROUTER_TOKEN"
        token = token or _environment_value(environment_key)
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise ValueError("Router token must be a non-empty string or null")
        self.registry, self.url, self.token, self.auth_scheme = (
            registry,
            url,
            token,
            auth_scheme,
        )
        self.http = http or JsonHttpClient(timeout_seconds=5, attempts=2)

    def decide(
        self,
        layer: Layer,
        capability: str,
        context: Mapping[str, Any] | None = None,
    ) -> ComponentDecision:
        headers = {"Authorization": f"{self.auth_scheme} {self.token}"} if self.token else {}
        response = self.http.post(
            self.url,
            {
                "layer": layer.value,
                "capability": capability,
                "context": dict(context or {}),
                "allowed_components": [
                    spec.id
                    for spec in self.registry.specs
                    if spec.layer is layer and capability in spec.capabilities
                ],
            },
            headers,
        )
        if not isinstance(response, dict) or not isinstance(response.get("component_id"), str):
            raise ValueError("Router response must contain component_id")
        return self.registry.decision_for(
            layer,
            capability,
            response["component_id"],
            router="remote",
        )

    def select_compatible(
        self,
        layer: Layer,
        capabilities: tuple[str, ...],
        routing_text: str,
        context: Mapping[str, Any] | None = None,
    ) -> ComponentDecision:
        """Semantically select across compatible capabilities without a route table.

        The caller defines the typed compatibility pool. The remote router sees
        only component IDs that the edge registry has already admitted.
        """
        if isinstance(layer, str):
            try:
                layer = Layer(layer)
            except ValueError as exc:
                raise ValueError(f"Unknown component layer: {layer}") from exc
        if not isinstance(layer, Layer):
            raise TypeError("layer must be a Layer")
        if (
            not isinstance(capabilities, tuple)
            or not capabilities
            or len(set(capabilities)) != len(capabilities)
            or any(
                not isinstance(capability, str)
                or not capability.startswith(f"{layer.value}.")
                for capability in capabilities
            )
        ):
            raise ValueError(
                f"capabilities must contain unique values starting with '{layer.value}.'"
            )
        if not isinstance(routing_text, str) or not routing_text.strip():
            raise ValueError("routing_text must be a non-empty string")
        candidates = [
            spec
            for spec in self.registry.specs
            if spec.layer is layer
            and any(capability in spec.capabilities for capability in capabilities)
            and (self.registry.allow_remote or spec.runtime != "remote")
        ]
        if not candidates:
            raise LookupError(f"No compatible components are registered for layer: {layer.value}")
        current_context = {**dict(context or {}), "routing_text": routing_text}
        headers = {"Authorization": f"{self.auth_scheme} {self.token}"} if self.token else {}
        response = self.http.post(
            self.url,
            {
                "layer": layer.value,
                "capability": f"{layer.value}.select_compatible",
                "context": current_context,
                "allowed_components": [spec.id for spec in candidates],
            },
            headers,
        )
        if not isinstance(response, dict) or not isinstance(response.get("component_id"), str):
            raise ValueError("Router response must contain component_id")
        selected_id = response["component_id"]
        selected = next((spec for spec in candidates if spec.id == selected_id), None)
        if selected is None:
            raise LookupError(
                f"Router selected a component outside the compatible pool: {selected_id}"
            )
        matched = tuple(
            capability for capability in capabilities if capability in selected.capabilities
        )
        if len(matched) != 1:
            raise LookupError(
                f"Selected component must provide exactly one requested capability: {selected_id}"
            )
        return self.registry.decision_for(
            layer,
            matched[0],
            selected_id,
            router="remote_semantic",
        )

    @classmethod
    def from_baseten_chain(
        cls,
        registry: ComponentRegistry,
        chain_id: str,
        *,
        environment: str = "production",
        api_key: str | None = None,
        http: JsonHttpClient | None = None,
    ) -> RemoteComponentRouter:
        identifier = re.compile(r"^[A-Za-z0-9_-]+$")
        if not identifier.fullmatch(chain_id) or not identifier.fullmatch(environment):
            raise ValueError("Baseten chain or environment contains unsupported characters")
        deployment_path = (
            "development" if environment == "development" else f"environments/{environment}"
        )
        url = f"https://chain-{chain_id}.api.baseten.co/{deployment_path}/run_remote"
        return cls(
            registry,
            url,
            token=api_key,
            auth_scheme="Api-Key",
            http=http,
        )
