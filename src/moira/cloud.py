"""Cloud routing and Baseten inference clients with no Pi-side SDK dependency."""

from __future__ import annotations

import base64
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from time import sleep
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .components import ComponentDecision, ComponentRegistry, Layer


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

    def post(self, url: str, payload: Any, headers: Mapping[str, str]) -> Any:
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Cloud endpoints must use HTTPS")
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
        api_key = api_key or os.environ.get("BASETEN_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Set BASETEN_API_KEY or pass a Baseten API key")
        method = "predict" if entity == "model" else "run_remote"
        self.url = (
            f"https://{entity}-{entity_id}.api.baseten.co/environments/{environment}/{method}"
        )
        self._api_key = api_key
        self._http = http or JsonHttpClient()

    def predict(self, payload: Any) -> Any:
        return self._http.post(
            self.url,
            payload,
            {"Authorization": f"Api-Key {self._api_key}"},
        )


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
        token = token or os.environ.get(environment_key)
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
        url = f"https://chain-{chain_id}.api.baseten.co/environments/{environment}/run_remote"
        return cls(
            registry,
            url,
            token=api_key,
            auth_scheme="Api-Key",
            http=http,
        )
