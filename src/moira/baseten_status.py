"""Read-only Baseten deployment inventory for the production runtime."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .cloud import load_runtime_environment
from .production import PhysicalRuntimeConfig

EntityKind = Literal["model", "chain"]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class BasetenManagementClient:
    """Small read-only client for the Baseten Management API."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 15.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Baseten API key must be a non-empty string")
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = api_key
        self._timeout_seconds = float(timeout_seconds)
        self._opener = opener

    def _get(self, path: str) -> Mapping[str, Any]:
        if not path.startswith("/") or ".." in path:
            raise ValueError("Baseten Management API path is invalid")
        request = Request(
            f"https://api.baseten.co/v1{path}",
            headers={"Authorization": f"Api-Key {self._api_key}"},
            method="GET",
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            detail = exc.read(1024).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Baseten Management API returned HTTP {exc.code}: {detail}"
            ) from exc
        except (TimeoutError, URLError) as exc:
            raise ConnectionError("Baseten Management API is unavailable") from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("Baseten Management API response is too large")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("Baseten Management API response must be a JSON object")
        return value

    def inspect(self, entity: EntityKind, entity_id: str, environment: str) -> dict[str, Any]:
        for value, name in ((entity_id, "entity ID"), (environment, "environment")):
            if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"Baseten {name} contains unsupported characters")
        if entity == "model":
            return self._inspect_model(entity_id, environment)
        if entity == "chain":
            return self._inspect_chain(entity_id, environment)
        raise ValueError("Baseten entity must be model or chain")

    def _inspect_model(self, model_id: str, environment: str) -> dict[str, Any]:
        if environment == "development":
            deployment = self._get(f"/models/{model_id}/deployments/development")
            return {
                "reachable": True,
                "environment_found": True,
                "deployment_status": deployment.get("status"),
                "active_replicas": deployment.get("active_replica_count"),
                "instance_type": deployment.get("instance_type_name"),
            }
        response = self._get(f"/models/{model_id}/environments")
        environments = response.get("environments")
        if not isinstance(environments, list):
            raise TypeError("Baseten model environment response needs an environments list")
        selected = next(
            (
                item
                for item in environments
                if isinstance(item, dict) and item.get("name") == environment
            ),
            None,
        )
        if selected is None:
            return {
                "reachable": True,
                "environment_found": False,
                "deployment_status": None,
                "active_replicas": None,
                "instance_type": None,
            }
        deployment = selected.get("current_deployment")
        deployment = deployment if isinstance(deployment, dict) else {}
        instance = selected.get("instance_type")
        instance = instance if isinstance(instance, dict) else {}
        return {
            "reachable": True,
            "environment_found": True,
            "deployment_status": deployment.get("status"),
            "active_replicas": deployment.get("active_replica_count"),
            "instance_type": instance.get("name") or deployment.get("instance_type_name"),
        }

    def _inspect_chain(self, chain_id: str, environment: str) -> dict[str, Any]:
        response = self._get(f"/chains/{chain_id}/deployments")
        deployments = response.get("deployments")
        if not isinstance(deployments, list):
            raise TypeError("Baseten chain deployment response needs a deployments list")
        matching = [
            item
            for item in deployments
            if isinstance(item, dict)
            and (
                item.get("environment") == environment
                or (environment == "development" and item.get("environment") is None)
            )
        ]
        selected = max(matching, key=lambda item: str(item.get("created_at", "")), default=None)
        if selected is None:
            return {
                "reachable": True,
                "environment_found": False,
                "deployment_status": None,
                "chainlets": (),
            }
        chainlets = selected.get("chainlets")
        chainlets = chainlets if isinstance(chainlets, list) else []
        return {
            "reachable": True,
            "environment_found": True,
            "deployment_status": selected.get("status"),
            "chainlets": tuple(
                {
                    "name": item.get("name"),
                    "status": item.get("status"),
                    "active_replicas": item.get("active_replica_count"),
                    "instance_type": item.get("instance_type_name"),
                }
                for item in chainlets
                if isinstance(item, dict)
            ),
        }


def inspect_baseten_deployments(
    config: PhysicalRuntimeConfig,
    *,
    live: bool = True,
    environ: Mapping[str, str] | None = None,
    client: BasetenManagementClient | None = None,
) -> dict[str, Any]:
    """Report configuration and optional live health without exposing credentials."""

    load_runtime_environment()
    values = environ if environ is not None else os.environ
    api_key = values.get("BASETEN_API_KEY")
    if client is None and live:
        if not api_key:
            raise ValueError("BASETEN_API_KEY is not set")
        client = BasetenManagementClient(api_key)

    deployment_specifications = [
        (
            "semantic-router",
            config.router.id_env,
            "chain",
            config.router.environment_env,
            config.router.default_environment,
        ),
        *(
            (
                component_id,
                endpoint.id_env,
                endpoint.entity,
                endpoint.environment_env,
                endpoint.default_environment,
            )
            for component_id, endpoint in config.remote_components.items()
            if endpoint.transport == "baseten_deployment"
        ),
    ]
    endpoints: dict[str, Any] = {}
    live_errors: list[str] = []
    for component_id, id_env, entity, environment_env, default_environment in (
        deployment_specifications
    ):
        if id_env is None or entity is None or environment_env is None:
            raise ValueError(f"Baseten deployment configuration is incomplete: {component_id}")
        entity_id = values.get(id_env)
        environment = values.get(environment_env, default_environment)
        report: dict[str, Any] = {
            "entity": entity,
            "id_environment_variable": id_env,
            "configured": bool(entity_id),
            "environment": environment,
            "live_checked": False,
        }
        if live and entity_id:
            assert client is not None
            try:
                report.update(client.inspect(entity, entity_id, environment))
                report["live_checked"] = True
            except (ValueError, TypeError, OSError, RuntimeError, ConnectionError) as exc:
                report.update({"live_checked": True, "reachable": False, "error": str(exc)})
                live_errors.append(component_id)
        endpoints[component_id] = report

    for component_id, endpoint in config.remote_components.items():
        if endpoint.transport == "baseten_deployment":
            continue
        variable = endpoint.required_environment_variable
        configured = variable is None or bool(values.get(variable))
        if endpoint.transport == "baseten_model_api":
            configured = bool(
                (values.get(variable) if variable else None) or endpoint.default_model
            )
        endpoints[component_id] = {
            "transport": endpoint.transport,
            "configuration_environment_variable": variable,
            "configured": configured,
            "live_checked": False,
            "model": (
                (values.get(endpoint.model_env) if endpoint.model_env else None)
                or endpoint.default_model
                if endpoint.transport == "baseten_model_api"
                else None
            ),
        }

    configured = sum(bool(value["configured"]) for value in endpoints.values())
    healthy = sum(
        bool(value.get("environment_found"))
        and value.get("deployment_status") in {"ACTIVE", "SCALED_TO_ZERO"}
        for value in endpoints.values()
    )
    return {
        "api_key_present": bool(api_key),
        "live": live,
        "configured_endpoints": configured,
        "total_endpoints": len(endpoints),
        "healthy_configured_endpoints": healthy if live else None,
        "endpoints": endpoints,
        "live_errors": tuple(live_errors),
    }
