import io
import json
from urllib.error import HTTPError

import pytest

from moira.baseten_status import BasetenManagementClient, inspect_baseten_deployments
from moira.production import load_physical_runtime_config


class Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, _limit):
        return json.dumps(self.value).encode()


def test_management_client_summarizes_model_without_exposing_credentials():
    requests = []

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        return Response(
            {
                "status": "ACTIVE",
                "active_replica_count": 1,
                "instance_type_name": "L4:4x16",
            }
        )

    result = BasetenManagementClient("secret", opener=open_request).inspect(
        "model", "model_123", "development"
    )

    assert result == {
        "reachable": True,
        "environment_found": True,
        "deployment_status": "ACTIVE",
        "active_replicas": 1,
        "instance_type": "L4:4x16",
    }
    assert requests[0][0].full_url.endswith("/v1/models/model_123/deployments/development")
    assert requests[0][0].get_header("Authorization") == "Api-Key secret"
    assert "secret" not in json.dumps(result)


def test_management_client_selects_latest_chain_environment_deployment():
    def open_request(_request, *, timeout):
        return Response(
            {
                "deployments": [
                    {
                        "created_at": "2026-01-01T00:00:00Z",
                        "environment": "development",
                        "status": "FAILED",
                    },
                    {
                        "created_at": "2026-01-02T00:00:00Z",
                        "environment": "development",
                        "status": "ACTIVE",
                        "chainlets": [
                            {
                                "name": "Router",
                                "status": "ACTIVE",
                                "active_replica_count": 0,
                                "instance_type_name": "1x4",
                            }
                        ],
                    },
                ]
            }
        )

    result = BasetenManagementClient("secret", opener=open_request).inspect(
        "chain", "chain_123", "development"
    )

    assert result["deployment_status"] == "ACTIVE"
    assert result["chainlets"] == (
        {
            "name": "Router",
            "status": "ACTIVE",
            "active_replicas": 0,
            "instance_type": "1x4",
        },
    )


def test_inventory_lists_every_remote_endpoint_without_values(monkeypatch):
    config = load_physical_runtime_config("config/pi4_runtime.json")
    environment = {
        "BASETEN_API_KEY": "secret",
        "MOIRA_STT_URL": "http://127.0.0.1:8765/v1/stt",
    }

    result = inspect_baseten_deployments(config, live=False, environ=environment)

    assert result["api_key_present"] is True
    # The two GLM Model APIs have explicit defaults and STT has a LAN URL.
    assert result["configured_endpoints"] == 3
    assert result["endpoints"]["baseten-stt"]["configured"] is True
    assert result["endpoints"]["baseten-stt"]["transport"] == "json_http"
    assert result["endpoints"]["baseten-vision-scene"]["configured"] is True
    assert result["endpoints"]["semantic-router"]["configured"] is False
    assert "secret" not in json.dumps(result)


def test_management_error_redacts_request_authorization():
    def open_request(request, *, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    client = BasetenManagementClient("secret", opener=open_request)
    with pytest.raises(RuntimeError, match="HTTP 401: bad key") as error:
        client.inspect("model", "model_123", "development")
    assert "secret" not in str(error.value)
