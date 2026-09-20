"""Wake configured production Baseten models before a physical demo."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from moira.baseten_status import BasetenManagementClient
from moira.production import load_physical_runtime_config


def _wake(component_id: str, model_id: str, api_key: str) -> tuple[str, int]:
    request = Request(
        f"https://model-{model_id}.api.baseten.co/wake",
        data=b"",
        headers={"Authorization": f"Api-Key {api_key}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            response.read(1024)
            return component_id, response.status
    except HTTPError as exc:
        detail = exc.read(1024).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Wake request for {component_id} returned HTTP {exc.code}: {detail}"
        ) from exc
    except (TimeoutError, URLError) as exc:
        raise ConnectionError(f"Wake request for {component_id} failed") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pi4_runtime.json"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise ValueError("timeouts must be positive")

    load_dotenv(override=False)
    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise RuntimeError("Set BASETEN_API_KEY in .env")
    config = load_physical_runtime_config(args.config)
    targets: dict[str, tuple[str, str]] = {}
    skipped: list[str] = []
    for component_id, endpoint in config.remote_components.items():
        if endpoint.transport != "baseten_deployment" or endpoint.entity != "model":
            continue
        if endpoint.id_env is None:
            continue
        model_id = os.environ.get(endpoint.id_env)
        if not model_id:
            continue
        environment = os.environ.get(
            endpoint.environment_env,
            endpoint.default_environment,
        )
        if environment != "production":
            skipped.append(component_id)
            continue
        targets[component_id] = (model_id, environment)
    if not targets:
        raise RuntimeError("No configured production Baseten models were found")

    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        accepted = dict(
            executor.map(
                lambda item: _wake(item[0], item[1][0], api_key),
                targets.items(),
            )
        )

    client = BasetenManagementClient(api_key, timeout_seconds=30)
    statuses: dict[str, str | None] = {}
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        statuses = {
            component_id: client.inspect("model", model_id, environment).get("deployment_status")
            for component_id, (model_id, environment) in targets.items()
        }
        if all(status == "ACTIVE" for status in statuses.values()):
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "wake_http_status": accepted,
                        "deployments": statuses,
                        "skipped_nonproduction_models": sorted(skipped),
                    },
                    indent=2,
                )
            )
            return 0
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Baseten models did not become active before timeout: {statuses}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
