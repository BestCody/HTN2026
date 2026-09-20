"""Exercise the deployed semantic router Chain without loading robot hardware."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from moira.cloud import BasetenEndpoint, JsonHttpClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Allow a scaled-to-zero CPU Chain to wake (default: 180)",
    )
    args = parser.parse_args()
    load_dotenv()
    chain_id = os.environ.get("BASETEN_ROUTER_CHAIN_ID")
    if not chain_id:
        raise SystemExit("Set BASETEN_ROUTER_CHAIN_ID in .env after deployment")
    environment = os.environ.get("BASETEN_ROUTER_ENVIRONMENT", "development")
    endpoint = BasetenEndpoint(
        chain_id,
        entity="chain",
        environment=environment,
        http=JsonHttpClient(timeout_seconds=args.timeout_seconds, attempts=1),
    )
    allowed = ["baseten-waypoint-policy", "baseten-bimanual-act"]
    response = endpoint.predict(
        {
            "layer": "manipulation",
            "capability": "manipulation.select_compatible",
            "context": {"routing_text": "Coordinate both arms to lift the tray together"},
            "allowed_components": allowed,
        }
    )
    if not isinstance(response, dict) or response.get("component_id") not in allowed:
        raise SystemExit("Router returned an invalid or out-of-pool component")
    if response["component_id"] != "baseten-bimanual-act":
        raise SystemExit(
            "Router smoke test chose the wrong specialist: " + response["component_id"]
        )
    print(response["component_id"])


if __name__ == "__main__":
    main()
