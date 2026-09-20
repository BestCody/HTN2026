"""Live smoke test for the configured Model APIs, Chains, and RTX voice LAN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.software_integration import (  # noqa: E402
    load_software_integration_config,
    run_software_integration,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "software_integration.json",
    )
    args = parser.parse_args()
    config = load_software_integration_config(args.config)
    run = run_software_integration(config)
    print(
        json.dumps(
            {
                "status": "passed",
                "transcript": run.transcript,
                "objects": [item.id for item in run.result.world.objects],
                "candidates": len(run.result.candidates),
                "selected": run.result.plan.candidate.id,
                "executed": run.result.control.executed,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
