"""Run the complete non-actuating Charlie integration path."""

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
                "execute": run.result.control.executed,
                "transcript": run.transcript,
                "objects": [item.id for item in run.result.world.objects],
                "personal_accommodations": list(run.result.personal.accommodations),
                "routed_components": list(run.routed_components),
                "candidate_count": len(run.result.candidates),
                "simulated_horizon_seconds": [
                    item.horizon_seconds for item in run.result.simulations
                ],
                "selected_plan": run.result.plan.candidate.id,
                "selected_safe": run.result.plan.simulation.safe,
                "world_models": list(run.world_models),
                "outcome": run.result.outcome.status if run.result.outcome else None,
                "camera_verification": run.result.world_after is not None,
                "feedback_logged": config.journal_path.is_file(),
                "response_audio_bytes": len(run.result.response_audio or b""),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
