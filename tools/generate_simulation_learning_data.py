"""Generate fixed-elbow simulation episodes from CAD and recorded servo commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.simulation_learning import (  # noqa: E402
    generate_simulation_dataset,
    load_simulation_learning_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "simulation_learning.json",
    )
    args = parser.parse_args()
    result = generate_simulation_dataset(load_simulation_learning_config(args.config))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
