"""Build a compact acceptance report from router evaluation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("full_pool"), dict):
        raise ValueError(f"Malformed router evaluation: {path}")
    return value


def _metrics(value: dict[str, Any]) -> dict[str, Any]:
    full = value["full_pool"]
    interfaces = value.get("interfaces", {})
    return {
        "count": full["count"],
        "accuracy": full["accuracy"],
        "macro_f1": full["macro_f1"],
        "manipulation_accuracy": interfaces.get("manipulation.policy.v1", {}).get("accuracy"),
        "world_accuracy": interfaces.get("world.predict.v1", {}).get("accuracy"),
        "invalid_predictions": full["invalid_predictions"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = _metrics(_load(args.baseline))
    splits = {
        "validation": _metrics(_load(args.validation)),
        "holdout": _metrics(_load(args.holdout)),
        "test": _metrics(_load(args.test)),
    }
    accepted = (
        splits["validation"]["accuracy"] >= 0.95
        and splits["holdout"]["accuracy"] >= 0.95
        and splits["test"]["accuracy"] >= 0.95
        and all(value["manipulation_accuracy"] == 1.0 for value in splits.values())
        and all(value["world_accuracy"] == 1.0 for value in splits.values())
        and all(value["invalid_predictions"] == 0 for value in splits.values())
    )
    result = {
        "schema_version": 1,
        "accepted": accepted,
        "acceptance_rule": (
            "validation, holdout, and test >= 0.95; manipulation and world interface "
            "accuracy = 1.0; zero invalid predictions"
        ),
        "baseline": baseline,
        "trained": splits,
        "checkpoint_model_sha256": hashlib.sha256(
            (args.checkpoint / "model.safetensors").read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
