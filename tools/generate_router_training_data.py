"""Create leakage-checked hard-negative triplets for the specialist router."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("instruction"), str)
            or not value["instruction"].strip()
            or not isinstance(value.get("expert_id"), str)
            or not value["expert_id"].strip()
        ):
            raise ValueError(f"Invalid routing row at {path}:{line_number}")
        rows.append(
            {
                "instruction": value["instruction"].strip(),
                "expert_id": value["expert_id"].strip(),
            }
        )
    if not rows:
        raise ValueError(f"{path} contains no routing rows")
    return rows


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog(path: Path) -> dict[str, dict[str, Any]]:
    raw = _read_json(path).get("experts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Specialist catalog must contain a non-empty experts list")
    result: dict[str, dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise TypeError("Every specialist must be an object")
        expert_id = entry.get("id")
        simple, abstract = entry.get("simple"), entry.get("abstract")
        interfaces = entry.get("interfaces")
        if (
            not isinstance(expert_id, str)
            or not expert_id.strip()
            or not isinstance(simple, str)
            or not simple.strip()
            or not isinstance(abstract, str)
            or not abstract.strip()
            or not isinstance(interfaces, list)
            or not interfaces
            or any(not isinstance(item, str) or not item.strip() for item in interfaces)
        ):
            raise ValueError(f"Malformed specialist catalog entry: {expert_id!r}")
        if expert_id in result:
            raise ValueError(f"Duplicate specialist ID: {expert_id}")
        result[expert_id] = {
            "simple": simple.strip(),
            "abstract": abstract.strip(),
            "interfaces": tuple(interfaces),
        }
    return result


def generate(
    catalog_path: Path,
    train_paths: list[Path],
    validation_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
    *,
    model_name: str,
    device: str,
) -> dict[str, Any]:
    catalog = _catalog(catalog_path)
    train_rows = [row for path in train_paths for row in _read_jsonl(path)]
    other_splits = {path.name: _read_jsonl(path) for path in validation_paths}
    unknown = sorted({row["expert_id"] for row in train_rows} - set(catalog))
    if unknown:
        raise ValueError(f"Training rows reference unknown specialists: {', '.join(unknown)}")

    train_texts = {_normalized(row["instruction"]) for row in train_rows}
    for split, rows in other_splits.items():
        overlap = train_texts & {_normalized(row["instruction"]) for row in rows}
        if overlap:
            raise ValueError(f"Train/evaluation leakage in {split}: {sorted(overlap)[0]}")

    encoder = SentenceTransformer(model_name, device=device)
    ids = list(catalog)
    class_instructions = {
        expert_id: [
            row["instruction"] for row in train_rows if row["expert_id"] == expert_id
        ]
        for expert_id in ids
    }
    negative_texts: list[tuple[str, str]] = []
    for expert_id in ids:
        for text in (
            catalog[expert_id]["simple"],
            catalog[expert_id]["abstract"],
            *class_instructions[expert_id],
        ):
            negative_texts.append((expert_id, text))
    negative_vectors = encoder.encode(
        [text for _, text in negative_texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_vectors = encoder.encode(
        [row["instruction"] for row in train_rows],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    triplets: list[dict[str, str]] = []
    negative_counts: Counter[str] = Counter()
    for row, query in zip(train_rows, query_vectors, strict=True):
        positive_id = row["expert_id"]
        positive_interfaces = set(catalog[positive_id]["interfaces"])
        candidates = [
            index
            for index, (expert_id, _text) in enumerate(negative_texts)
            if expert_id != positive_id
            and positive_interfaces.intersection(catalog[expert_id]["interfaces"])
        ]
        if not candidates:
            candidates = [
                index
                for index, (expert_id, _text) in enumerate(negative_texts)
                if expert_id != positive_id
            ]
        negative_index = max(
            candidates,
            key=lambda index: float(negative_vectors[index] @ query),
        )
        negative_id, negative_text = negative_texts[negative_index]
        negative_counts[negative_id] += 1
        positives = [catalog[positive_id]["simple"], catalog[positive_id]["abstract"]]
        positives.extend(
            text
            for text in class_instructions[positive_id]
            if _normalized(text) != _normalized(row["instruction"])
        )
        for positive_text in positives:
            triplets.append(
                {
                    "anchor": row["instruction"],
                    "positive": positive_text,
                    "negative": negative_text,
                    "expert_id": positive_id,
                    "negative_expert_id": negative_id,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in triplets),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "base_model": model_name,
        "catalog": str(catalog_path).replace("\\", "/"),
        "catalog_sha256": _sha256(catalog_path),
        "sources": {
            str(path).replace("\\", "/"): _sha256(path) for path in train_paths
        },
        "specialists": len(catalog),
        "source_examples": len(train_rows),
        "training_triplets": len(triplets),
        "examples_per_specialist": dict(
            sorted(Counter(row["expert_id"] for row in train_rows).items())
        ),
        "hard_negative_counts": dict(sorted(negative_counts.items())),
        "evaluation_splits": {
            name: {"examples": len(rows), "sha256": _sha256(path)}
            for path in validation_paths
            for name, rows in ((path.name, other_splits[path.name]),)
        },
        "leakage_check": "passed",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("examples/physical_ai_specialists.json"),
    )
    parser.add_argument(
        "--train",
        type=Path,
        nargs="+",
        default=[
            Path("examples/physical_ai_routing_samples.jsonl"),
            Path("examples/physical_ai_router_adversarial_train.jsonl"),
        ],
    )
    parser.add_argument(
        "--validation",
        type=Path,
        nargs="+",
        default=[
            Path("examples/physical_ai_routing_validation.jsonl"),
            Path("examples/physical_ai_routing_holdout.jsonl"),
            Path("examples/physical_ai_routing_test.jsonl"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/router_training/triplets.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/router_training/dataset_manifest.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = generate(
        args.catalog,
        args.train,
        args.validation,
        args.output,
        args.manifest,
        model_name=args.model,
        device=args.device,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
