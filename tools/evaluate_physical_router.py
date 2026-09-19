"""Evaluate the frozen physical-AI router globally and by compatible interface."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from moira.backends import MINILM_MODEL, SentenceTransformerEncoder
from moira.evaluation import evaluate_routing, load_samples
from moira.experts import ExpertRegistry
from moira.routing import PrototypeEmbeddingRouter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experts", type=Path, default=Path("examples/physical_ai_specialists.json")
    )
    parser.add_argument(
        "--samples", type=Path, default=Path("examples/physical_ai_routing_test.jsonl")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default=MINILM_MODEL)
    parser.add_argument("--style", choices=("simple", "abstract"), default="simple")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    registry = ExpertRegistry.from_json(args.experts)
    experts = registry.snapshot()
    samples = load_samples(args.samples)
    encoder = SentenceTransformerEncoder(
        args.model,
        device=args.device,
        local_files_only=args.offline,
    )
    full_labels = [expert.id for expert in experts]
    report = {
        "router": "frozen_minilm_prototype_cosine",
        "model": args.model,
        "device": args.device,
        "style": args.style,
        "catalog": str(args.experts),
        "samples": str(args.samples),
        "full_pool": evaluate_routing(
            PrototypeEmbeddingRouter(registry, encoder, style=args.style), samples, full_labels
        ),
        "interfaces": {},
    }

    interface_experts: dict[str, list[str]] = defaultdict(list)
    for expert in experts:
        for interface in expert.interfaces:
            interface_experts[interface].append(expert.id)
    for interface, labels in sorted(interface_experts.items()):
        scoped_samples = [sample for sample in samples if sample.expert_id in labels]
        if len(labels) < 2 or not scoped_samples:
            continue
        scoped = registry.for_interface(interface)
        report["interfaces"][interface] = evaluate_routing(
            PrototypeEmbeddingRouter(scoped, encoder, style=args.style),
            scoped_samples,
            labels,
        )

    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

