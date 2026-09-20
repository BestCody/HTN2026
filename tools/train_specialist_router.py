"""Fine-tune the MiniLM bi-encoder used for metadata-driven specialist routing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import (
    TripletDistanceMetric,
    TripletLoss,
)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _read_triplets(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        required = ("anchor", "positive", "negative")
        if (
            not isinstance(value, dict)
            or any(
                not isinstance(value.get(key), str) or not value[key].strip()
                for key in required
            )
        ):
            raise ValueError(f"Invalid triplet at {path}:{line_number}")
        rows.append({key: value[key].strip() for key in required})
    if not rows:
        raise ValueError("Router training dataset is empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("outputs/router_training/triplets.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/specialist-router"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the queued hackathon router training run")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = _read_triplets(args.train)
    dataset = Dataset.from_list(rows)
    model = SentenceTransformer(args.model, device="cuda")
    loss = TripletLoss(
        model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=0.18,
    )
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output / "trainer_state"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=0.1,
        weight_decay=0.01,
        fp16=True,
        tf32=True,
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    metadata = {
        "schema_version": 1,
        "base_model": args.model,
        "training_examples": len(rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "objective": "cosine triplet loss over query, specialist description, hard negative",
    }
    (args.output / "moira_training.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
