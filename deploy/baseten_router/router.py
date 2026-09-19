"""General zero-shot MoIRA router for compatible physical-AI specialists.

Deploy with: truss chains push router.py --promote
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import truss_chains as chains

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CATALOG_PATH = Path(__file__).with_name("specialists.json")


@chains.mark_entrypoint
class PhysicalComponentRouter(chains.ChainletBase):
    """Rank a caller-supplied compatible pool from textual expert metadata.

    The Pi remains authoritative for compatibility and sends only component IDs
    that satisfy the requested interface/capability. This Chain never turns a
    task into a hardcoded component ID and never returns an unapproved ID.
    """

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            requirements_file=chains.make_abs_path_here("requirements.txt"),
        ),
    )

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        entries = data.get("experts") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise ValueError("Specialist catalog must contain an experts list")
        self._prototypes: dict[str, tuple[str, ...]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Every specialist catalog entry must be an object")
            expert_id, description = entry.get("id"), entry.get("simple")
            examples = entry.get("routing_examples", [])
            if (
                not isinstance(expert_id, str)
                or not expert_id.strip()
                or not isinstance(description, str)
                or not description.strip()
                or not isinstance(examples, list)
                or any(not isinstance(item, str) or not item.strip() for item in examples)
                or len(set(examples)) != len(examples)
                or len(examples) > 16
            ):
                raise ValueError(
                    "Every specialist needs an id, description, and valid routing examples"
                )
            if expert_id in self._prototypes:
                raise ValueError(f"Duplicate specialist ID: {expert_id}")
            self._prototypes[expert_id] = (description, *examples)
        self._encoder = SentenceTransformer(MODEL_ID, device="cpu")

    async def run_remote(
        self,
        layer: str,
        capability: str,
        context: dict[str, Any],
        allowed_components: list[str],
    ) -> dict[str, Any]:
        if not isinstance(layer, str) or not layer.strip():
            raise ValueError("layer must be a non-empty string")
        if not isinstance(capability, str) or not capability.startswith(f"{layer}."):
            raise ValueError("Capability does not belong to the requested layer")
        if (
            not isinstance(allowed_components, list)
            or not allowed_components
            or any(not isinstance(item, str) or not item.strip() for item in allowed_components)
            or len(set(allowed_components)) != len(allowed_components)
        ):
            raise ValueError("allowed_components must contain unique non-empty IDs")
        if len(allowed_components) == 1:
            return {
                "component_id": allowed_components[0],
                "strategy": "contract_singleton",
                "model": None,
                "scores": [],
            }
        if not isinstance(context, dict):
            raise ValueError("context must be an object")
        routing_text = context.get("routing_text")
        if not isinstance(routing_text, str) or not routing_text.strip():
            raise ValueError("routing_text is required when multiple specialists are compatible")
        missing = [item for item in allowed_components if item not in self._prototypes]
        if missing:
            raise ValueError(f"Compatible specialists missing from catalog: {', '.join(missing)}")

        prototypes = [self._prototypes[item] for item in allowed_components]
        texts = [text for component in prototypes for text in component]
        vectors = self._encoder.encode(
            [*texts, routing_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query = vectors[-1]
        scores = []
        start = 0
        for component in prototypes:
            stop = start + len(component)
            scores.append(max(float(vector @ query) for vector in vectors[start:stop]))
            start = stop
        winner = max(range(len(scores)), key=scores.__getitem__)
        return {
            "component_id": allowed_components[winner],
            "strategy": "minilm_prototype_cosine",
            "model": MODEL_ID,
            "scores": [
                {"component_id": component_id, "score": score}
                for component_id, score in zip(allowed_components, scores, strict=True)
            ],
        }
