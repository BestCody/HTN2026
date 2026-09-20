"""General zero-shot MoIRA router for compatible physical-AI specialists.

Deploy with: truss chains push router.py --promote
"""

import json
import math
from pathlib import Path

import truss_chains as chains
from pydantic import BaseModel, ConfigDict

MODEL_ID = "moira-specialist-router-minilm-l6-v2"
MODEL_PATH = chains.make_abs_path_here("model")
MODEL_CACHE = "/app/moira-router-models"
CATALOG_PATH = Path(__file__).with_name("specialists.json")


class RouterContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    routing_text: str | None = None


class ComponentScore(BaseModel):
    component_id: str
    score: float


class RouterResponse(BaseModel):
    component_id: str
    strategy: str
    model: str | None
    scores: list[ComponentScore]


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
        options=chains.ChainletOptions(
            env_variables={
                "HF_HUB_OFFLINE": "1",
                "SENTENCE_TRANSFORMERS_HOME": MODEL_CACHE,
            }
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
        self._encoder = SentenceTransformer(
            MODEL_PATH,
            device="cpu",
            cache_folder=MODEL_CACHE,
            local_files_only=True,
        )

    async def run_remote(
        self,
        layer: str,
        capability: str,
        context: RouterContext,
        allowed_components: list[str],
    ) -> RouterResponse:
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
            return RouterResponse(
                component_id=allowed_components[0],
                strategy="contract_singleton",
                model=None,
                scores=[],
            )
        routing_text = context.routing_text
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
            component_vectors = vectors[start:stop]
            dimensions = len(query)
            if any(len(vector) != dimensions for vector in component_vectors):
                raise ValueError("Specialist prototype embedding dimensions differ")
            centroid = tuple(
                math.fsum(float(vector[index]) for vector in component_vectors)
                / len(component_vectors)
                for index in range(dimensions)
            )
            norm = math.hypot(*centroid)
            if norm == 0 or not math.isfinite(norm):
                raise ValueError("Specialist prototype centroid has zero norm")
            scores.append(
                math.fsum(
                    value * float(query[index]) / norm
                    for index, value in enumerate(centroid)
                )
            )
            start = stop
        winner = max(range(len(scores)), key=scores.__getitem__)
        return RouterResponse(
            component_id=allowed_components[winner],
            strategy="minilm_prototype_centroid_cosine",
            model=MODEL_ID,
            scores=[
                ComponentScore(component_id=component_id, score=score)
                for component_id, score in zip(allowed_components, scores, strict=True)
            ],
        )
