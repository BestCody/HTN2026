import asyncio
import importlib.util
import sys
from types import ModuleType

import pytest


class Vector(tuple):
    def __matmul__(self, other):
        return sum(left * right for left, right in zip(self, other, strict=True))


class FakeEncoder:
    def __init__(self, model_id, *, device):
        self.model_id, self.device = model_id, device

    def encode(self, texts, **options):
        assert options == {"normalize_embeddings": True, "convert_to_numpy": True}
        return [
            Vector((1.0, 0.0))
            if "two arm" in text.lower() or "two-arm" in text.lower()
            else Vector((0.0, 1.0))
            for text in texts
        ]


def load_deployment_module(monkeypatch):
    chains = ModuleType("truss_chains")

    class ChainletBase:
        pass

    class Configuration:
        def __init__(self, **values):
            self.values = values

    chains.ChainletBase = ChainletBase
    chains.RemoteConfig = Configuration
    chains.DockerImage = Configuration
    chains.make_abs_path_here = lambda value: value
    chains.mark_entrypoint = lambda value: value
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeEncoder
    monkeypatch.setitem(sys.modules, "truss_chains", chains)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    spec = importlib.util.spec_from_file_location(
        "test_baseten_router_module", "deploy/baseten_router/router.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_deployable_router_semantically_ranks_only_allowed_specialists(monkeypatch):
    module = load_deployment_module(monkeypatch)
    router = module.PhysicalComponentRouter()

    result = asyncio.run(
        router.run_remote(
            "manipulation",
            "manipulation.select_compatible",
            {"routing_text": "Coordinate two arms to lift this together"},
            ["baseten-waypoint-policy", "baseten-bimanual-act"],
        )
    )

    assert result["component_id"] == "baseten-bimanual-act"
    assert result["strategy"] == "minilm_prototype_cosine"
    assert {item["component_id"] for item in result["scores"]} == {
        "baseten-waypoint-policy",
        "baseten-bimanual-act",
    }


def test_deployable_router_has_no_hidden_substitution(monkeypatch):
    module = load_deployment_module(monkeypatch)
    router = module.PhysicalComponentRouter()

    singleton = asyncio.run(
        router.run_remote("control", "control.single_arm", {}, ["edge-hardware"])
    )
    assert singleton["component_id"] == "edge-hardware"
    assert singleton["strategy"] == "contract_singleton"

    with pytest.raises(ValueError, match="missing from catalog"):
        asyncio.run(
            router.run_remote(
                "manipulation",
                "manipulation.select_compatible",
                {"routing_text": "pick up the object"},
                ["baseten-waypoint-policy", "unregistered-policy"],
            )
        )
