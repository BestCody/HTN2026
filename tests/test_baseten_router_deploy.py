import ast
import asyncio
import importlib.util
import sys
from types import ModuleType

import pytest


class Vector(tuple):
    def __matmul__(self, other):
        return sum(left * right for left, right in zip(self, other, strict=True))


class FakeEncoder:
    def __init__(self, model_id, *, device, cache_folder, local_files_only):
        self.model_id, self.device = model_id, device
        assert cache_folder == "/app/moira-router-models"
        assert local_files_only is True

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
    chains.ChainletOptions = Configuration
    chains.make_abs_path_here = lambda value: value
    chains.mark_entrypoint = lambda value: value
    pydantic = ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **values):
            for key, value in values.items():
                setattr(self, key, value)
            annotations = getattr(type(self), "__annotations__", {})
            for key in annotations:
                if not hasattr(self, key):
                    setattr(self, key, getattr(type(self), key, None))

        def model_dump(self):
            def convert(value):
                if isinstance(value, BaseModel):
                    return value.model_dump()
                if isinstance(value, list):
                    return [convert(item) for item in value]
                return value

            return {
                key: convert(getattr(self, key))
                for key in getattr(type(self), "__annotations__", {})
            }

    pydantic.BaseModel = BaseModel
    pydantic.ConfigDict = lambda **values: values
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeEncoder
    monkeypatch.setitem(sys.modules, "truss_chains", chains)
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)
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
            module.RouterContext(routing_text="Coordinate two arms to lift this together"),
            ["baseten-waypoint-policy", "baseten-bimanual-act"],
        )
    ).model_dump()

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
        router.run_remote(
            "control", "control.single_arm", module.RouterContext(), ["edge-hardware"]
        )
    ).model_dump()
    assert singleton["component_id"] == "edge-hardware"
    assert singleton["strategy"] == "contract_singleton"

    with pytest.raises(ValueError, match="missing from catalog"):
        asyncio.run(
            router.run_remote(
                "manipulation",
                "manipulation.select_compatible",
                module.RouterContext(routing_text="pick up the object"),
                ["baseten-waypoint-policy", "unregistered-policy"],
            )
        )


def test_router_keeps_runtime_types_for_truss_chain_validation():
    module = ast.parse(open("deploy/baseten_router/router.py", encoding="utf-8").read())
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in module.body
    )
