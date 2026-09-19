import ast
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from moira.cloud import BasetenComponent, BasetenEndpoint, RemoteComponentRouter
from moira.components import ComponentDecision, ComponentRegistry, ComponentSpec, Layer
from moira.contracts import decode_physical_response
from moira.physical import PersonalContext, VoiceGroundingInput, WorldState
from moira.pi import PiRuntimeProfile


class Echo:
    def __init__(self, name="echo"):
        self.name = name
        self.closed = False

    def run(self, request):
        return self.name, request

    def close(self):
        self.closed = True


def spec(component_id, ram=1, *, concurrency=1, runtime="local"):
    return ComponentSpec(
        component_id,
        Layer.VOICE,
        ("voice.ground",),
        f"{component_id}-model",
        runtime=runtime,
        estimated_ram_mb=ram,
        max_concurrency=concurrency,
    )


def test_component_router_requires_exact_capabilities_and_evicts_idle_lru():
    with pytest.raises(ValueError, match="exact"):
        ComponentSpec("bad", Layer.VOICE, ("*",), "general-model")

    registry = ComponentRegistry(ram_budget_mb=10)
    first, second = Echo("first"), Echo("second")
    registry.register(spec("first", 6), lambda: first)
    registry.register(spec("second", 6), lambda: second)
    first_decision = registry.decision_for(Layer.VOICE, "voice.ground", "first", router="test")
    second_decision = registry.decision_for(Layer.VOICE, "voice.ground", "second", router="test")
    assert registry.invoke(first_decision, "request") == ("first", "request")
    assert registry.loaded_ids == ("first",)
    assert registry.invoke(second_decision, "request") == ("second", "request")
    assert first.closed and registry.loaded_ids == ("second",)


def test_registry_rejects_forged_router_decision_metadata_before_invocation():
    registry = ComponentRegistry(ram_budget_mb=10)
    component = Echo("voice")
    registry.register(spec("voice"), lambda: component)
    forged = ComponentDecision(
        "voice.ground",
        "voice",
        Layer.VOICE,
        "different-model",
        "local",
        (),
        "untrusted-router",
    )
    with pytest.raises(LookupError, match="metadata does not match"):
        registry.invoke(forged, "request")
    assert registry.loaded_ids == ()


def test_active_component_is_not_evicted_and_lease_expires():
    registry = ComponentRegistry(ram_budget_mb=10)
    registry.register(spec("first", 6), Echo)
    registry.register(spec("second", 6), Echo)
    with registry.session("first") as active:
        with pytest.raises(MemoryError):
            with registry.session("second"):
                pass
    with pytest.raises(RuntimeError, match="closed"):
        active.run("late")


def test_component_concurrency_is_explicit():
    barrier = Barrier(2)

    class Parallel:
        def run(self, request):
            barrier.wait(timeout=2)
            return request

    registry = ComponentRegistry(ram_budget_mb=10)
    registry.register(spec("parallel", concurrency=2), Parallel)
    decision = registry.decide(Layer.VOICE, "voice.ground")
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(lambda value: registry.invoke(decision, value), (1, 2))) == {1, 2}


class FakeHttp:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def post(self, url, payload, headers):
        self.calls.append((url, payload, headers))
        if self.error:
            raise self.error
        return self.response


def test_remote_router_only_accepts_allow_list_and_has_no_failure_fallback():
    registry = ComponentRegistry(ram_budget_mb=10)
    registry.register(spec("intent"), Echo)
    http = FakeHttp({"component_id": "intent"})
    router = RemoteComponentRouter(
        registry,
        "https://router.example/v1/route",
        token="secret",
        http=http,
    )
    decision = router.decide(
        Layer.VOICE,
        "voice.ground",
        {"device": "pi4", "routing_text": "Put my cup beside the plate"},
    )
    assert decision.component_id == "intent" and decision.router == "remote"
    assert http.calls[0][1]["allowed_components"] == ["intent"]
    assert http.calls[0][1]["context"]["routing_text"] == "Put my cup beside the plate"
    assert http.calls[0][2] == {"Authorization": "Bearer secret"}

    unavailable = RemoteComponentRouter(
        registry,
        "https://router.example/v1/route",
        http=FakeHttp(error=ConnectionError("offline")),
    )
    with pytest.raises(ConnectionError, match="offline"):
        unavailable.decide(Layer.VOICE, "voice.ground")

    invalid = RemoteComponentRouter(
        registry,
        "https://router.example/v1/route",
        http=FakeHttp({"component_id": "not-allowed"}),
    )
    with pytest.raises(LookupError, match="unknown component"):
        invalid.decide(Layer.VOICE, "voice.ground")


def test_remote_router_semantically_selects_across_compatible_capabilities():
    registry = ComponentRegistry(ram_budget_mb=10)
    registry.register(
        ComponentSpec("single", Layer.MANIPULATION, ("manipulation.single",), "single"),
        Echo,
    )
    registry.register(
        ComponentSpec("dual", Layer.MANIPULATION, ("manipulation.dual",), "dual"),
        Echo,
    )
    http = FakeHttp({"component_id": "dual"})
    router = RemoteComponentRouter(registry, "https://router.example/v1/route", http=http)

    decision = router.select_compatible(
        Layer.MANIPULATION,
        ("manipulation.single", "manipulation.dual"),
        "Use the left and right arms together to carry the tray",
        {"plan_id": "candidate-1"},
    )

    assert decision.component_id == "dual"
    assert decision.capability == "manipulation.dual"
    assert decision.router == "remote_semantic"
    payload = http.calls[0][1]
    assert payload["allowed_components"] == ["single", "dual"]
    assert payload["context"]["routing_text"].startswith("Use the left and right")


def test_semantic_component_selection_rejects_ids_outside_compatible_pool():
    registry = ComponentRegistry(ram_budget_mb=10)
    registry.register(
        ComponentSpec("single", Layer.MANIPULATION, ("manipulation.single",), "single"),
        Echo,
    )
    router = RemoteComponentRouter(
        registry,
        "https://router.example/v1/route",
        http=FakeHttp({"component_id": "unapproved"}),
    )

    with pytest.raises(LookupError, match="outside the compatible pool"):
        router.select_compatible(
            Layer.MANIPULATION,
            ("manipulation.single",),
            "pick up the object",
        )


def test_baseten_endpoint_uses_production_environment_and_api_key():
    http = FakeHttp({"result": "ok"})
    endpoint = BasetenEndpoint("model_123", api_key="key", http=http)
    assert endpoint.predict({"input": "task"}) == {"result": "ok"}
    url, payload, headers = http.calls[0]
    assert url == "https://model-model_123.api.baseten.co/environments/production/predict"
    assert payload == {"input": "task"}
    assert headers == {"Authorization": "Api-Key key"}


def test_baseten_endpoint_loads_ignored_local_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    (tmp_path / ".env").write_text("BASETEN_API_KEY=local-secret\n", encoding="utf-8")
    http = FakeHttp({"result": "ok"})

    endpoint = BasetenEndpoint("model_123", http=http)
    endpoint.predict({"input": "task"})

    assert http.calls[0][2] == {"Authorization": "Api-Key local-secret"}


def test_process_baseten_key_takes_precedence_over_local_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BASETEN_API_KEY", "process-secret")
    (tmp_path / ".env").write_text("BASETEN_API_KEY=local-secret\n", encoding="utf-8")
    http = FakeHttp({"result": "ok"})

    endpoint = BasetenEndpoint("model_123", http=http)
    endpoint.predict({"input": "task"})

    assert http.calls[0][2] == {"Authorization": "Api-Key process-secret"}


def test_baseten_component_decodes_scene_grounded_voice_nlp_contract():
    http = FakeHttp(
        {
            "intent": {
                "transcript": "bring the mug",
                "action": "bring",
                "target_object_ids": ["mug-1"],
                "constraints": ["left arm is broken"],
            }
        }
    )
    endpoint = BasetenEndpoint("voice_nlp", api_key="key", http=http)
    component = BasetenComponent(endpoint, decode=decode_physical_response)
    request = VoiceGroundingInput(
        "bring the mug",
        WorldState((), {}),
        PersonalContext("sam", {}, ("left arm is broken",), (), {}),
        (),
    )
    result = component.run(request)
    assert result.action == "bring"
    assert result.target_object_ids == ("mug-1",)


def test_baseten_chain_router_uses_api_key_and_never_substitutes_locally():
    registry = ComponentRegistry(ram_budget_mb=10)
    registry.register(spec("voice-nlp"), Echo)
    http = FakeHttp(error=ConnectionError("chain unavailable"))
    router = RemoteComponentRouter.from_baseten_chain(
        registry,
        "router_123",
        api_key="key",
        http=http,
    )
    with pytest.raises(ConnectionError, match="unavailable"):
        router.decide(Layer.VOICE, "voice.ground")
    assert http.calls[0][0].endswith("/environments/production/run_remote")
    assert http.calls[0][2] == {"Authorization": "Api-Key key"}


def test_pi_manifest_is_complete_and_profile_is_bounded():
    manifest = "examples/pi4_components.json"
    data = json.loads(open(manifest, encoding="utf-8").read())
    factories = {entry["id"]: Echo for entry in data["components"]}
    registry = ComponentRegistry.from_json(manifest, factories, ram_budget_mb=256)
    assert len(registry.specs) == 31
    assert registry.decide(Layer.CONTROL, "control.single_arm").runtime == "hardware"
    assert registry.decide(Layer.CONTROL, "control.bimanual").runtime == "hardware"
    assert registry.decide(Layer.KINEMATICS, "kinematics.inverse").runtime == "local"
    assert registry.decide(Layer.WORLD, "world.grasp_contact").runtime == "remote"
    tactile_ids = {
        registry.decide(Layer.TACTILE, capability).component_id
        for capability in (
            "tactile.contact",
            "tactile.slip",
            "tactile.force",
            "tactile.grasp_stability",
        )
    }
    assert tactile_ids == {"local-tactile-signal"}
    one_gb = PiRuntimeProfile.for_pi4(1024)
    assert one_gb.component_ram_budget_mb == 358
    assert one_gb.simulation_workers == 2
    with pytest.raises(ValueError, match="capped"):
        PiRuntimeProfile(256, simulation_workers=3)


def test_baseten_router_uses_general_metadata_instead_of_a_capability_route_table():
    expected = json.loads(
        Path("examples/physical_ai_specialists.json").read_text(encoding="utf-8")
    )
    deployed = json.loads(
        Path("deploy/baseten_router/specialists.json").read_text(encoding="utf-8")
    )
    assert deployed == expected

    module = ast.parse(Path("deploy/baseten_router/router.py").read_text(encoding="utf-8"))
    assigned_names = {
        target.id
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "ROUTES" not in assigned_names
    assert "MODEL_ID" in assigned_names
