from dataclasses import replace

import pytest

from moira import AdapterServer, Expert, ExpertRegistry, InMemoryServer, MoIRA, RoutingDecision
from moira.demo import run_demo


class Backend:
    def __init__(self):
        self.events = []
        self.active = None
        self.fail_load = False

    def load_adapter(self, expert):
        self.events.append(("load", expert.id))
        if self.fail_load:
            raise OSError("missing adapter")

    def unload_adapter(self, expert_id):
        self.events.append(("unload", expert_id))

    def activate_adapter(self, expert_id):
        self.active = expert_id
        self.events.append(("activate", expert_id))

    def reset(self, instruction):
        self.events.append(("reset", instruction))
        self.instruction = instruction

    def act(self, observation):
        return self.active, self.instruction, observation


class Router:
    def __init__(self):
        self.calls = 0
        self.expert_id = "a"

    def route(self, instruction):
        self.calls += 1
        return RoutingDecision(self.expert_id, "test")


@pytest.fixture
def experts():
    return [Expert("a", "s", "a", "a"), Expert("b", "s", "a", "b")]


def test_route_once_per_episode_and_pass_original_instruction(experts):
    router, backend = Router(), Backend()
    controller = MoIRA(ExpertRegistry(experts), router, AdapterServer(backend))
    with controller.episode("unchanged instruction") as episode:
        for observation in range(4):
            assert episode.act(observation) == ("a", "unchanged instruction", observation)
        assert router.calls == 1
        with pytest.raises(RuntimeError, match="active episode"):
            with controller.episode("overlapping"):
                pass
    with pytest.raises(RuntimeError, match="closed"):
        episode.act(0)
    router.expert_id = "b"
    with controller.episode("next") as next_episode:
        assert next_episode.act(0)[0] == "b"


@pytest.mark.parametrize("mode, loads, unloads", [("disk", 3, 2), ("multi", 2, 0)])
def test_serving_modes_reuse_or_evict(experts, mode, loads, unloads):
    backend = Backend()
    server = AdapterServer(backend, mode=mode)
    for expert in [experts[0], experts[0], experts[1], experts[0]]:
        with server.session(expert):
            pass
    assert sum(event == "load" for event, _ in backend.events) == loads
    assert sum(event == "unload" for event, _ in backend.events) == unloads
    assert len(server.loaded_ids) == (1 if mode == "disk" else 2)
    server.close()
    assert not server.loaded_ids


def test_multi_preload_and_replaced_checkpoint(experts):
    backend = Backend()
    server = AdapterServer(backend, mode="multi")
    server.preload(tuple(experts))
    backend.events.clear()
    with server.session(experts[0]):
        with pytest.raises(RuntimeError):
            server.close()
        with pytest.raises(RuntimeError):
            server.preload(tuple(experts))
    assert backend.events == [("activate", "a")]
    with server.session(replace(experts[0], adapter_path="new")):
        pass
    assert backend.events[-3:] == [("unload", "a"), ("load", "a"), ("activate", "a")]


def test_load_and_episode_errors_release_lease(experts):
    backend = Backend()
    server = AdapterServer(backend)
    backend.fail_load = True
    with pytest.raises(OSError):
        with server.session(experts[0]):
            pass
    assert not server.loaded_ids
    backend.fail_load = False
    with pytest.raises(ValueError):
        with server.session(experts[0]):
            raise ValueError("environment failure")
    with server.session(experts[1]):
        assert backend.active == "b"


def test_resident_missing_policy_releases_lock(experts):
    server = InMemoryServer({"a": Backend()})
    with pytest.raises(KeyError):
        with server.session(experts[1]):
            pass
    with server.session(experts[0]):
        pass


@pytest.mark.parametrize("adapter", [False, True])
def test_policy_reference_expires_with_its_session(experts, adapter):
    backend = Backend()
    server = AdapterServer(backend) if adapter else InMemoryServer({"a": backend})
    with server.session(experts[0]) as policy:
        policy.reset("task")
        assert policy.act(1)[-1] == 1
    with pytest.raises(RuntimeError, match="closed"):
        policy.act(2)


def test_independent_resident_policies_can_overlap_but_shared_instances_cannot(experts):
    first, second = Backend(), Backend()
    server = InMemoryServer({"a": first, "b": second})
    with server.session(experts[0]):
        with server.session(experts[1]):
            pass
        with pytest.raises(RuntimeError, match="active episode"):
            with server.session(experts[0]):
                pass

    aliased = InMemoryServer({"a": first, "b": first})
    with aliased.session(experts[0]):
        with pytest.raises(RuntimeError, match="active episode"):
            with aliased.session(experts[1]):
                pass


def test_adapter_preload_rolls_back_and_close_attempts_every_unload(experts):
    class FailingBackend(Backend):
        def __init__(self):
            super().__init__()
            self.fail_load_ids = set()
            self.fail_unload_ids = set()

        def load_adapter(self, expert):
            super().load_adapter(expert)
            if expert.id in self.fail_load_ids:
                raise OSError(f"cannot load {expert.id}")

        def unload_adapter(self, expert_id):
            super().unload_adapter(expert_id)
            if expert_id in self.fail_unload_ids:
                raise OSError(f"cannot unload {expert_id}")

    backend = FailingBackend()
    backend.fail_load_ids.add("b")
    server = AdapterServer(backend, mode="multi")
    with pytest.raises(OSError, match="cannot load b"):
        server.preload(tuple(experts))
    assert server.loaded_ids == ()
    assert backend.events[-1] == ("unload", "a")

    backend.fail_load_ids.clear()
    server.preload(tuple(experts))
    backend.fail_unload_ids.add("a")
    with pytest.raises(RuntimeError, match="a"):
        server.close()
    assert server.loaded_ids == ("a",)
    assert ("unload", "b") in backend.events


def test_preload_rejects_checkpoint_replacement_before_mutation(experts):
    backend = Backend()
    server = AdapterServer(backend, mode="multi")
    server.preload((experts[0],))
    backend.events.clear()
    with pytest.raises(ValueError, match="replace loaded"):
        server.preload((replace(experts[0], adapter_path="new"), experts[1]))
    assert backend.events == []
    assert server.loaded_ids == ("a",)


def test_controller_rejects_router_bound_to_another_registry(experts):
    registry = ExpertRegistry(experts)
    router = Router()
    router.registry = ExpertRegistry(experts)
    with pytest.raises(ValueError, match="same ExpertRegistry"):
        MoIRA(registry, router, InMemoryServer({"a": Backend(), "b": Backend()}))


def test_demo_runs_to_success():
    result = run_demo()
    assert result.expert_id == "right"
    assert result.success
    assert result.actions == (1, 1, 1)
