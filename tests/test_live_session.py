import json
import sys
from pathlib import Path
from time import time
from types import ModuleType

import pytest

from moira.components import ComponentRegistry
from moira.edge_components import AlsaCommandRecorder, OpenCVCameraSource
from moira.physical import (
    CameraFrame,
    OutcomeInput,
    PhysicalAI,
    RobotState,
    TaskRequest,
)
from moira.production import load_physical_runtime_config
from moira.session import JsonlRunJournal


def test_usb_camera_captures_bounded_jpeg_and_releases(monkeypatch):
    class Encoded:
        def tobytes(self):
            return b"jpeg"

    class Capture:
        def __init__(self, source):
            self.source = source
            self.open = True
            self.reads = 0

        def isOpened(self):
            return self.open

        def set(self, key, value):
            return True

        def read(self):
            self.reads += 1
            return True, object()

        def release(self):
            self.open = False

    fake = ModuleType("cv2")
    created = []

    def video_capture(source):
        value = Capture(source)
        created.append(value)
        return value

    fake.VideoCapture = video_capture
    fake.CAP_PROP_FRAME_WIDTH = 1
    fake.CAP_PROP_FRAME_HEIGHT = 2
    fake.IMWRITE_JPEG_QUALITY = 3
    fake.imencode = lambda extension, image, options: (True, Encoded())
    monkeypatch.setitem(sys.modules, "cv2", fake)

    source = OpenCVCameraSource(0, warmup_frames=2)
    frame = source.capture()
    assert frame.data == b"jpeg"
    assert frame.media_type == "image/jpeg"
    assert created[0].reads == 3
    source.close()
    assert not created[0].open


def test_alsa_recorder_emits_wav_without_a_fallback(monkeypatch):
    calls = []

    class Result:
        stdout = b"R" * 45
        stderr = b""

    def run(command, **options):
        calls.append((command, options))
        return Result()

    monkeypatch.setattr("moira.edge_components.subprocess.run", run)
    audio = AlsaCommandRecorder(device="plughw:2,0").record(4)
    assert audio == b"R" * 45
    command, options = calls[0]
    assert command[:4] == ["arecord", "--quiet", "--device", "plughw:2,0"]
    assert command[-2:] == ["wav", "-"]
    assert options["check"] is True
    assert options["timeout"] == 6.0


def test_required_camera_verification_blocks_before_component_invocation():
    system = PhysicalAI(
        ComponentRegistry(ram_budget_mb=32),
        robot_model_id="test-robot",
        gripper_geometry={"max_width_m": 0.05},
        available_arms=("left",),
        require_camera_verification=True,
    )
    request = TaskRequest(
        "user",
        (CameraFrame("camera", b"jpeg", time()),),
        instruction="move the block",
        robot_state=RobotState({"left": (0.0, 0.0, 0.0)}, {"left": 0.05}, time()),
        execute=True,
    )
    with pytest.raises(RuntimeError, match="post-action camera"):
        system.run(request)


def test_unexecuted_outcome_rejects_post_action_world():
    from moira.physical import (
        CandidatePlan,
        ControlReport,
        FinalPlan,
        PlanStep,
        SimulationOutcome,
        WorldState,
    )

    candidate = CandidatePlan(
        "plan", (PlanStep("step", "hold", ("left",), 0.1),), "no-op"
    )
    plan = FinalPlan(candidate, SimulationOutcome("plan", 1.0, True, 2.5), "no-op")
    world = WorldState((), {})
    with pytest.raises(ValueError, match="unexecuted"):
        OutcomeInput(plan, ControlReport("plan", False, True, ()), world, world)


def test_failure_journal_excludes_camera_and_audio_payloads(tmp_path):
    journal = JsonlRunJournal(tmp_path / "runs.jsonl")
    request = TaskRequest(
        "user",
        (CameraFrame("camera", b"secret-image-bytes", time()),),
        audio=b"secret-audio-bytes",
        speak=False,
    )
    journal.append_failure(
        run_id="run-1",
        started_at=time(),
        request=request,
        error=RuntimeError("endpoint missing"),
        robot_model_id="robot",
    )
    raw = (tmp_path / "runs.jsonl").read_text(encoding="utf-8")
    record = json.loads(raw)
    assert "secret-image-bytes" not in raw
    assert "secret-audio-bytes" not in raw
    assert record["request"]["audio_bytes"] == len(b"secret-audio-bytes")
    assert record["request"]["cameras"][0]["payload_bytes"] == len(
        b"secret-image-bytes"
    )
    assert record["error"]["message"] == "endpoint missing"


def test_runtime_config_covers_every_remote_component():
    config = load_physical_runtime_config(Path("config/pi4_runtime.json"))
    manifest = json.loads(config.component_manifest.read_text(encoding="utf-8"))
    expected = {
        value["id"] for value in manifest["components"] if value["runtime"] == "remote"
    }
    assert set(config.remote_components) == expected
