import json
import sys
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from time import time
from types import ModuleType

import pytest
from PIL import Image

from moira.components import ComponentRegistry
from moira.edge_components import (
    AlsaCommandRecorder,
    FfmpegDshowCommandRecorder,
    LanCameraSource,
    OpenCVCameraSource,
    V4L2JpegSource,
)
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


def test_lan_camera_decodes_authenticated_jpeg_contract(monkeypatch):
    payload = json.dumps(
        {
            "camera_id": "co6-usb",
            "captured_at": 123.5,
            "media_type": "image/jpeg",
            "data_base64": b64encode(b"\xff\xd8\xffjpeg").decode("ascii"),
        }
    ).encode()
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            captured["limit"] = limit
            return payload

    def urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("moira.edge_components.urllib.request.urlopen", urlopen)
    frame = LanCameraSource(
        "http://robot-brain.local:8765/v1/camera",
        token="secret",
        camera_id="co6-usb",
    ).capture()
    assert frame.data == b"\xff\xd8\xffjpeg"
    assert frame.captured_at == 123.5
    assert captured["authorization"] == "Bearer secret"
    assert captured["timeout"] == 5.0
    assert captured["limit"] == 4 * 1024 * 1024 + 1


def test_lan_camera_rejects_wrong_camera_identity(monkeypatch):
    payload = json.dumps(
        {
            "camera_id": "wrong-camera",
            "captured_at": 123.5,
            "media_type": "image/jpeg",
            "data_base64": b64encode(b"\xff\xd8\xffjpeg").decode("ascii"),
        }
    ).encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            return payload

    monkeypatch.setattr(
        "moira.edge_components.urllib.request.urlopen", lambda request, timeout: Response()
    )
    source = LanCameraSource(
        "http://robot-brain.local:8765/v1/camera",
        token="secret",
        camera_id="co6-usb",
    )
    with pytest.raises(RuntimeError, match="identity"):
        source.capture()


def test_lan_camera_applies_configured_rotation(monkeypatch):
    original = Image.new("RGB", (40, 20))
    for x in range(20):
        for y in range(10):
            original.putpixel((x, y), (255, 0, 0))
    image_bytes = BytesIO()
    original.save(image_bytes, format="JPEG", quality=100, subsampling=0)
    payload = json.dumps(
        {
            "camera_id": "co6-usb",
            "captured_at": 123.5,
            "media_type": "image/jpeg",
            "data_base64": b64encode(image_bytes.getvalue()).decode("ascii"),
        }
    ).encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            return payload

    monkeypatch.setattr(
        "moira.edge_components.urllib.request.urlopen", lambda request, timeout: Response()
    )
    frame = LanCameraSource(
        "http://robot-brain.local:8765/v1/camera",
        token="secret",
        camera_id="co6-usb",
        rotation_degrees=180,
    ).capture()

    with Image.open(BytesIO(frame.data)) as rotated:
        assert rotated.size == (40, 20)
        red = rotated.convert("RGB").getpixel((30, 15))
        assert red[0] > 180
        assert red[0] > red[1] * 2
        assert red[0] > red[2] * 2


def test_v4l2_camera_captures_native_mjpeg_without_opencv(monkeypatch):
    calls = []

    class Result:
        stdout = b"\xff\xd8\xffcamera-data\xff\xd9"
        stderr = b"<"

    def run(command, **options):
        calls.append((command, options))
        return Result()

    monkeypatch.setattr("moira.edge_components.subprocess.run", run)
    source = V4L2JpegSource(
        "/dev/v4l/by-id/co6-video-index0",
        width=640,
        height=480,
        fps=30,
    )
    frame = source.capture()

    assert frame.camera_id == "co6-usb"
    assert frame.data == Result.stdout
    command, options = calls[0]
    assert command[0] == "v4l2-ctl"
    assert "--silent" in command
    assert "--device=/dev/v4l/by-id/co6-video-index0" in command
    assert "--set-fmt-video=width=640,height=480,pixelformat=MJPG" in command
    assert options == {"check": True, "capture_output": True, "timeout": 5.0}


def test_v4l2_camera_rejects_non_jpeg_frame(monkeypatch):
    class Result:
        stdout = b"not-a-jpeg"
        stderr = b""

    monkeypatch.setattr(
        "moira.edge_components.subprocess.run", lambda command, **options: Result()
    )
    with pytest.raises(RuntimeError, match="JPEG frame"):
        V4L2JpegSource("/dev/video0").capture()


def test_v4l2_camera_adds_missing_uvc_jpeg_end_marker(monkeypatch):
    class Result:
        stdout = b"\xff\xd8\xffcamera-without-eoi"
        stderr = b""

    monkeypatch.setattr(
        "moira.edge_components.subprocess.run", lambda command, **options: Result()
    )
    assert V4L2JpegSource("/dev/video0").capture().data.endswith(b"\xff\xd9")


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


def test_windows_recorder_emits_pcm_wav_without_a_fallback(monkeypatch):
    calls = []

    class Result:
        stdout = b"\x01\x00" * 64
        stderr = b""

    def run(command, **options):
        calls.append((command, options))
        return Result()

    monkeypatch.setattr("moira.edge_components.subprocess.run", run)
    audio = FfmpegDshowCommandRecorder("Laptop Microphone").record(5)

    assert audio.startswith(b"RIFF") and audio[8:12] == b"WAVE"
    command, options = calls[0]
    assert command[:6] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "dshow"]
    assert "audio=Laptop Microphone" in command
    assert command[-3:] == ["-f", "s16le", "pipe:1"]
    assert options == {"check": True, "capture_output": True, "timeout": 8.0}


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
        execution_confirmed=True,
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
