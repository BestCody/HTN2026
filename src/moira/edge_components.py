"""Small local components and hardware adapters suitable for Raspberry Pi 4B."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .robot_config import RobotModel

from .physical import (
    ArmTelemetry,
    CameraFrame,
    CandidatePlan,
    CandidatePlanningInput,
    DetectedObject,
    FeedbackInput,
    FeedbackReport,
    FinalPlan,
    GroundedIntent,
    MemoryQuery,
    MemoryRecord,
    PerceptionInput,
    PersonalContext,
    PlanSelectionInput,
    PlanStep,
    SpeechInput,
    SpeechSynthesisInput,
    VoiceGroundingInput,
    WorldState,
)


class SQLitePersonalMemory:
    """Persistent per-user context with one short SQLite transaction per call."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    plan_summary TEXT NOT NULL,
                    feedback_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_user_time
                    ON events(user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS object_masses (
                    user_id TEXT NOT NULL,
                    object_label TEXT NOT NULL,
                    mass_kg REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, object_label)
                );
                """
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def set_profile(
        self,
        user_id: str,
        *,
        preferences: dict[str, Any] | None = None,
        accommodations: tuple[str, ...] | None = None,
        workspace: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if preferences is not None and not isinstance(preferences, dict):
            raise TypeError("preferences must be a dictionary or null")
        if workspace is not None and not isinstance(workspace, dict):
            raise TypeError("workspace must be a dictionary or null")
        if accommodations is not None and (
            not isinstance(accommodations, tuple)
            or any(not isinstance(item, str) or not item.strip() for item in accommodations)
        ):
            raise ValueError("accommodations must contain non-empty strings")
        values = {
            "preferences": preferences,
            "accommodations": accommodations,
            "workspace": workspace,
        }
        with self._connection() as connection:
            for key, value in values.items():
                if value is None:
                    continue
                connection.execute(
                    """
                    INSERT INTO profiles(user_id, key, value_json) VALUES (?, ?, ?)
                    ON CONFLICT(user_id, key) DO UPDATE SET value_json=excluded.value_json
                    """,
                    (user_id, key, json.dumps(value)),
                )

    def run(self, request: MemoryQuery | MemoryRecord) -> PersonalContext:
        if isinstance(request, MemoryRecord):
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO events(
                        user_id, instruction, plan_summary, feedback_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        request.user_id,
                        request.instruction,
                        request.plan_summary,
                        json.dumps(
                            {
                                "learned_facts": request.feedback.learned_facts,
                                "next_time_adjustments": request.feedback.next_time_adjustments,
                            }
                        ),
                        time(),
                    ),
                )
                for label, mass in request.feedback.object_masses_kg.items():
                    connection.execute(
                        """
                        INSERT INTO object_masses(user_id, object_label, mass_kg, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(user_id, object_label) DO UPDATE SET
                            mass_kg=excluded.mass_kg,
                            updated_at=excluded.updated_at
                        """,
                        (request.user_id, label, mass, time()),
                    )
            request = MemoryQuery(request.user_id, request.instruction, {})
        if not isinstance(request, MemoryQuery):
            raise TypeError("Personal memory expects MemoryQuery or MemoryRecord")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT key, value_json FROM profiles WHERE user_id=?",
                (request.user_id,),
            ).fetchall()
            events = connection.execute(
                """
                SELECT instruction, plan_summary, feedback_json
                FROM events WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT 8
                """,
                (request.user_id,),
            ).fetchall()
            mass_rows = connection.execute(
                "SELECT object_label, mass_kg FROM object_masses WHERE user_id=?",
                (request.user_id,),
            ).fetchall()
        profile = {key: json.loads(value) for key, value in rows}
        stored_workspace = profile.get("workspace", {})
        workspace = {**stored_workspace, **dict(request.workspace)}
        comments = []
        for instruction, summary, feedback_json in events:
            comments.append(f"{instruction} — {summary}")
            feedback = json.loads(feedback_json)
            comments.extend(feedback.get("learned_facts", ()))
            comments.extend(feedback.get("next_time_adjustments", ()))
        return PersonalContext(
            request.user_id,
            profile.get("preferences", {}),
            tuple(profile.get("accommodations", ())),
            tuple(comments),
            workspace,
            dict(mass_rows),
        )


class StructuredFramePerception:
    """Development fixture that fuses object records already attached to frames."""

    def run(self, request: PerceptionInput) -> WorldState:
        if not isinstance(request, PerceptionInput):
            raise TypeError("Perception component expects PerceptionInput")
        objects: dict[str, DetectedObject] = {}
        hazards = []
        for frame in request.frames:
            if not isinstance(frame.data, dict):
                raise TypeError(
                    "StructuredFramePerception is a fixture; production frames need a vision model"
                )
            for value in frame.data.get("objects", []):
                item = value if isinstance(value, DetectedObject) else DetectedObject(**value)
                previous = objects.get(item.id)
                if previous is None or item.confidence > previous.confidence:
                    objects[item.id] = item
            hazards.extend(str(value) for value in frame.data.get("hazards", []))
        return WorldState(
            tuple(objects.values()),
            dict(request.workspace),
            tuple(dict.fromkeys(hazards)),
            time(),
            {},
            str(request.workspace.get("coordinate_frame", "robot_base")),
            str(request.workspace.get("up_axis", "z")),
        )


class SceneGroundedVoiceNLP:
    """Offline DUM-E-style fixture grounding speech against perceived objects."""

    def run(self, request: VoiceGroundingInput) -> GroundedIntent:
        if not isinstance(request, VoiceGroundingInput):
            raise TypeError("Voice NLP component expects VoiceGroundingInput")
        transcript = " ".join(request.transcript.strip().split())
        lowered = transcript.lower()
        reference_context = " ".join(
            [lowered]
            + [turn.text.lower() for turn in request.dialogue[-4:]]
            + [comment.lower() for comment in request.personal.recent_comments[:4]]
        )
        targets = tuple(
            item.id
            for item in request.world.objects
            if item.id.lower() in reference_context or item.label.lower() in reference_context
        )
        action = next(
            (
                verb
                for verb in ("bring", "pick", "place", "move", "hold", "wave", "nod")
                if verb in lowered
            ),
            "perform",
        )
        needs_object = action in ("bring", "pick", "place", "move", "hold")
        needs_clarification = needs_object and not targets
        question = "Which object do you mean?" if needs_clarification else None
        constraints = tuple(request.personal.accommodations)
        return GroundedIntent(
            transcript,
            action,
            targets,
            constraints,
            needs_clarification,
            question,
        )


class Utf8SpeechFixture:
    """Test-only STT fixture; production uses a Baseten or local STT deployment."""

    def run(self, request: SpeechInput) -> str:
        if not isinstance(request, SpeechInput):
            raise TypeError("Speech fixture expects SpeechInput")
        return request.audio.decode("utf-8")


class Utf8SpeechSynthesisFixture:
    """Test-only TTS fixture returning tagged bytes instead of waveform audio."""

    def run(self, request: SpeechSynthesisInput) -> bytes:
        if not isinstance(request, SpeechSynthesisInput):
            raise TypeError("Speech fixture expects SpeechSynthesisInput")
        return f"TTS:{request.text}".encode()


class LayeredRulePlanner:
    """Deterministic development planner demonstrating the typed layer contract."""

    def run(
        self, request: CandidatePlanningInput | PlanSelectionInput
    ) -> tuple[CandidatePlan, ...] | FinalPlan:
        if isinstance(request, CandidatePlanningInput):
            if not request.world.objects:
                raise ValueError("No perceived objects are available to plan against")
            targets = {item.id: item for item in request.world.objects}
            target = (
                targets[request.intent.target_object_ids[0]]
                if request.intent.target_object_ids
                else request.world.objects[0]
            )
            accessible_side = "accessible right side"
            preferences = request.personal.preferences
            if preferences.get("delivery_side") in ("left", "right"):
                accessible_side = f"accessible {preferences['delivery_side']} side"
            accommodations = " ".join(request.personal.accommodations).lower()
            if "right arm" in accommodations and "broken" in accommodations:
                accessible_side = "accessible left side"
            elif "left arm" in accommodations and "broken" in accommodations:
                accessible_side = "accessible right side"
            plans = []
            for name, arms in (("single", ("left",)), ("bimanual", ("left", "right"))):
                plans.append(
                    CandidatePlan(
                        name,
                        (
                            PlanStep(
                                f"{name}-grasp",
                                "grasp",
                                arms,
                                0.8,
                                target.id,
                                {
                                    "delivery": accessible_side,
                                    "target_position_m": target.position_m,
                                },
                            ),
                            PlanStep(
                                f"{name}-place",
                                "place",
                                arms,
                                1.0,
                                target.id,
                                {
                                    "delivery": accessible_side,
                                    "target_position_m": target.position_m,
                                },
                            ),
                        ),
                        f"Move {target.label} to the {accessible_side} using {name} handling",
                    )
                )
            return tuple(plans[: request.limit])
        if isinstance(request, PlanSelectionInput):
            outcomes = {outcome.plan_id: outcome for outcome in request.simulations}
            safe = [
                (outcomes[candidate.id], candidate)
                for candidate in request.candidates
                if outcomes[candidate.id].safe
            ]
            if not safe:
                raise RuntimeError("No simulated candidate is safe")
            outcome, candidate = max(safe, key=lambda item: item[0].score)
            return FinalPlan(candidate, outcome, candidate.rationale)
        raise TypeError("Planner expects candidate-generation or selection input")


class LoadFeedbackComponent:
    def __init__(self, *, single_arm_payload_kg: float = 0.8) -> None:
        if (
            not isinstance(single_arm_payload_kg, (int, float))
            or not math.isfinite(single_arm_payload_kg)
            or single_arm_payload_kg <= 0
        ):
            raise ValueError("single_arm_payload_kg must be finite and positive")
        self.single_arm_payload_kg = single_arm_payload_kg

    @classmethod
    def from_robot_model(cls, model: RobotModel) -> LoadFeedbackComponent:
        model.require_motion_ready()
        return cls(single_arm_payload_kg=model.payload_limit_kg)

    def run(self, request: FeedbackInput) -> FeedbackReport:
        if not isinstance(request, FeedbackInput):
            raise TypeError("Feedback component expects FeedbackInput")
        objects = {item.id: item for item in request.world.objects}
        facts = []
        adjustments = list(request.control.issues)
        masses = {}
        if request.load is None:
            for item in request.control.telemetry:
                if item.measured_mass_kg is None:
                    continue
                target_id = next(
                    (
                        step.target_object_id
                        for step in request.plan.candidate.steps
                        if step.id == item.step_id
                    ),
                    None,
                )
                label = objects[target_id].label if target_id in objects else target_id or "object"
                facts.append(f"{label} measured {item.measured_mass_kg:.3f} kg")
                masses[label] = item.measured_mass_kg
                if item.measured_mass_kg > self.single_arm_payload_kg:
                    adjustments.append(f"Use both arms for {label}")
        else:
            for label, mass in request.load.object_masses_kg.items():
                facts.append(f"{label} measured {mass:.3f} kg")
                masses[label] = mass
                if mass > self.single_arm_payload_kg:
                    adjustments.append(f"Use both arms for {label}")
        if request.failure is not None and request.failure.failure is not None:
            adjustments.append(f"Address classified failure: {request.failure.failure}")
            adjustments.extend(
                f"{key}={value}" for key, value in request.failure.recommended_adjustments.items()
            )
        if request.outcome is not None and request.outcome.status in ("failed", "uncertain"):
            adjustments.append(f"Verify task outcome: {request.outcome.status}")
        if request.prediction_error is not None:
            for model, error in request.prediction_error.per_model_absolute_error.items():
                if error > 0.35:
                    adjustments.append(
                        f"Recalibrate {model}; observed prediction error was {error:.3f}"
                    )
        if not request.control.success:
            adjustments.append("Re-simulate with a more conservative approach")
        return FeedbackReport(
            tuple(dict.fromkeys(facts)),
            tuple(dict.fromkeys(adjustments)),
            masses,
        )


class RpicamStillSource:
    """Capture one bounded JPEG using Raspberry Pi OS's rpicam-still command."""

    def __init__(
        self,
        camera_id: str = "csi0",
        *,
        width: int = 640,
        height: int = 480,
        timeout_seconds: float = 5,
    ) -> None:
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (width, height)
        ):
            raise ValueError("Camera width and height must be positive integers")
        if (
            not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Camera timeout must be finite and positive")
        self.camera_id = camera_id
        self.command = (
            "rpicam-still",
            "--nopreview",
            "--immediate",
            "--width",
            str(width),
            "--height",
            str(height),
            "--output",
            "-",
        )
        self.timeout_seconds = timeout_seconds

    def capture(self) -> CameraFrame:
        try:
            result = subprocess.run(
                self.command,
                check=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("rpicam-still is not installed on this Raspberry Pi") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace")[-1000:]
            raise RuntimeError(f"Camera capture failed: {detail}") from exc
        if not result.stdout:
            raise RuntimeError("Camera capture returned an empty image")
        return CameraFrame(self.camera_id, result.stdout, time())


class OpenCVCameraSource:
    """Capture bounded JPEG frames from a USB/UVC webcam such as the CO6 camera."""

    def __init__(
        self,
        source: int | str = 0,
        *,
        camera_id: str = "usb0",
        width: int = 640,
        height: int = 480,
        warmup_frames: int = 3,
        jpeg_quality: int = 85,
    ) -> None:
        if not isinstance(source, (int, str)) or isinstance(source, bool):
            raise TypeError("Camera source must be a device index or path")
        if isinstance(source, int) and source < 0:
            raise ValueError("Camera device index must be nonnegative")
        if isinstance(source, str) and not source.strip():
            raise ValueError("Camera device path must be non-empty")
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (width, height)
        ):
            raise ValueError("Camera width and height must be positive integers")
        if (
            not isinstance(warmup_frames, int)
            or isinstance(warmup_frames, bool)
            or warmup_frames < 0
        ):
            raise ValueError("warmup_frames must be a nonnegative integer")
        if (
            not isinstance(jpeg_quality, int)
            or isinstance(jpeg_quality, bool)
            or not 1 <= jpeg_quality <= 100
        ):
            raise ValueError("jpeg_quality must be an integer in [1, 100]")
        self.source = source
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self.jpeg_quality = jpeg_quality
        self._capture: Any | None = None
        self._lock = RLock()

    def _open(self, cv2: Any) -> Any:
        if self._capture is not None and self._capture.isOpened():
            return self._capture
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open USB camera source: {self.source}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture
        return capture

    def capture(self) -> CameraFrame:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Install USB camera support with 'pip install .[camera]'"
            ) from exc
        with self._lock:
            capture = self._open(cv2)
            image = None
            for _ in range(self.warmup_frames + 1):
                ok, image = capture.read()
                if not ok or image is None:
                    self.close()
                    raise RuntimeError(f"USB camera {self.source} did not return a frame")
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok or encoded is None:
                raise RuntimeError("OpenCV could not encode the camera frame as JPEG")
            return CameraFrame(self.camera_id, encoded.tobytes(), time())

    def close(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None


class AlsaCommandRecorder:
    """Record a fixed-duration WAV command from an ALSA microphone on Raspberry Pi."""

    def __init__(
        self,
        *,
        device: str | None = None,
        sample_rate_hz: int = 16_000,
        channels: int = 1,
        timeout_margin_seconds: float = 2.0,
    ) -> None:
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("ALSA device must be a non-empty string or null")
        if (
            not isinstance(sample_rate_hz, int)
            or isinstance(sample_rate_hz, bool)
            or not 8_000 <= sample_rate_hz <= 48_000
        ):
            raise ValueError("Audio sample rate must be an integer in [8000, 48000]")
        if channels not in (1, 2):
            raise ValueError("Audio channels must be 1 or 2")
        if (
            not isinstance(timeout_margin_seconds, (int, float))
            or isinstance(timeout_margin_seconds, bool)
            or not math.isfinite(timeout_margin_seconds)
            or timeout_margin_seconds <= 0
        ):
            raise ValueError("Audio timeout margin must be finite and positive")
        self.device = device
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.timeout_margin_seconds = float(timeout_margin_seconds)

    def record(self, duration_seconds: int) -> bytes:
        if (
            not isinstance(duration_seconds, int)
            or isinstance(duration_seconds, bool)
            or not 1 <= duration_seconds <= 30
        ):
            raise ValueError("Command recording duration must be an integer in [1, 30]")
        command = ["arecord", "--quiet"]
        if self.device is not None:
            command.extend(("--device", self.device))
        command.extend(
            (
                "--duration",
                str(duration_seconds),
                "--format",
                "S16_LE",
                "--rate",
                str(self.sample_rate_hz),
                "--channels",
                str(self.channels),
                "--file-type",
                "wav",
                "-",
            )
        )
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=duration_seconds + self.timeout_margin_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("arecord is not installed on this Raspberry Pi") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Microphone recording exceeded its time limit") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace")[-1000:]
            raise RuntimeError(f"Microphone recording failed: {detail}") from exc
        if len(result.stdout) <= 44:
            raise RuntimeError("Microphone recording returned an empty WAV file")
        return result.stdout


class SimulatedArmDriver:
    """Deterministic driver used by the edge E2E test and physical demo."""

    def __init__(self, arm: str, *, measured_mass_kg: float = 1.0) -> None:
        if arm not in ("left", "right"):
            raise ValueError("Simulated arm must be left or right")
        if (
            not isinstance(measured_mass_kg, (int, float))
            or not math.isfinite(measured_mass_kg)
            or measured_mass_kg <= 0
        ):
            raise ValueError("Simulated measured mass must be finite and positive")
        self.arm = arm
        self.measured_mass_kg = measured_mass_kg
        self.steps: list[str] = []
        self.stopped = False

    def execute(self, step: PlanStep, world: WorldState) -> ArmTelemetry:
        self.steps.append(step.id)
        return ArmTelemetry(
            self.arm,
            step.id,
            True,
            self.measured_mass_kg,
            normal_force_n=self.measured_mass_kg * 9.81,
            slip_probability=0.03,
        )

    def execute_chunk(self, chunk, trajectory, world: WorldState) -> ArmTelemetry:
        del trajectory, world
        if self.arm not in chunk.arms:
            raise ValueError(f"Action chunk {chunk.id} does not command the {self.arm} arm")
        self.steps.append(chunk.step_id)
        return ArmTelemetry(
            self.arm,
            chunk.step_id,
            True,
            self.measured_mass_kg,
            normal_force_n=self.measured_mass_kg * 9.81,
            slip_probability=0.03,
        )

    def stop(self) -> None:
        self.stopped = True
