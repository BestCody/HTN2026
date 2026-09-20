"""Live camera session and append-only evidence journal for physical runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from time import time
from typing import Any
from uuid import uuid4

from .physical import (
    CameraSource,
    CandidatePlan,
    ClarificationResult,
    DialogueTurn,
    EmergencyStopResult,
    GroundedIntent,
    PhysicalAI,
    PhysicalAIResult,
    RobotState,
    TactileSample,
    TaskRequest,
    capture_frames,
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Cannot journal {type(value).__name__}")


class JsonlRunJournal:
    """Persist one complete, reviewable record for each attempted task."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _request_record(request: TaskRequest) -> dict[str, Any]:
        return {
            "user_id": request.user_id,
            "input_kind": "audio" if request.audio is not None else "text",
            "instruction": request.instruction,
            "audio_bytes": len(request.audio) if request.audio is not None else None,
            "execute": request.execute,
            "execution_confirmed": request.execution_confirmed,
            "confirmed_plan_id": (
                request.confirmed_candidate.id
                if request.confirmed_candidate is not None
                else None
            ),
            "speak": request.speak,
            "workspace": _json_value(dict(request.workspace or {})),
            "cameras": [
                {
                    "camera_id": frame.camera_id,
                    "captured_at": frame.captured_at,
                    "media_type": frame.media_type,
                    "payload_bytes": (
                        len(frame.data) if isinstance(frame.data, bytes) else None
                    ),
                }
                for frame in request.frames
            ],
            "robot_state": _json_value(request.robot_state),
            "tactile_sample_count": len(request.tactile_samples),
        }

    def _write(self, record: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            _json_value(record),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as destination:
                destination.write(encoded + "\n")
                destination.flush()

    def append_result(
        self,
        *,
        run_id: str,
        started_at: float,
        request: TaskRequest,
        result: PhysicalAIResult | ClarificationResult | EmergencyStopResult,
        robot_model_id: str,
    ) -> None:
        completed_at = time()
        common = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_seconds": completed_at - started_at,
            "robot_model_id": robot_model_id,
            "request": self._request_record(request),
        }
        if isinstance(result, EmergencyStopResult):
            common["status"] = "emergency_stop"
            common["result"] = {
                "transcript": result.transcript,
                "response_text": result.response_text,
                "response_audio_bytes": (
                    len(result.response_audio) if result.response_audio is not None else None
                ),
                "stop_issues": list(result.stop_issues),
                "routing": _json_value(result.routing),
            }
        elif isinstance(result, ClarificationResult):
            common["status"] = "clarification"
            common["result"] = {
                "transcript": result.transcript,
                "question": result.question,
                "world_before": _json_value(result.world),
                "personal": _json_value(result.personal),
                "routing": _json_value(result.routing),
            }
        else:
            common["status"] = result.outcome.status if result.outcome is not None else "unknown"
            common["result"] = {
                "transcript": result.intent.transcript,
                "intent": _json_value(result.intent),
                "personal": _json_value(result.personal),
                "world_before": _json_value(result.world),
                "world_pre_execute": _json_value(result.world_pre_execute),
                "world_after": _json_value(result.world_after),
                "candidates": _json_value(result.candidates),
                "simulations": _json_value(result.simulations),
                "selected_plan": _json_value(result.plan),
                "routing": _json_value(result.routing),
                "world_predictions": _json_value(result.world_predictions),
                "control": _json_value(result.control),
                "outcome": _json_value(result.outcome),
                "prediction_error": _json_value(result.prediction_error),
                "feedback": _json_value(result.feedback),
                "response_text": result.response_text,
                "response_audio_bytes": (
                    len(result.response_audio) if result.response_audio is not None else None
                ),
            }
        self._write(common)

    def append_failure(
        self,
        *,
        run_id: str,
        started_at: float,
        request: TaskRequest,
        error: Exception,
        robot_model_id: str,
    ) -> None:
        completed_at = time()
        self._write(
            {
                "schema_version": 1,
                "run_id": run_id,
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_seconds": completed_at - started_at,
                "robot_model_id": robot_model_id,
                "status": "error",
                "request": self._request_record(request),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )


class PhysicalSession:
    """Capture, run, verify, and journal one physical-AI task."""

    def __init__(
        self,
        system: PhysicalAI,
        cameras: Sequence[CameraSource],
        journal: JsonlRunJournal,
    ) -> None:
        if not isinstance(system, PhysicalAI):
            raise TypeError("system must be a PhysicalAI instance")
        if not cameras or any(not callable(getattr(camera, "capture", None)) for camera in cameras):
            raise ValueError("At least one camera source with capture() is required")
        if not isinstance(journal, JsonlRunJournal):
            raise TypeError("journal must be a JsonlRunJournal")
        self.system = system
        self.cameras = tuple(cameras)
        self.journal = journal

    def run(
        self,
        *,
        user_id: str,
        instruction: str | None = None,
        audio: bytes | None = None,
        workspace: Mapping[str, Any] | None = None,
        dialogue: tuple[DialogueTurn, ...] = (),
        tactile_samples: tuple[TactileSample, ...] = (),
        friction_coefficient: float | None = None,
        robot_state: RobotState | None = None,
        execute: bool = False,
        execution_confirmed: bool = False,
        speak: bool = True,
        confirmed_intent: GroundedIntent | None = None,
        confirmed_candidate: CandidatePlan | None = None,
    ) -> PhysicalAIResult | ClarificationResult | EmergencyStopResult:
        started_at = time()
        if not math.isfinite(started_at):
            raise RuntimeError("System clock did not return a finite timestamp")
        frames = capture_frames(self.cameras)
        request = TaskRequest(
            user_id,
            frames,
            instruction=instruction,
            audio=audio,
            workspace=workspace,
            dialogue=dialogue,
            tactile_samples=tactile_samples,
            friction_coefficient=friction_coefficient,
            robot_state=robot_state,
            execute=execute,
            execution_confirmed=execution_confirmed,
            speak=speak,
            confirmed_intent=confirmed_intent,
            confirmed_candidate=confirmed_candidate,
        )
        run_id = uuid4().hex
        try:
            result = self.system.run(
                request,
                pre_action_capture=(
                    (lambda: capture_frames(self.cameras)) if execute else None
                ),
                post_action_capture=(
                    (lambda: capture_frames(self.cameras)) if execute else None
                ),
            )
        except Exception as exc:
            self.journal.append_failure(
                run_id=run_id,
                started_at=started_at,
                request=request,
                error=exc,
                robot_model_id=self.system.robot_model_id,
            )
            raise
        self.journal.append_result(
            run_id=run_id,
            started_at=started_at,
            request=request,
            result=result,
            robot_model_id=self.system.robot_model_id,
        )
        return result

    def close(self) -> None:
        failures: list[Exception] = []
        try:
            self.system.registry.unload_all()
        except Exception as exc:
            failures.append(exc)
        for camera in self.cameras:
            close = getattr(camera, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise RuntimeError(
                "One or more physical-session resources failed to close"
            ) from failures[0]

    def __enter__(self) -> PhysicalSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
