"""Confirmation, correction, clarification, and cancellation around physical tasks."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Event, Lock, Thread
from typing import Any

from .physical import (
    ClarificationResult,
    ConfirmedPlanChangedError,
    DialogueTurn,
    EmergencyStopResult,
    PhysicalAIResult,
    RobotState,
    TactileSample,
    is_emergency_stop_command,
)
from .session import PhysicalSession


def _normalized(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Interaction response must be non-empty text")
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


@dataclass(frozen=True)
class HumanInteractionPolicy:
    """Explicit limits for a bounded voice-confirmation conversation."""

    max_clarification_turns: int = 3
    max_dialogue_turns: int = 8
    confirmation_phrases: tuple[str, ...] = (
        "yes",
        "yes proceed",
        "proceed",
        "execute",
        "do it",
        "looks good",
    )
    cancellation_phrases: tuple[str, ...] = ("no", "nope", "cancel task")

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_clarification_turns, int)
            or isinstance(self.max_clarification_turns, bool)
            or not 1 <= self.max_clarification_turns <= 5
        ):
            raise ValueError("max_clarification_turns must be an integer in [1, 5]")
        if (
            not isinstance(self.max_dialogue_turns, int)
            or isinstance(self.max_dialogue_turns, bool)
            or not 2 <= self.max_dialogue_turns <= 16
        ):
            raise ValueError("max_dialogue_turns must be an integer in [2, 16]")
        for name, phrases in (
            ("confirmation_phrases", self.confirmation_phrases),
            ("cancellation_phrases", self.cancellation_phrases),
        ):
            if (
                not isinstance(phrases, tuple)
                or not phrases
                or len({_normalized(phrase) for phrase in phrases}) != len(phrases)
            ):
                raise ValueError(f"{name} must contain unique non-empty phrases")


@dataclass(frozen=True)
class PlanProposal:
    """A non-actuating plan that requires a fresh explicit confirmation."""

    confirmation_id: str
    prompt: str
    prompt_audio: bytes | None
    result: PhysicalAIResult

    def __post_init__(self) -> None:
        if not isinstance(self.confirmation_id, str) or len(self.confirmation_id) != 16:
            raise ValueError("confirmation_id must be a 16-character plan binding")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Plan proposal prompt must be non-empty")
        if self.prompt_audio is not None and not isinstance(self.prompt_audio, bytes):
            raise TypeError("Plan proposal audio must be bytes or null")
        if not isinstance(self.result, PhysicalAIResult):
            raise TypeError("Plan proposal requires a PhysicalAIResult")
        if self.result.control.executed:
            raise ValueError("A plan proposal cannot contain executed motor control")


InteractionResult = PlanProposal | PhysicalAIResult | ClarificationResult | EmergencyStopResult


class SpokenEmergencyStopMonitor:
    """Listen for local stop language while a blocking motor request is active."""

    def __init__(
        self,
        capture_audio: Callable[[], bytes],
        transcribe_audio: Callable[[bytes], str],
        emergency_stop: Callable[[], Any],
    ) -> None:
        for name, callback in (
            ("capture_audio", capture_audio),
            ("transcribe_audio", transcribe_audio),
            ("emergency_stop", emergency_stop),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        self._capture_audio = capture_audio
        self._transcribe_audio = transcribe_audio
        self._emergency_stop = emergency_stop
        self._closed = Event()
        self._triggered = Event()
        self._lock = Lock()
        self._failure: Exception | None = None
        self._transcript: str | None = None
        self._thread: Thread | None = None

    @property
    def triggered(self) -> bool:
        return self._triggered.is_set()

    @property
    def transcript(self) -> str | None:
        with self._lock:
            return self._transcript

    @property
    def failure(self) -> Exception | None:
        with self._lock:
            return self._failure

    def _stop_for_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failure = exc
        self._triggered.set()
        self._emergency_stop()

    def _listen(self) -> None:
        try:
            while not self._closed.is_set():
                audio = self._capture_audio()
                if self._closed.is_set():
                    return
                if not isinstance(audio, bytes) or not audio:
                    raise ValueError("Emergency-stop microphone returned empty audio")
                transcript = self._transcribe_audio(audio)
                if not isinstance(transcript, str):
                    raise TypeError("Emergency-stop transcription must be text")
                transcript = " ".join(transcript.strip().split())
                if not transcript:
                    continue
                if is_emergency_stop_command(transcript):
                    with self._lock:
                        self._transcript = transcript
                    self._triggered.set()
                    self._emergency_stop()
                    return
        except Exception as exc:
            self._stop_for_failure(exc)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Emergency-stop monitor can only be started once")
        self._thread = Thread(
            target=self._listen,
            name="moira-spoken-emergency-stop",
            daemon=True,
        )
        self._thread.start()

    def close(self, timeout_seconds: float = 10.0) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._closed.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(float(timeout_seconds))
        if thread.is_alive():
            raise TimeoutError("Emergency-stop monitor did not shut down")


class HumanAwarePhysicalSession:
    """Require plan preview and explicit confirmation before physical execution."""

    def __init__(
        self,
        session: PhysicalSession,
        *,
        user_id: str,
        workspace: dict[str, Any],
        policy: HumanInteractionPolicy | None = None,
    ) -> None:
        if not isinstance(session, PhysicalSession):
            raise TypeError("session must be a PhysicalSession")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not isinstance(workspace, dict):
            raise TypeError("workspace must be a dictionary")
        self.session = session
        self.user_id = user_id
        self.workspace = dict(workspace)
        self.policy = policy or HumanInteractionPolicy()
        self._dialogue: list[DialogueTurn] = []
        self._clarification_turns = 0
        self._pending: PlanProposal | None = None

    @property
    def dialogue(self) -> tuple[DialogueTurn, ...]:
        return tuple(self._dialogue)

    @property
    def pending(self) -> PlanProposal | None:
        return self._pending

    def is_confirmation_response(self, text: str) -> bool:
        normalized = _normalized(text)
        return normalized in {
            _normalized(value) for value in self.policy.confirmation_phrases
        }

    def _append(self, role: str, text: str) -> None:
        self._dialogue.append(DialogueTurn(role, text))
        excess = len(self._dialogue) - self.policy.max_dialogue_turns
        if excess > 0:
            del self._dialogue[:excess]

    def _stop_result(self, transcript: str, reason: str, *, speak: bool) -> EmergencyStopResult:
        issues = self.session.system.emergency_stop()
        audio = (
            self.session.system.synthesize_speech(reason, self.user_id) if speak else None
        )
        self._pending = None
        self._append("user", transcript)
        self._append("assistant", reason)
        return EmergencyStopResult(transcript, reason, audio, issues, ())

    @staticmethod
    def _confirmation_id(result: PhysicalAIResult) -> str:
        targets = {
            item.id: list(item.position_m)
            for item in result.world.objects
            if item.id in result.intent.target_object_ids
        }
        encoded = json.dumps(
            {
                "intent": asdict(result.intent),
                "candidate": asdict(result.plan.candidate),
                "targets": targets,
                "observed_at": result.world.observed_at,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def plan(
        self,
        *,
        instruction: str | None = None,
        audio: bytes | None = None,
        tactile_samples: tuple[TactileSample, ...] = (),
        friction_coefficient: float | None = None,
        robot_state: RobotState | None = None,
        speak: bool = True,
    ) -> PlanProposal | ClarificationResult | EmergencyStopResult:
        if instruction is not None and is_emergency_stop_command(instruction):
            return self._stop_result(
                instruction,
                "Task cancelled. Motor control is stopped and remains latched.",
                speak=speak,
            )
        result = self.session.run(
            user_id=self.user_id,
            instruction=instruction,
            audio=audio,
            workspace=self.workspace,
            dialogue=self.dialogue,
            tactile_samples=tactile_samples,
            friction_coefficient=friction_coefficient,
            robot_state=robot_state,
            execute=False,
            speak=speak,
        )
        if isinstance(result, EmergencyStopResult):
            self._pending = None
            self._append("user", result.transcript)
            self._append("assistant", result.response_text)
            return result
        transcript = (
            result.transcript
            if isinstance(result, ClarificationResult)
            else result.intent.transcript
        )
        self._append("user", transcript)
        if isinstance(result, ClarificationResult):
            self._clarification_turns += 1
            self._pending = None
            self._append("assistant", result.question)
            if self._clarification_turns > self.policy.max_clarification_turns:
                return self._stop_result(
                    transcript,
                    "I could not resolve the request safely. The task is cancelled.",
                    speak=speak,
                )
            return result
        self._clarification_turns = 0
        confirmation_id = self._confirmation_id(result)
        prompt = (
            f"I heard: {result.intent.transcript}. I selected {result.plan.candidate.id}. "
            "Say yes to proceed, give a correction, or say stop."
        )
        prompt_audio = (
            self.session.system.synthesize_speech(prompt, self.user_id) if speak else None
        )
        proposal = PlanProposal(confirmation_id, prompt, prompt_audio, result)
        self._pending = proposal
        self._append("assistant", prompt)
        return proposal

    def respond(
        self,
        *,
        confirmation_id: str | None,
        instruction: str | None = None,
        audio: bytes | None = None,
        robot_state: RobotState | None = None,
        tactile_samples: tuple[TactileSample, ...] = (),
        friction_coefficient: float | None = None,
        speak: bool = True,
    ) -> InteractionResult:
        if (instruction is None) == (audio is None):
            raise ValueError("Provide exactly one interaction response as text or audio")
        transcript = (
            instruction
            if instruction is not None
            else self.session.system.transcribe_audio(audio or b"")
        )
        normalized = _normalized(transcript)
        if is_emergency_stop_command(transcript) or normalized in {
            _normalized(value) for value in self.policy.cancellation_phrases
        }:
            return self._stop_result(
                transcript,
                "Task cancelled. Motor control is stopped and remains latched.",
                speak=speak,
            )
        pending = self._pending
        confirmations = {_normalized(value) for value in self.policy.confirmation_phrases}
        if normalized in confirmations and pending is None:
            raise RuntimeError("There is no active plan to confirm")
        if pending is not None and normalized in confirmations:
            if confirmation_id is None or not hmac.compare_digest(
                pending.confirmation_id, confirmation_id
            ):
                self._pending = None
                raise RuntimeError("Confirmation is missing or belongs to a stale plan")
            if robot_state is None:
                raise ValueError("Confirmed physical execution requires a robot_state")
            self._pending = None
            self._append("user", transcript)
            try:
                result = self.session.run(
                    user_id=self.user_id,
                    instruction=pending.result.intent.transcript,
                    workspace=self.workspace,
                    dialogue=self.dialogue,
                    tactile_samples=tactile_samples,
                    friction_coefficient=friction_coefficient,
                    robot_state=robot_state,
                    execute=True,
                    execution_confirmed=True,
                    speak=speak,
                    confirmed_intent=pending.result.intent,
                    confirmed_candidate=pending.result.plan.candidate,
                )
            except ConfirmedPlanChangedError:
                return self.plan(
                    instruction=pending.result.intent.transcript,
                    tactile_samples=tactile_samples,
                    friction_coefficient=friction_coefficient,
                    robot_state=robot_state,
                    speak=speak,
                )
            if isinstance(result, ClarificationResult):
                self._append("assistant", result.question)
            elif isinstance(result, EmergencyStopResult):
                self._append("assistant", result.response_text)
            return result
        self._pending = None
        return self.plan(
            instruction=transcript,
            tactile_samples=tactile_samples,
            friction_coefficient=friction_coefficient,
            robot_state=robot_state,
            speak=speak,
        )

    def emergency_stop(
        self,
        reason: str = "Emergency stop requested. Motor control remains latched.",
        *,
        speak: bool = False,
    ) -> EmergencyStopResult:
        """Interrupt active hardware from another listener or operator thread."""

        return self._stop_result("stop", reason, speak=speak)
