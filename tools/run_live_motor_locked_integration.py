"""Run the production camera path with motor execution structurally disabled."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.edge_components import FfmpegDshowCommandRecorder, LanCameraSource  # noqa: E402
from moira.judge_demo import JudgeDemoTrace  # noqa: E402
from moira.physical import ClarificationResult, PhysicalAIResult  # noqa: E402
from moira.production import (  # noqa: E402
    load_physical_runtime_config,
    prepare_physical_workspace,
)
from moira.software_integration import (  # noqa: E402
    build_software_integration_session,
    commanded_home_state,
    load_software_integration_config,
    validate_software_integration_result,
)
from moira.wake_word import command_follows_wake_word, wake_word_detected  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integration-config",
        type=Path,
        default=ROOT / "config" / "software_integration.json",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT / "config" / "pi4_runtime.json",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / "config" / "live_demo_workspace.json",
    )
    parser.add_argument(
        "--audio-source",
        choices=("microphone", "fixture"),
        default="microphone",
        help="Use the laptop microphone by default; fixture is for reproducible tests",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Record one command immediately without waiting for the wake word",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    integration = load_software_integration_config(args.integration_config)
    runtime = load_physical_runtime_config(args.runtime_config)
    raw_workspace = json.loads(args.workspace.read_text(encoding="utf-8"))
    if raw_workspace.pop("schema_version", None) != 1:
        raise ValueError("live demo workspace schema_version must be 1")
    workspace = prepare_physical_workspace(
        runtime,
        raw_workspace,
        camera_id=runtime.camera.camera_id,
    )
    if runtime.camera.transport != "lan_http":
        raise RuntimeError("live motor-locked integration requires the Pi LAN camera")
    camera_url = os.environ.get(runtime.camera.url_env or "")
    camera_token = os.environ.get(runtime.camera.token_env or "")
    if not camera_url or not camera_token:
        raise RuntimeError("Pi camera URL and token must be configured in .env")
    camera = LanCameraSource(
        camera_url,
        token=camera_token,
        camera_id=runtime.camera.camera_id,
        rotation_degrees=runtime.camera.rotation_degrees,
    )
    trace = JudgeDemoTrace()
    trace.heading(integration)
    session, _memory, driver, models = build_software_integration_session(
        integration,
        camera_source=camera,
        event_sink=trace,
    )
    try:
        if args.audio_source == "microphone":
            microphone = os.environ.get(runtime.audio.device_env, "")
            if not microphone:
                raise RuntimeError(f"Set {runtime.audio.device_env} to the Windows microphone name")
            recorder = FfmpegDshowCommandRecorder(
                microphone,
                sample_rate_hz=runtime.audio.sample_rate_hz,
                channels=runtime.audio.channels,
            )
            if args.one_shot:
                print(
                    f"Listening for {runtime.audio.command_duration_seconds} seconds. "
                    "Speak the robot command now...",
                    flush=True,
                )
                audio = recorder.record(runtime.audio.command_duration_seconds)
            else:
                print(
                    f'Wake listener armed. Say "Hey {runtime.audio.wake_word.title()}" '
                    "followed by a command. Press Ctrl+C to exit.",
                    flush=True,
                )
                while True:
                    candidate_audio = recorder.record(runtime.audio.wake_window_seconds)
                    transcript = session.system.transcribe_audio(
                        candidate_audio,
                        allow_empty=True,
                    )
                    if not wake_word_detected(transcript, runtime.audio.wake_word):
                        heard = transcript if transcript else "<silence>"
                        print(f"Wake word absent; ignored: {heard}", flush=True)
                        continue
                    print(f"Wake word detected: {transcript}", flush=True)
                    if command_follows_wake_word(transcript, runtime.audio.wake_word):
                        audio = candidate_audio
                    else:
                        print(
                            f"Listening for the command for "
                            f"{runtime.audio.command_duration_seconds} seconds...",
                            flush=True,
                        )
                        audio = recorder.record(runtime.audio.command_duration_seconds)
                    break
        else:
            audio = integration.command_audio.read_bytes()
        try:
            result = session.run(
                user_id="demo-user",
                audio=audio,
                workspace=workspace,
                robot_state=commanded_home_state(
                    models,
                    max_gripper_width_m=integration.assumptions["max_gripper_width_m"],
                ),
                execute=False,
                speak=True,
            )
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "status": "clarification",
                        "motor_execution": False,
                        "question": str(exc),
                        "instruction": "Correct the scene or command, then run Charlie again.",
                    },
                    indent=2,
                )
            )
            return 2
        except RuntimeError as exc:
            if not str(exc).startswith("Every model-predicted plan was unsafe;"):
                raise
            print(
                json.dumps(
                    {
                        "status": "unsafe_plan",
                        "motor_execution": False,
                        "reason": str(exc),
                        "instruction": (
                            "Use a small object within the arm's reachable arc. "
                            "Keep the camera and robot fixed, then try again."
                        ),
                    },
                    indent=2,
                )
            )
            return 2
        if isinstance(result, ClarificationResult):
            print(
                json.dumps(
                    {
                        "status": "clarification",
                        "motor_execution": False,
                        "question": result.question,
                        "instruction": (
                            "Place the named object and destination inside the camera "
                            "view, then answer Charlie's clarification or rerun."
                        ),
                    },
                    indent=2,
                )
            )
            return 2
        if not isinstance(result, PhysicalAIResult):
            raise TypeError("live integration returned an unknown result")
        run = validate_software_integration_result(
            integration,
            result,
            driver,
            required_transcript_terms=(),
            generic_pick_place=True,
        )
        response_path = ROOT / "outputs" / "live_demo_response.wav"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_bytes(run.result.response_audio or b"")
        trace.summary(run, response_path)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "motor_execution": run.result.control.executed,
                    "camera_id": runtime.camera.camera_id,
                    "transcript": run.transcript,
                    "objects": [item.id for item in run.result.world.objects],
                    "candidate_count": len(run.result.candidates),
                    "selected_plan": run.result.plan.candidate.id,
                    "selected_safe": run.result.plan.simulation.safe,
                    "prediction_horizons_s": [
                        item.horizon_seconds for item in run.result.simulations
                    ],
                    "world_models": list(run.world_models),
                },
                indent=2,
            )
        )
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
