"""Run Charlie's real non-actuating pipeline with a live judge-facing trace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.brand import DISPLAY_NAME  # noqa: E402
from moira.judge_demo import JudgeDemoTrace  # noqa: E402
from moira.software_integration import (  # noqa: E402
    load_software_integration_config,
    run_human_error_integration,
    run_software_integration,
)


def _configure_utf8_output() -> None:
    """Keep Rich glyphs intact in Windows Terminal and redirected demo logs."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _play_audio(path: Path) -> None:
    if sys.platform != "win32":
        raise RuntimeError("--play-response currently requires Windows")
    import winsound

    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def main() -> int:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description=(
            f"Show live {DISPLAY_NAME} model routing and prediction decisions in the terminal."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "software_integration.json",
    )
    parser.add_argument(
        "--response-audio",
        type=Path,
        default=ROOT / "outputs" / "judge_demo_response.wav",
    )
    parser.add_argument("--play-response", action="store_true")
    parser.add_argument(
        "--human-error-demo",
        action="store_true",
        help="Show ambiguous speech, clarification, correction, and safe replanning.",
    )
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    trace = JudgeDemoTrace(color=False if args.no_color else None)
    try:
        config = load_software_integration_config(args.config)
        trace.heading(config)
        run = (
            run_human_error_integration(
                config,
                event_sink=trace,
                correction_sink=trace.human_correction,
            )
            if args.human_error_demo
            else run_software_integration(config, event_sink=trace)
        )
        response = run.result.response_audio
        if not response:
            raise RuntimeError("TTS returned no response audio")
        args.response_audio.parent.mkdir(parents=True, exist_ok=True)
        args.response_audio.write_bytes(response)
        trace.summary(run, args.response_audio)
        if args.play_response:
            _play_audio(args.response_audio)
        return 0
    except Exception as exc:
        trace.failure(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
