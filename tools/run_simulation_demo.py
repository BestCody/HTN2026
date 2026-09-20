"""Run MExT's voice-to-action demo in the MuJoCo digital twin."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rich import box  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from moira.judge_demo import JudgeDemoTrace  # noqa: E402
from moira.physical import PipelineEvent  # noqa: E402
from moira.simulation_demo import run_simulation_demo  # noqa: E402


def _configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


class SimulationTrace:
    """Reuse the judge trace while labeling simulated control truthfully."""

    def __init__(self, *, color: bool | None = None) -> None:
        self.base = JudgeDemoTrace(color=color)
        self.console = self.base.console

    def heading(self) -> None:
        title = Text(justify="center")
        title.append("MExT\n", style="bold bright_cyan")
        title.append("OPENROUTER FOR ROBOTICS\n", style="bold bright_magenta")
        title.append("CHECKPOINT-BACKED DIGITAL TWIN", style="bold bright_white")
        self.console.print(Panel(title, box=box.DOUBLE, border_style="bright_cyan"))
        self.console.print(
            "VOICE  >>>  VISION  >>>  ROUTER  >>>  3 FUTURES  >>>  SAFE PLAN  >>>  MUJOCO",
            style="bold bright_yellow",
            justify="center",
        )
        self.console.print()

    def __call__(self, event: PipelineEvent) -> None:
        if event.kind == "pipeline.started":
            self.console.print(
                Panel(
                    Text.assemble(
                        ("SIM ONLINE", "bold bright_green"),
                        "   ",
                        (f"CAD ARM  {event.details['robot_model_id']}", "bright_cyan"),
                        "   ",
                        ("PHYSICAL PWM DISCONNECTED", "bold black on bright_yellow"),
                    ),
                    box=box.HEAVY,
                    border_style="bright_cyan",
                )
            )
            return
        if event.kind == "control.finished":
            executed = bool(event.details.get("executed"))
            success = bool(event.details.get("success"))
            if not executed:
                label = "PLAN PREVIEW COMPLETE // AWAITING VOICE CONFIRMATION"
                style = "bold black on bright_yellow"
                border = "bright_yellow"
            elif success:
                label = "MUJOCO EXECUTION COMPLETE"
                style = "bold black on bright_green"
                border = "bright_green"
            else:
                label = "MUJOCO EXECUTION FAILED"
                style = "bold white on bright_red"
                border = "bright_red"
            self.console.print(
                Panel(
                    Text(
                        label,
                        style=style,
                        justify="center",
                    ),
                    box=box.HEAVY,
                    border_style=border,
                )
            )
            return
        self.base(event)

    def summary(self, run) -> None:
        table = Table(
            title="MExT SIMULATION RESULT",
            box=box.DOUBLE_EDGE,
            border_style="bright_cyan",
        )
        table.add_column("Evidence", style="bold bright_magenta")
        table.add_column("Result", style="bright_white")
        table.add_row("Voice transcript", run.result.intent.transcript)
        table.add_row("Selected plan", run.result.plan.candidate.id)
        table.add_row(
            "Policy route",
            next(
                (item for item in run.routed_components if "policy" in item),
                "see journal",
            ),
        )
        table.add_row("World models", ", ".join(run.world_models))
        table.add_row("Execution", str(run.result.control.success))
        table.add_row("Camera verification", run.result.outcome.status)
        table.add_row("Overview + arm POV", str(run.artifacts.animation))
        self.console.print(table)

    def failure(self, exc: BaseException) -> None:
        self.base.failure(exc)


def main() -> int:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description=(
            "Route a spoken command through MExT and execute it on the CAD-derived "
            "MuJoCo arm instead of physical hardware."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "simulation_demo.json",
    )
    parser.add_argument(
        "--instruction",
        default="Move the red cube onto the green platform.",
        help="Text to synthesize and feed back through speech recognition.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Use a real recorded WAV command instead of synthesized demo speech.",
    )
    parser.add_argument("--live", action="store_true", help="Show a live split-screen view.")
    parser.add_argument("--no-speech", action="store_true", help="Do not synthesize the reply.")
    parser.add_argument("--open-animation", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    trace = SimulationTrace(color=False if args.no_color else None)
    trace.heading()
    try:
        if args.audio is not None and not args.audio.is_file():
            raise FileNotFoundError(f"Audio command does not exist: {args.audio}")
        run = run_simulation_demo(
            args.config,
            instruction=args.instruction,
            audio_path=args.audio,
            event_sink=trace,
            live_preview=args.live,
            speak=not args.no_speech,
        )
        trace.summary(run)
        if args.open_animation:
            if sys.platform != "win32":
                raise RuntimeError("--open-animation currently requires Windows")
            os.startfile(run.artifacts.animation)  # type: ignore[attr-defined]
        return 0
    except Exception as exc:
        trace.failure(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
