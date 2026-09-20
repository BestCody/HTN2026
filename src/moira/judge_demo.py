"""Animated terminal presentation for the non-actuating MoIRA judge demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import RLock
from typing import TextIO

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from .physical import PhysicalAIResult, PipelineEvent
from .software_integration import SoftwareIntegrationConfig, SoftwareIntegrationRun


class JudgeDemoTrace:
    """Turn real pipeline events into a high-energy, thread-safe terminal show."""

    _LOGO = (
        " M   M   OOO   III  RRRR     A   ",
        " MM MM  O   O   I   R   R   A A  ",
        " M M M  O   O   I   RRRR   AAAAA ",
        " M   M   OOO   III  R  R  A     A",
    )

    def __init__(self, stream: TextIO | None = None, *, color: bool | None = None) -> None:
        self.stream = stream or sys.stdout
        if color is None:
            color = bool(getattr(self.stream, "isatty", lambda: False)()) and not os.environ.get(
                "NO_COLOR"
            )
        self.color = color
        self.console = Console(
            file=self.stream,
            force_terminal=color,
            no_color=not color,
            color_system="truecolor" if color else None,
            highlight=False,
            width=None if color else 120,
        )
        self._lock = RLock()
        self._progress = Progress(
            SpinnerColumn("dots12", style="bold bright_cyan"),
            TextColumn("[bold bright_white]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
            refresh_per_second=12,
        )
        self._progress_started = False
        self._tasks: dict[str, list[int]] = {}

    @staticmethod
    def _joined(values: object, *, empty: str = "none") -> str:
        if not values:
            return empty
        if isinstance(values, dict):
            return ", ".join(f"{key}={value}" for key, value in values.items())
        if isinstance(values, (tuple, list, set)):
            return ", ".join(str(value) for value in values)
        return str(values)

    @staticmethod
    def _score_bar(score: float, width: int = 20) -> str:
        filled = max(0, min(width, round(score * width)))
        return "█" * filled + "░" * (width - filled)

    def _line(self, event: PipelineEvent, label: str, message: str, style: str) -> None:
        line = Text()
        line.append(f"[{event.elapsed_seconds:7.2f}s] ", style="dim")
        line.append(f"{label:<11}", style=f"bold {style}")
        line.append(f" {message}", style="white")
        self.console.print(line)

    def _start_progress(self) -> None:
        if not self._progress_started:
            self._progress.start()
            self._progress_started = True

    def _start_task(self, key: str, description: str) -> None:
        self._start_progress()
        task_id = self._progress.add_task(description, total=None)
        self._tasks.setdefault(key, []).append(task_id)

    def _finish_task(self, key: str) -> None:
        tasks = self._tasks.get(key)
        if not tasks:
            return
        self._progress.remove_task(tasks.pop(0))
        if not tasks:
            self._tasks.pop(key, None)

    def _stop_progress(self) -> None:
        if self._progress_started:
            self._progress.stop()
            self._progress_started = False
            self._tasks.clear()

    def __call__(self, event: PipelineEvent) -> None:
        """Render one event; callbacks may arrive from parallel simulation threads."""

        with self._lock:
            self._render(event)

    def _render(self, event: PipelineEvent) -> None:
        details = event.details
        if event.kind == "pipeline.started":
            self.console.print(
                Panel(
                    Text.assemble(
                        ("SYSTEM ONLINE", "bold bright_green"),
                        "   ",
                        (f"ROBOT  {details['robot_model_id']}", "bright_cyan"),
                        "   ",
                        (f"CAMERAS  {details['camera_count']}", "bright_cyan"),
                        "   ",
                        ("PWM LOCKED", "bold black on bright_yellow"),
                    ),
                    border_style="bright_cyan",
                    box=box.HEAVY,
                    padding=(0, 2),
                )
            )
            return
        if event.kind == "route.selected":
            layer = event.layer.value.upper() if event.layer else "MODEL"
            via = event.router or "exact-contract"
            self._line(
                event,
                "MODEL ROUTE",
                (
                    f"{layer} / {event.capability}  >>>  {event.component_id}  "
                    f"[{event.model}]  runtime={event.runtime}  via={via}"
                ),
                "bright_cyan",
            )
            return
        if event.kind == "component.started":
            self._start_task(
                f"component:{event.component_id}",
                f"{event.component_id}  //  {event.model}",
            )
            return
        if event.kind == "component.completed":
            self._finish_task(f"component:{event.component_id}")
            latency = float(details.get("latency_seconds", 0.0))
            self._line(
                event,
                "COMPLETE",
                f"{event.component_id} -> {details.get('output_type')}  ({latency:.3f}s)",
                "bright_green",
            )
            return
        if event.kind == "component.failed":
            self._finish_task(f"component:{event.component_id}")
            latency = float(details.get("latency_seconds", 0.0))
            self._line(
                event,
                "FAILED",
                f"{event.component_id}: {details.get('error_type')} after {latency:.3f}s",
                "bold bright_red",
            )
            return
        if event.kind == "speech.transcribed":
            self.console.print(
                Panel(
                    Text.assemble(
                        ("VOICE COMMAND LOCKED\n", "bold bright_yellow"),
                        (f'“{details["text"]}”', "bold bright_white"),
                    ),
                    title="[bold bright_yellow]WHISPER LARGE V3 TURBO[/]",
                    border_style="bright_yellow",
                    box=box.ROUNDED,
                )
            )
            return
        if event.kind == "scene.perceived":
            objects = details.get("objects", ())
            table = Table(box=None, show_header=False, padding=(0, 2))
            table.add_column(style="bold bright_cyan")
            table.add_column(style="bright_white")
            for item in objects:
                table.add_row(
                    "TARGET LOCK",
                    f"{item['label']}  |  confidence {float(item['confidence']):.0%}",
                )
            if not objects:
                table.add_row("NO TARGETS", "No grounded objects")
            self.console.print(
                Panel(
                    table,
                    title="[bold bright_cyan]GLM VISION // SCENE GROUNDED[/]",
                    border_style="bright_cyan",
                    box=box.ROUNDED,
                )
            )
            return
        if event.kind == "memory.recalled":
            self.console.print(
                Panel(
                    Text.assemble(
                        ("PREFERENCES  ", "bold bright_magenta"),
                        self._joined(details.get("preferences")),
                        "\n",
                        ("ADAPTATION   ", "bold bright_magenta"),
                        self._joined(details.get("accommodations")),
                    ),
                    title="[bold bright_magenta]PERSONAL INTELLIGENCE[/]",
                    border_style="bright_magenta",
                    box=box.ROUNDED,
                )
            )
            return
        if event.kind == "intent.grounded":
            self._line(
                event,
                "NLP INTENT",
                (
                    f"action={details.get('action')}  |  "
                    f"roles={self._joined(details.get('object_roles'))}  |  "
                    f"constraints={self._joined(details.get('constraints'))}"
                ),
                "bright_yellow",
            )
            return
        if event.kind == "clarification.required":
            self.console.print(
                Panel(
                    Text.assemble(
                        ("HUMAN INPUT IS AMBIGUOUS\n", "bold bright_yellow"),
                        (str(details.get("question")), "bold bright_white"),
                        "\n",
                        ("No plan executed. Waiting for a correction.", "dim"),
                    ),
                    title="[bold black on bright_yellow] CLARIFICATION REQUIRED [/]",
                    border_style="bright_yellow",
                    box=box.HEAVY,
                    padding=(1, 3),
                )
            )
            return
        if event.kind == "safety.emergency_stop":
            self._stop_progress()
            issues = self._joined(details.get("issues"))
            self.console.print(
                Panel(
                    Text.assemble(
                        ("EMERGENCY STOP LATCHED\n", "bold bright_red"),
                        (str(details.get("response")), "bold bright_white"),
                        "\n",
                        (f"stop-hook issues: {issues}", "dim"),
                    ),
                    title="[bold white on red] MOTION ABORTED [/]",
                    border_style="bright_red",
                    box=box.HEAVY,
                )
            )
            return
        if event.kind == "plans.generated":
            candidates = "\n".join(
                f"  {index}. {candidate}"
                for index, candidate in enumerate(details.get("candidate_ids", ()), start=1)
            )
            self.console.print(
                Panel(
                    Text(candidates, style="bold bright_white"),
                    title=(
                        "[bold bright_magenta]PLANNER CHAIN // "
                        f"{details.get('candidate_count')} PARALLEL FUTURES[/]"
                    ),
                    border_style="bright_magenta",
                    box=box.DOUBLE,
                )
            )
            return
        if event.kind == "semantic_router.started":
            candidate = str(details.get("candidate_id"))
            self._start_task(
                f"router:{candidate}",
                f"MiniLM ranking specialist models for {candidate}",
            )
            self._line(
                event,
                "ROUTER",
                f"MiniLM comparing compatible manipulation specialists for {candidate}",
                "bright_magenta",
            )
            return
        if event.kind == "semantic_router.completed":
            candidate = str(details.get("candidate_id"))
            self._finish_task(f"router:{candidate}")
            latency = float(details.get("latency_seconds", 0.0))
            self.console.print(
                Panel(
                    Text.assemble(
                        ("MiniLM selected ", "bright_white"),
                        (str(event.component_id), "bold bright_magenta"),
                        "\n",
                        (str(event.model), "bold bright_cyan"),
                        "\n",
                        (f"candidate {candidate}  |  {latency:.3f}s", "dim"),
                    ),
                    title="[bold bright_magenta]MoIRA ROUTER DECISION[/]",
                    border_style="bright_magenta",
                    box=box.HEAVY,
                    padding=(0, 2),
                )
            )
            return
        if event.kind == "simulation.completed":
            score = float(details.get("score", 0.0))
            safe = bool(details.get("safe"))
            status = "SAFE" if safe else "BLOCKED"
            style = "bold bright_green" if safe else "bold bright_red"
            prediction = Text()
            prediction.append(f"{details.get('candidate_id')}\n", style="bold bright_white")
            prediction.append(self._score_bar(score), style=style)
            prediction.append(f"  score={score:.3f}  {status}\n", style=style)
            prediction.append(
                (
                    f"{float(details.get('horizon_seconds', 0.0)):.2f}s future  |  "
                    f"models: {self._joined(details.get('world_models'))}"
                ),
                style="dim",
            )
            self.console.print(
                Panel(
                    prediction,
                    title="[bold bright_cyan]PARALLEL PHYSICS PREDICTION[/]",
                    border_style="bright_green" if safe else "bright_red",
                    box=box.ROUNDED,
                )
            )
            return
        if event.kind == "plan.selected":
            score = float(details.get("score", 0.0))
            self.console.print(
                Panel(
                    Text.assemble(
                        ("WINNING FUTURE\n", "bold bright_green"),
                        (str(details.get("candidate_id")), "bold bright_white"),
                        "\n",
                        (self._score_bar(score), "bold bright_green"),
                        (f"  score={score:.3f}  SAFE", "bold bright_green"),
                    ),
                    title="[bold black on bright_green] DECISION LOCKED [/]",
                    border_style="bright_green",
                    box=box.DOUBLE,
                    padding=(1, 3),
                )
            )
            return
        if event.kind == "scene.revalidated":
            drift = details.get("target_drift_m", {})
            self._line(
                event,
                "SCENE CHECK",
                (
                    f"targets stable: {self._joined(drift)}  |  limit="
                    f"{float(details.get('maximum_allowed_drift_m', 0.0)):.3f}m"
                ),
                "bright_green",
            )
            return
        if event.kind == "scene.execution_revalidated":
            self._line(
                event,
                "LIVE SCENE GUARD",
                (
                    f"completed={self._joined(details.get('completed_actions'))}  |  "
                    f"watching={self._joined(details.get('guarded_target_ids'))}"
                ),
                "bright_green",
            )
            return
        if event.kind == "scene.changed":
            self._stop_progress()
            self.console.print(
                Panel(
                    Text(str(details.get("reason")), style="bold bright_red"),
                    title="[bold white on red] SCENE CHANGED - EXECUTION BLOCKED [/]",
                    border_style="bright_red",
                    box=box.HEAVY,
                )
            )
            return
        if event.kind == "plan.confirmation_invalidated":
            self._stop_progress()
            self.console.print(
                Panel(
                    Text(
                        "The fresh intent or selected plan differs from the confirmed plan. "
                        "No motor command was sent; confirmation is required again.",
                        style="bold bright_yellow",
                    ),
                    title="[bold black on bright_yellow] PLAN CHANGED - RECONFIRM [/]",
                    border_style="bright_yellow",
                    box=box.HEAVY,
                )
            )
            return
        if event.kind == "control.finished":
            if details.get("executed"):
                message = Text(
                    f"PHYSICAL EXECUTION COMPLETE  //  success={details.get('success')}",
                    style="bold black on bright_green",
                    justify="center",
                )
                border = "bright_green"
            else:
                message = Text(
                    "MOTOR OUTPUT LOCKED  //  ZERO PWM COMMANDS SENT",
                    style="bold black on bright_yellow",
                    justify="center",
                )
                border = "bright_yellow"
            self.console.print(Panel(message, border_style=border, box=box.HEAVY))
            return
        if event.kind == "outcome.verified":
            mode = "camera" if details.get("camera_verified") else "planned/telemetry"
            self._line(
                event,
                "VERIFIED",
                (
                    f"status={details.get('status')}  confidence="
                    f"{float(details.get('confidence', 0.0)):.0%}  source={mode}"
                ),
                "bright_green",
            )
            return
        if event.kind == "feedback.saved":
            self._line(
                event,
                "LEARNING",
                (
                    "feedback committed  |  prediction-error samples="
                    f"{details.get('prediction_error_samples')}"
                ),
                "bright_magenta",
            )
            return
        if event.kind == "pipeline.completed":
            self._stop_progress()
            self.console.print(
                Panel(
                    Text(
                        (
                            f"PIPELINE COMPLETE  //  plan={details.get('selected_plan')}  //  "
                            f"outcome={details.get('outcome')}  //  "
                            f"speech={details.get('response_audio_bytes')} bytes"
                        ),
                        style="bold bright_green",
                        justify="center",
                    ),
                    border_style="bright_green",
                    box=box.HEAVY,
                )
            )

    def heading(self, config: SoftwareIntegrationConfig) -> None:
        with self._lock:
            logo = Text(justify="center")
            for index, line in enumerate(self._LOGO):
                style = "bold bright_cyan" if index % 2 == 0 else "bold bright_magenta"
                logo.append(line, style=style)
                logo.append("\n")
            logo.append("PHYSICAL AI MODEL ROUTER", style="bold bright_white")
            logo.append("\nOPENROUTER FOR THE PHYSICAL WORLD", style="bold bright_magenta")
            self.console.print(
                Panel(
                    logo,
                    border_style="bright_cyan",
                    box=box.DOUBLE,
                    padding=(1, 4),
                )
            )
            topology = Text(justify="center")
            stages = (
                "MIC",
                "WHISPER",
                "VISION + MEMORY",
                "MoIRA ROUTER",
                "3 FUTURES",
                "SAFE PLAN",
            )
            for index, stage in enumerate(stages):
                if index:
                    topology.append("  >>>  ", style="bold bright_magenta")
                topology.append(f" {stage} ", style="bold black on bright_cyan")
            self.console.print(topology)
            self.console.print(
                Text(
                    (
                        "LIVE SPECIALIST SELECTION  //  "
                        f"{config.profile.simulation_horizon_seconds:.2f}s PREDICTION  //  "
                        "PLAN-ONLY SAFETY MODE"
                    ),
                    style="bold bright_yellow",
                    justify="center",
                )
            )
            self.console.print()

    def human_correction(self, question: str, correction: str) -> None:
        """Reveal the bounded correction turn used by the human-error scenario."""

        with self._lock:
            self.console.print(
                Panel(
                    Text.assemble(
                        ("ROBOT ASKED  ", "bold bright_yellow"),
                        question,
                        "\n",
                        ("USER CORRECTED  ", "bold bright_cyan"),
                        correction,
                        "\n",
                        ("Re-running grounding and routing from the corrected request.", "dim"),
                    ),
                    title="[bold black on bright_cyan] HUMAN RECOVERY TURN [/]",
                    border_style="bright_cyan",
                    box=box.DOUBLE,
                    padding=(1, 3),
                )
            )

    def summary(self, run: SoftwareIntegrationRun, response_path: Path) -> None:
        result: PhysicalAIResult = run.result
        with self._lock:
            self._stop_progress()
            table = Table(
                title="CANDIDATE FUTURES",
                title_style="bold bright_cyan",
                border_style="bright_cyan",
                header_style="bold bright_magenta",
                box=box.DOUBLE_EDGE,
                expand=True,
            )
            table.add_column("Candidate", ratio=3)
            table.add_column("Predicted score", ratio=2)
            table.add_column("Safety", justify="center")
            table.add_column("Horizon", justify="right")
            table.add_column("Decision", justify="center")
            for simulation in result.simulations:
                selected = simulation.plan_id == result.plan.candidate.id
                safe = "SAFE" if simulation.safe else "BLOCKED"
                style = "bold bright_green" if selected else "white"
                table.add_row(
                    simulation.plan_id,
                    f"{self._score_bar(simulation.score, 12)} {simulation.score:.3f}",
                    safe,
                    f"{simulation.horizon_seconds:.2f}s",
                    "WINNER" if selected else "-",
                    style=style,
                )
            self.console.print(table)
            self.console.print(
                Panel(
                    Text.assemble(
                        ("SELECTED  ", "bold bright_green"),
                        (result.plan.candidate.id, "bold bright_white"),
                        "\n",
                        ("ROBOT SAYS  ", "bold bright_magenta"),
                        result.response_text,
                        "\n",
                        ("AUDIO  ", "bold bright_cyan"),
                        str(response_path),
                        "\n",
                        ("SAFETY  ", "bold bright_yellow"),
                        "MOTORS LOCKED - zero physical commands",
                    ),
                    title="[bold black on bright_green] DEMO COMPLETE [/]",
                    border_style="bright_green",
                    box=box.HEAVY,
                    padding=(1, 3),
                )
            )

    def failure(self, exc: BaseException) -> None:
        with self._lock:
            self._stop_progress()
            self.console.print(
                Panel(
                    Text(
                        f"{type(exc).__name__}: {exc}",
                        style="bold bright_red",
                        justify="center",
                    ),
                    title="[bold white on red] DEMO FAILED [/]",
                    border_style="bright_red",
                    box=box.HEAVY,
                )
            )
