"""Continuous Charlie voice -> specialist router -> guarded Pi gripper demo."""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "outputs" / "guarded_joint_state.json"
LEGACY_GRIPPER_STATE = ROOT / "outputs" / "guarded_gripper_state.json"
ENVELOPE_PATH = ROOT / "config" / "arm1_guarded_envelope.json"
sys.path.insert(0, str(ROOT / "src"))

from moira.cloud import BasetenEndpoint, JsonEndpoint, JsonHttpClient  # noqa: E402
from moira.edge_components import FfmpegDshowCommandRecorder  # noqa: E402
from moira.presentation_names import (  # noqa: E402
    routing_strategy_display_name,
    specialist_display_name,
)
from moira.wake_word import wake_word_detected  # noqa: E402

PI_EXERCISE = (
    "cd /home/client/moira-controller/app && "
    "/home/client/moira-controller/.venv/bin/python "
    "tools/pi_guarded_servo_calibration.py "
    "--config config/arm1_guarded_envelope.json "
    "exercise --joint {joint} --start {start} --target {target} "
    "--rate {rate} --armed"
)

PI_SEQUENCE = (
    "cd /home/client/moira-controller/app && "
    "/home/client/moira-controller/.venv/bin/python "
    "tools/pi_guarded_servo_calibration.py "
    "--config config/arm1_guarded_envelope.json sequence {steps} --armed"
)

ACTION_NAMES = {
    "pickup_from_hand": "Pick Up Object From Hand",
    "open": "Open Gripper",
    "close": "Close Gripper",
    "base_front": "Face Forward",
    "base_left": "Turn Left",
    "base_back": "Turn Back",
    "shoulder_lower": "Lower Arm",
    "shoulder_raise": "Raise Arm",
}

JOINT_NAMES = {
    "J1_BASE_YAW": "Base Rotation",
    "J2_SHOULDER": "Shoulder",
    "J3_GRIPPER": "Gripper",
}


def intent(text: str) -> str | None:
    words = set(re.findall(r"[a-z]+", text.casefold()))
    if words & {"pick", "grab", "take"} and words & {
        "hand",
        "thing",
        "object",
        "item",
        "pencil",
        "this",
    }:
        return "pickup_from_hand"
    if words & {"open", "release"} and words & {"gripper", "grippers", "claw", "claws"}:
        return "open"
    if words & {"close", "clamp"} and words & {"gripper", "grippers", "claw", "claws"}:
        return "close"
    if words & {"front", "forward"} and words & {"turn", "face", "base", "arm"}:
        return "base_front"
    if "left" in words and words & {"turn", "face", "base", "arm"}:
        return "base_left"
    if words & {"back", "backward", "behind"} and words & {"turn", "face", "base", "arm"}:
        return "base_back"
    if words & {"lower", "down", "drop"} and "arm" in words:
        return "shoulder_lower"
    if words & {"raise", "lift", "up"} and "arm" in words:
        return "shoulder_raise"
    return None


def exact_bounded_command(text: str) -> bool:
    """Recover a safe command when STT drops the wake-word prefix."""

    words = set(re.findall(r"[a-z]+", text.casefold()))
    permitted = {
        "open",
        "close",
        "release",
        "clamp",
        "the",
        "a",
        "please",
        "gripper",
        "grippers",
        "claw",
        "claws",
        "pencil",
        "pick",
        "grab",
        "take",
        "up",
        "this",
        "thing",
        "object",
        "item",
        "in",
        "my",
        "hand",
        "turn",
        "face",
        "base",
        "arm",
        "left",
        "front",
        "forward",
        "back",
        "backward",
        "behind",
        "lower",
        "down",
        "drop",
        "raise",
        "lift",
    }
    return intent(text) is not None and bool(words) and words <= permitted


class CharlieVoiceMotionDemo:
    """One terminal owns capture, routing, actuation, and presentation."""

    LOGO = (
        "  CCCC  H   H   AAA   RRRR   L      III  EEEEE",
        " C      H   H  A   A  R   R  L       I   E    ",
        " C      HHHHH  AAAAA  RRRR   L       I   EEEE ",
        "  CCCC  H   H  A   A  R  R   LLLLL  III  EEEEE",
    )

    def __init__(self, *, execute: bool, text: str | None) -> None:
        self.execute = execute
        self.text = text
        self.console = Console(highlight=False)
        envelope = json.loads(ENVELOPE_PATH.read_text(encoding="utf-8"))
        self.targets = envelope["voice_demo_targets"]
        self.rates = {
            name: float(value["maximum_rate_deg_s"])
            for name, value in envelope["servos"].items()
        }
        self.host = urlparse(os.environ["MOIRA_ROBOT_URL"]).hostname
        if not self.host:
            raise RuntimeError("MOIRA_ROBOT_URL has no Pi host")
        self.recorder = None
        if text is None:
            self.recorder = FfmpegDshowCommandRecorder(os.environ["MOIRA_MICROPHONE_DEVICE"])
        self.stt = JsonEndpoint(
            os.environ["MOIRA_STT_URL"],
            token=os.environ.get("MOIRA_LAN_TOKEN"),
            http=JsonHttpClient(timeout_seconds=45, allow_private_http=True),
        )
        self.router = BasetenEndpoint(
            os.environ["BASETEN_ROUTER_CHAIN_ID"],
            entity="chain",
            environment=os.environ.get("BASETEN_ROUTER_ENVIRONMENT", "development"),
            http=JsonHttpClient(timeout_seconds=120, attempts=1),
        )
        self.password = os.environ.get("MOIRA_PI_SSH_PASSWORD")
        if execute and not self.password:
            self.password = getpass.getpass("Pi SSH password: ")

    def banner(self) -> None:
        logo = Text("\n".join(self.LOGO), style="bold bright_cyan")
        logo.append("\n\n          LIVE SINGLE-ARM PHYSICAL INTELLIGENCE", style="bold bright_white")
        self.console.print(
            Panel(
                logo,
                subtitle=(
                    "[yellow]VOICE -> SPEECH SPECIALIST -> TASK ROUTER -> "
                    "MOVEMENT SPECIALIST -> ROBOT[/]"
                ),
                border_style="bright_cyan",
                box=box.HEAVY,
                padding=(1, 3),
            )
        )
        topology = Table(box=box.ROUNDED, border_style="blue", expand=True)
        topology.add_column("STAGE", style="bold bright_cyan")
        topology.add_column("ACTIVE COMPONENT", style="bright_white")
        topology.add_row("VOICE INPUT", "Laptop microphone")
        topology.add_row("SPEECH SPECIALIST", "Whisper Large V3 Turbo converts speech to text")
        topology.add_row("TASK ROUTER", "Fine-tuned MiniLM selects the right robot specialist")
        topology.add_row("MOVEMENT SPECIALIST", "Plans bounded single-arm movements")
        topology.add_row("ROBOT CONTROL", "Raspberry Pi -> base 0 | shoulder 1 | gripper 2")
        self.console.print(topology)

    def _transcribe(self) -> str:
        if self.text is not None:
            return self.text
        assert self.recorder is not None
        with self.console.status(
            '[bold bright_yellow]MIC LIVE -- say "Hey Charlie" and a motion command[/]',
            spinner="dots12",
        ):
            audio = self.recorder.record(5)
        with self.console.status(
            "[bold bright_cyan]SPEECH SPECIALIST // converting voice to text[/]",
            spinner="aesthetic",
        ):
            response = self.stt.predict(
                {"audio": {"base64": base64.b64encode(audio).decode("ascii")}}
            )
        return str(response.get("text", "")).strip()

    def _route(self, transcript: str) -> tuple[str | None, str | None]:
        with self.console.status(
            "[bold magenta]TASK ROUTER // selecting the best compatible specialist[/]",
            spinner="bouncingBall",
        ):
            response = self.router.predict(
                {
                    "layer": "manipulation",
                    "capability": "manipulation.select_compatible",
                    "context": {"routing_text": transcript},
                    "allowed_components": ["baseten-waypoint-policy"],
                }
            )
        if not isinstance(response, dict):
            return None, None
        return response.get("component_id"), response.get("strategy")

    def _positions(self) -> dict[str, int]:
        positions = {
            "J1_BASE_YAW": int(self.targets["base_front_deg"]),
            "J2_SHOULDER": int(self.targets["shoulder_raised_deg"]),
            "J3_GRIPPER": int(self.targets["gripper_open_deg"]),
        }
        if STATE_PATH.is_file():
            try:
                state = json.loads(STATE_PATH.read_text(encoding="utf-8-sig"))
                saved = state.get("positions") if isinstance(state, dict) else None
                if isinstance(saved, dict):
                    for joint in positions:
                        if isinstance(saved.get(joint), int):
                            positions[joint] = saved[joint]
            except (OSError, ValueError):
                pass
        elif LEGACY_GRIPPER_STATE.is_file():
            try:
                state = json.loads(LEGACY_GRIPPER_STATE.read_text(encoding="utf-8-sig"))
                if isinstance(state, dict) and state.get("last_command_deg") in (83, 180):
                    positions["J3_GRIPPER"] = state["last_command_deg"]
            except (OSError, ValueError):
                pass
        return positions

    @staticmethod
    def _save_positions(positions: dict[str, int]) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"positions": positions, "source": "charlie_voice_motion"}) + "\n",
            encoding="utf-8",
        )
        temporary.replace(STATE_PATH)

    def _plan(self, action: str, positions: dict[str, int]) -> list[dict[str, object]]:
        targets = self.targets
        simple = {
            "open": ("J3_GRIPPER", "gripper_open_deg"),
            "close": ("J3_GRIPPER", "gripper_closed_deg"),
            "base_front": ("J1_BASE_YAW", "base_front_deg"),
            "base_left": ("J1_BASE_YAW", "base_left_deg"),
            "base_back": ("J1_BASE_YAW", "base_back_deg"),
            "shoulder_lower": ("J2_SHOULDER", "shoulder_lowered_deg"),
            "shoulder_raise": ("J2_SHOULDER", "shoulder_raised_deg"),
        }
        if action in simple:
            joint, target_name = simple[action]
            return [{"joint": joint, "start": positions[joint], "target": int(targets[target_name])}]
        if action == "pickup_from_hand":
            return [
                {"joint": "J2_SHOULDER", "start": positions["J2_SHOULDER"], "target": int(targets["shoulder_lowered_deg"])},
                {"joint": "J3_GRIPPER", "start": int(targets["gripper_open_deg"]), "target": int(targets["gripper_open_deg"])},
                {"pause": float(targets["pickup_load_pause_seconds"])},
                {"joint": "J3_GRIPPER", "start": int(targets["gripper_open_deg"]), "target": int(targets["gripper_closed_deg"])},
                {"joint": "J2_SHOULDER", "start": int(targets["shoulder_lowered_deg"]), "target": int(targets["shoulder_raised_deg"])},
            ]
        raise ValueError(f"unknown bounded action: {action}")

    def _command_for_plan(self, plan: list[dict[str, object]]) -> str:
        if len(plan) == 1:
            step = plan[0]
            joint = str(step["joint"])
            return PI_EXERCISE.format(
                joint=joint,
                start=step["start"],
                target=step["target"],
                rate=self.rates[joint],
            )
        encoded = []
        for step in plan:
            if "pause" in step:
                encoded.append(f"--step pause:{step['pause']}")
            else:
                joint = str(step["joint"])
                encoded.append(
                    f"--step {joint}:{step['start']}:{step['target']}:{self.rates[joint]}"
                )
        return PI_SEQUENCE.format(steps=" ".join(encoded))

    def _actuate(self, command: str) -> None:
        import paramiko

        known_hosts = Path.home() / ".ssh" / "known_hosts"
        client = paramiko.SSHClient()
        if known_hosts.is_file():
            client.load_host_keys(str(known_hosts))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(self.host, username="client", password=self.password, timeout=8)
        try:
            _, stdout, stderr = client.exec_command(command, timeout=45)
            last_update = ""
            for line in stdout:
                raw = line.rstrip()
                try:
                    update = json.loads(raw)
                except ValueError:
                    continue
                if "command_deg" in update:
                    last_update = (
                        f"channel={update['channel']}  angle={update['command_deg']} deg  "
                        f"PWM={update['tick']}  t={update['elapsed_s']:.2f}s"
                    )
                    self.console.print(f"[dim]PI PWM[/]  [green]*[/] {last_update}", end="\r")
            if last_update:
                self.console.print(" " * (len(last_update) + 14), end="\r")
            error = stderr.read().decode("utf-8", "replace").strip()
            status = stdout.channel.recv_exit_status()
            if status:
                raise RuntimeError(f"Pi guarded movement failed ({status}): {error}")
        finally:
            client.close()

    def run_once(self, cycle: int) -> int:
        del cycle
        self.console.rule("[bold bright_cyan]LIVE VOICE INPUT[/]")
        transcript = self._transcribe()
        self.console.print(
            Panel(
                transcript or "<silence>",
                title="[bold bright_yellow]SPEECH SPECIALIST // TRANSCRIPT[/]",
                border_style="bright_yellow",
            )
        )
        action = intent(transcript)
        if not wake_word_detected(transcript, "charlie"):
            if not exact_bounded_command(transcript):
                self.console.print("[dim]WAKE WORD ABSENT  //  no route, no motion[/]")
                return 2
            self.console.print(
                "[bold bright_green]WAKE PREFIX RECOVERED[/]  //  "
                "exact bounded motion command recognized"
            )
        if action is None:
            self.console.print(
                Panel(
                    "Try open/close gripper, turn left/front/back, raise/lower arm, "
                    "or pick up this thing in my hand. No motion sent.",
                    title="[yellow]GROUNDING NEEDS CLARIFICATION[/]",
                    border_style="yellow",
                )
            )
            return 2
        selected, strategy = self._route(transcript)
        route = Table(box=box.DOUBLE_EDGE, border_style="magenta", expand=True)
        route.add_column("UNDERSTOOD COMMAND", style="bold bright_white")
        route.add_column("SELECTED SPECIALIST", style="bold bright_green")
        route.add_column("WHY IT WAS SELECTED", style="bright_magenta")
        route.add_row(
            ACTION_NAMES[action],
            specialist_display_name(selected),
            routing_strategy_display_name(strategy),
        )
        self.console.print(route)
        if selected != "baseten-waypoint-policy":
            self.console.print(
                Panel(
                    "The Task Router did not select Charlie's installed Movement Specialist.",
                    title="[red]ROUTE BLOCKED[/]",
                    border_style="red",
                )
            )
            return 2
        positions = self._positions()
        plan = self._plan(action, positions)
        motion = Table(box=box.ROUNDED, border_style="bright_green", expand=True)
        motion.add_column("STEP", style="bold bright_green")
        motion.add_column("BOUNDED PHYSICAL ACTION", style="bright_white")
        for index, step in enumerate(plan, start=1):
            if "pause" in step:
                description = f"Hold current outputs for {step['pause']} seconds"
            else:
                joint = str(step["joint"])
                description = (
                    f"{JOINT_NAMES[joint]}: {step['start']} -> {step['target']} deg "
                    f"at {self.rates[joint]:g} deg/s"
                )
            motion.add_row(str(index), description)
        self.console.print(motion)
        if not self.execute:
            self.console.print("[bold yellow]DRY RUN  //  zero motor commands sent[/]")
            return 0
        with self.console.status(
            "[bold bright_green]RASPBERRY PI -- executing bounded PWM trajectory[/]",
            spinner="material",
        ):
            self._actuate(self._command_for_plan(plan))
        for step in plan:
            if "joint" in step:
                positions[str(step["joint"])] = int(step["target"])
        self._save_positions(positions)
        self.console.print(
            Panel(
                f"{ACTION_NAMES[action].upper()} COMPLETE  //  {len(plan)} planned stages executed",
                title="[bold black on bright_green] PHYSICAL EXECUTION COMPLETE ",
                border_style="bright_green",
            )
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Charlie live voice-to-motion terminal")
    parser.add_argument("--text", help="Use a transcript instead of the microphone")
    parser.add_argument("--execute", action="store_true", help="Send guarded commands to the Pi")
    parser.add_argument("--continuous", action="store_true", help="Keep listening after each cycle")
    args = parser.parse_args()
    if args.text and args.continuous:
        parser.error("--text and --continuous cannot be combined")
    load_dotenv(ROOT / ".env", override=False)
    demo = CharlieVoiceMotionDemo(execute=args.execute, text=args.text)
    demo.banner()
    cycle = 1
    while True:
        try:
            status = demo.run_once(cycle)
        except KeyboardInterrupt:
            demo.console.print("\n[bold yellow]CHARLIE LISTENER STOPPED[/]")
            return 130
        except Exception as exc:
            demo.console.print(
                Panel(
                    f"{type(exc).__name__}: {exc}",
                    title="[bold white on red] LIVE PIPELINE ERROR ",
                    subtitle="No motor command completed",
                    border_style="red",
                )
            )
            status = 1
        if not args.continuous:
            return status
        cycle += 1
        demo.console.print("[dim]Listener remains armed. Press Ctrl+C to exit.[/]\n")


if __name__ == "__main__":
    raise SystemExit(main())
