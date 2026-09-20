"""Run an intentionally narrow, local-only PCA9685 calibration movement."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from moira.guarded_calibration import load_guarded_calibration_envelope  # noqa: E402


class RawPCA9685:
    def __init__(self, envelope):
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
        except ImportError as exc:
            raise RuntimeError(
                "Install Raspberry Pi hardware dependencies with 'pip install .[hardware]'"
            ) from exc
        self.envelope = envelope
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(
            self.i2c,
            address=envelope.i2c_address,
            reference_clock_speed=envelope.reference_clock_hz,
        )
        self.pca.frequency = envelope.pwm_frequency_hz

    def tick(self, channel: int, tick: int) -> None:
        if channel in self.envelope.disabled_channels:
            raise ValueError(f"channel {channel} is disabled")
        if not 0 < tick < 4096:
            raise ValueError("PCA9685 tick must be in [1, 4095]")
        self.pca.channels[channel].duty_cycle = tick << 4

    def stop(self) -> None:
        channels = {
            servo.channel for servo in self.envelope.servos.values()
        } | set(self.envelope.disabled_channels)
        for channel in channels:
            self.pca.channels[channel].duty_cycle = 0

    def close(self) -> None:
        self.stop()
        self.pca.deinit()
        deinit = getattr(self.i2c, "deinit", None)
        if callable(deinit):
            deinit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "arm1_guarded_envelope.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("stop")
    exercise = subparsers.add_parser("exercise")
    exercise.add_argument("--joint", required=True)
    exercise.add_argument("--start", required=True, type=int)
    exercise.add_argument("--target", required=True, type=int)
    exercise.add_argument("--rate", required=True, type=float)
    exercise.add_argument("--settle-seconds", type=float, default=0.75)
    exercise.add_argument(
        "--armed",
        action="store_true",
        help="Required acknowledgement that the joint is supported and the workspace is clear",
    )
    sequence = subparsers.add_parser("sequence")
    sequence.add_argument(
        "--step",
        action="append",
        required=True,
        help="Movement JOINT:START:TARGET:RATE or pause:SECONDS; repeat in execution order",
    )
    sequence.add_argument("--settle-seconds", type=float, default=0.75)
    sequence.add_argument("--armed", action="store_true")
    args = parser.parse_args()
    envelope = load_guarded_calibration_envelope(args.config)
    if args.command in ("exercise", "sequence") and not args.armed:
        parser.error(f"{args.command} requires --armed after clearing the workspace")
    if args.command in ("exercise", "sequence") and not 0.0 <= args.settle_seconds <= 10.0:
        parser.error(f"{args.command} --settle-seconds must be in [0, 10]")
    ramp = (
        envelope.ramp(args.joint, args.start, args.target, args.rate)
        if args.command == "exercise"
        else ()
    )
    sequence_steps = []
    if args.command == "sequence":
        for raw in args.step:
            parts = raw.split(":")
            if len(parts) == 2 and parts[0] == "pause":
                seconds = float(parts[1])
                if not 0.0 <= seconds <= 10.0:
                    parser.error("sequence pause must be in [0, 10] seconds")
                sequence_steps.append(("pause", seconds))
                continue
            if len(parts) != 4:
                parser.error("sequence step must be JOINT:START:TARGET:RATE or pause:SECONDS")
            joint, start, target, rate = parts
            sequence_steps.append(
                ("move", joint, envelope.ramp(joint, int(start), int(target), float(rate)))
            )
    device = RawPCA9685(envelope)
    stopping = False

    def stop_handler(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True
        device.stop()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    started = monotonic()
    try:
        # The unused output is disabled before any active servo can be written.
        device.stop()
        if args.command in ("preflight", "stop"):
            print(
                json.dumps(
                    {
                        "status": "passed" if args.command == "preflight" else "stopped",
                        "motion": False,
                        "disabled_channels": sorted(
                            {
                                servo.channel for servo in envelope.servos.values()
                            }
                            | set(envelope.disabled_channels)
                        ),
                    }
                )
            )
            return 0
        active_holds = {}

        def run_ramp(joint, values, rate):
            servo = envelope.servos[joint]
            interval = 1.0 / rate
            ramp_started = monotonic()
            for index, (command_deg, tick) in enumerate(values):
                if stopping:
                    raise RuntimeError("guarded movement was stopped")
                deadline = ramp_started + index * interval
                remaining = deadline - monotonic()
                if remaining > 0:
                    sleep(remaining)
                for held_joint, held_tick in active_holds.items():
                    if held_joint != joint:
                        device.tick(envelope.servos[held_joint].channel, held_tick)
                device.tick(servo.channel, tick)
                print(
                    json.dumps(
                        {
                            "joint": joint,
                            "channel": servo.channel,
                            "command_deg": command_deg,
                            "tick": tick,
                            "elapsed_s": round(monotonic() - started, 3),
                        }
                    ),
                    flush=True,
                )
            if values:
                active_holds[joint] = values[-1][1]

        if args.command == "exercise":
            run_ramp(args.joint, ramp, args.rate)
        else:
            for step_index, step in enumerate(sequence_steps, start=1):
                if step[0] == "pause":
                    seconds = step[1]
                    pause_deadline = monotonic() + seconds
                    print(json.dumps({"step": step_index, "pause_seconds": seconds}), flush=True)
                    while monotonic() < pause_deadline:
                        for held_joint, held_tick in active_holds.items():
                            device.tick(envelope.servos[held_joint].channel, held_tick)
                        sleep(min(0.05, max(0.0, pause_deadline - monotonic())))
                else:
                    _, joint, values = step
                    rate = float(args.step[step_index - 1].split(":")[3])
                    run_ramp(joint, values, rate)
        # Settle while keeping each completed joint at its final target.
        settle_deadline = monotonic() + args.settle_seconds
        while monotonic() < settle_deadline:
            for held_joint, held_tick in active_holds.items():
                device.tick(envelope.servos[held_joint].channel, held_tick)
            sleep(min(0.05, max(0.0, settle_deadline - monotonic())))
        print(
            json.dumps(
                {
                    "status": "completed",
                    "command": args.command,
                    "joint": getattr(args, "joint", None),
                    "start_command_deg": getattr(args, "start", None),
                    "target_command_deg": getattr(args, "target", None),
                    "commanded_rate_deg_s": getattr(args, "rate", None),
                    "sequence_steps": len(sequence_steps),
                    "elapsed_s": round(monotonic() - started, 3),
                    "actual_joint_velocity_measured": False,
                }
            )
        )
        return 0
    finally:
        device.close()


if __name__ == "__main__":
    raise SystemExit(main())
