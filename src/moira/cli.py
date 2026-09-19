"""Small CLI for routing, classification evaluation, and an offline demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .backends import (
    DEFAULT_MAX_NEW_TOKENS,
    MINILM_MODEL,
    SMOLLM_MODEL,
    SentenceTransformerEncoder,
    TransformersGenerator,
)
from .demo import run_demo
from .edge_demo import run_edge_demo
from .evaluation import evaluate_routing, load_samples
from .experts import ExpertRegistry
from .pi import PiRuntimeProfile, inspect_host
from .robot_config import load_robot_model
from .routing import DEFAULT_MIN_MARGIN, EmbeddingRouter, HybridRouter, PromptExample, PromptRouter


def _router(args, registry):
    options = dict(device=args.device, revision=args.revision, local_files_only=args.offline)
    if args.router == "embedding":
        return EmbeddingRouter(
            registry,
            SentenceTransformerEncoder(args.model or MINILM_MODEL, **options),
            style=args.style,
        )
    examples = []
    if args.examples:
        examples = [PromptExample(s.instruction, s.expert_id) for s in load_samples(args.examples)]
    if args.router == "hybrid":
        return HybridRouter(
            registry,
            SentenceTransformerEncoder(args.model or MINILM_MODEL, **options),
            TransformersGenerator(
                args.fallback_model or SMOLLM_MODEL,
                device=args.device,
                revision=args.fallback_revision,
                local_files_only=args.offline,
                max_new_tokens=(
                    args.max_new_tokens
                    if args.max_new_tokens is not None
                    else DEFAULT_MAX_NEW_TOKENS
                ),
            ),
            style=args.style,
            examples=examples,
            min_margin=(args.min_margin if args.min_margin is not None else DEFAULT_MIN_MARGIN),
        )
    return PromptRouter(
        registry,
        TransformersGenerator(
            args.model or SMOLLM_MODEL,
            max_new_tokens=(
                args.max_new_tokens if args.max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS
            ),
            **options,
        ),
        style=args.style,
        examples=examples,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MoIRA modular robot policy routing")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Offline toy pipeline; does not use pretrained routing models")
    commands.add_parser(
        "physical-demo",
        help="Offline Pi-oriented layered demo with perception, simulation, two arms, and memory",
    )
    pi_check = commands.add_parser("pi-check", help="Inspect Raspberry Pi 4B runtime capacity")
    pi_check.add_argument(
        "--strict", action="store_true", help="Fail unless this is a Pi 4 Model B"
    )
    robot_check = commands.add_parser(
        "robot-model-check", help="Validate arm metadata and report physical-motion readiness"
    )
    robot_check.add_argument("model", type=Path)
    robot_check.add_argument("--verify-source", action="store_true")
    robot_check.add_argument("--require-motion-ready", action="store_true")
    for command in ("route", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--experts", required=True, type=Path, help="Expert JSON manifest")
        sub.add_argument("--router", choices=("hybrid", "embedding", "prompt"), default="hybrid")
        sub.add_argument("--style", choices=("simple", "abstract"), default="simple")
        sub.add_argument("--model", help="Primary model; the embedding model in hybrid mode")
        sub.add_argument("--fallback-model", help="Language model used by the hybrid router")
        sub.add_argument("--fallback-revision", help="Hybrid fallback model revision")
        sub.add_argument(
            "--min-margin",
            type=float,
            default=None,
            help="Hybrid mode: minimum cosine gap to skip LM disambiguation (default: 0.05)",
        )
        sub.add_argument(
            "--max-new-tokens",
            type=int,
            default=None,
            help="Prompt/hybrid maximum generated tokens (default: 64)",
        )
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--revision", help="Hugging Face model revision, e.g. a commit hash")
        sub.add_argument("--offline", action="store_true", help="Use cached model files only")
        sub.add_argument("--examples", type=Path, help="Few-shot prompt examples as JSONL")
        if command == "route":
            sub.add_argument("instruction")
        else:
            sub.add_argument("--samples", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command not in ("demo", "physical-demo", "pi-check", "robot-model-check"):
        if args.router != "hybrid" and (args.fallback_model or args.fallback_revision):
            parser.error("--fallback-model and --fallback-revision require --router hybrid")
        if args.router != "hybrid" and args.min_margin is not None:
            parser.error("--min-margin requires --router hybrid")
        if args.router == "embedding" and args.examples:
            parser.error("--examples is only used by prompt or hybrid routing")
        if args.router == "embedding" and args.max_new_tokens is not None:
            parser.error("--max-new-tokens is only used by prompt or hybrid routing")
    try:
        if args.command == "demo":
            output = {
                "demo": "toy encoder and policies; not a robotics benchmark",
                **asdict(run_demo()),
            }
        elif args.command == "physical-demo":
            output = {
                "demo": "layered physical-AI fixture; no real hardware is commanded",
                **asdict(run_edge_demo()),
            }
        elif args.command == "pi-check":
            host = inspect_host()
            if args.strict:
                host.require_pi4()
            output = {
                "host": asdict(host),
                "is_pi4_model_b": host.is_pi4_model_b,
                "recommended_profile": asdict(PiRuntimeProfile.for_pi4(host.total_memory_mb)),
            }
        elif args.command == "robot-model-check":
            model = load_robot_model(args.model, verify_source=args.verify_source)
            if args.require_motion_ready:
                model.require_motion_ready()
            output = {
                "model_id": model.model_id,
                "source_file": model.source_file,
                "source_sha256": model.source_sha256,
                "servo_model": model.servo_model,
                "servo_controller": "pca9685",
                "pca9685_i2c_address": model.servo_controller.i2c_address,
                "pca9685_pwm_frequency_hz": model.servo_controller.pwm_frequency_hz,
                "servo_supply_voltage": model.servo_controller.servo_supply_voltage,
                "servo_supply_current_a": model.servo_controller.servo_supply_current_a,
                "physical_arms": {
                    arm: {
                        "physical_id": installation.physical_id,
                        "installed": installation.installed,
                        "channels": {
                            joint: calibration.channel
                            for joint, calibration in model.servo_controller.actuators[arm].items()
                        },
                    }
                    for arm, installation in model.servo_controller.arm_installations.items()
                },
                "payload_limit_kg": model.payload_limit_kg,
                "kinematic_joints": [joint.name for joint in model.kinematic_joints],
                "gripper_joint": model.gripper_joint.name,
                "motion_ready": model.motion_ready,
                "bimanual_motion_ready": model.bimanual_motion_ready,
                "blockers": model.blockers,
                "readiness_issues": model.readiness_issues,
                "bimanual_readiness_issues": model.bimanual_readiness_issues,
            }
        else:
            registry = ExpertRegistry.from_json(args.experts)
            router = _router(args, registry)
            if args.command == "route":
                output = asdict(router.route(args.instruction))
            else:
                output = evaluate_routing(
                    router,
                    load_samples(args.samples),
                    [e.id for e in registry.snapshot()],
                )
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except (
        ValueError,
        TypeError,
        KeyError,
        LookupError,
        OSError,
        ImportError,
        RuntimeError,
        MemoryError,
        ConnectionError,
    ) as exc:
        parser.exit(2, f"moira: {exc}\n")
    return 0
