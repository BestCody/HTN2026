"""Small CLI for routing, classification evaluation, and an offline demo."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .backends import (
    DEFAULT_MAX_NEW_TOKENS,
    MINILM_MODEL,
    SMOLLM_MODEL,
    SentenceTransformerEncoder,
    TransformersGenerator,
)
from .baseten_status import inspect_baseten_deployments
from .calibration import (
    apply_servo_calibration_file,
    write_json_atomic,
    write_servo_calibration_template,
)
from .cloud import load_runtime_environment
from .demo import run_demo
from .edge_components import AlsaCommandRecorder, LanCameraSource, OpenCVCameraSource
from .edge_demo import run_edge_demo
from .episode_dataset import (
    create_camera_calibration_template,
    initialize_episode_dataset,
)
from .evaluation import evaluate_routing, load_samples
from .experts import ExpertRegistry
from .human_interaction import (
    HumanAwarePhysicalSession,
    PlanProposal,
    SpokenEmergencyStopMonitor,
)
from .physical import ClarificationResult, EmergencyStopResult, RobotState
from .pi import PiRuntimeProfile, inspect_host
from .production import (
    build_physical_session,
    inspect_physical_runtime,
    load_physical_runtime_config,
    prepare_physical_workspace,
)
from .robot_config import load_robot_model
from .robot_sources import inspect_3mf
from .routing import (
    DEFAULT_MIN_MARGIN,
    EmbeddingRouter,
    HybridRouter,
    PromptExample,
    PromptRouter,
    PrototypeEmbeddingRouter,
)
from .simulation_training import (
    create_simulation_training_template,
    simulation_training_preflight,
)


def _router(args, registry):
    options = dict(device=args.device, revision=args.revision, local_files_only=args.offline)
    if args.router in ("embedding", "prototype"):
        router_type = EmbeddingRouter if args.router == "embedding" else PrototypeEmbeddingRouter
        return router_type(
            registry,
            SentenceTransformerEncoder(args.model or MINILM_MODEL, **options),
            style=args.style,
        )
    examples = []
    if args.examples:
        candidate_ids = {expert.id for expert in registry.snapshot()}
        examples = [
            PromptExample(sample.instruction, sample.expert_id)
            for sample in load_samples(args.examples)
            if sample.expert_id in candidate_ids
        ]
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


def _json_mapping(path: Path, name: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _robot_state(path: Path | None) -> RobotState | None:
    if path is None:
        return None
    value = _json_mapping(path, "robot state")
    joints = value.get("joint_positions")
    widths = value.get("gripper_widths_m", {})
    if not isinstance(joints, dict) or not isinstance(widths, dict):
        raise TypeError("robot state requires joint_positions and gripper_widths_m objects")
    return RobotState(
        {str(arm): tuple(positions) for arm, positions in joints.items()},
        {str(arm): width for arm, width in widths.items()},
        value.get("observed_at", 0.0),
        value.get("source", "observed"),
    )


def _camera_device(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


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
    calibration_template = commands.add_parser(
        "servo-calibration-template",
        help="Create a strict measurement template for one installed physical arm",
    )
    calibration_template.add_argument("model", type=Path)
    calibration_template.add_argument("output", type=Path)
    calibration_template.add_argument("--arm", choices=("left", "right"), default="left")
    calibration_apply = commands.add_parser(
        "servo-calibration-apply",
        help="Validate and atomically apply a completed servo calibration record",
    )
    calibration_apply.add_argument("model", type=Path)
    calibration_apply.add_argument("calibration", type=Path)
    calibration_apply.add_argument("--output", required=True, type=Path)
    calibration_apply.add_argument("--packaged-output", type=Path)
    camera_template = commands.add_parser(
        "camera-calibration-template",
        help="Create an unset intrinsic and camera-to-base calibration record",
    )
    camera_template.add_argument("model", type=Path)
    camera_template.add_argument("output", type=Path)
    camera_template.add_argument("--camera-id", default="co6-usb")
    dataset_init = commands.add_parser(
        "episode-dataset-init",
        help="Initialize a training dataset pinned to robot, servo, and camera calibration",
    )
    dataset_init.add_argument("root", type=Path)
    dataset_init.add_argument("--model", required=True, type=Path)
    dataset_init.add_argument("--servo-calibration", required=True, type=Path)
    dataset_init.add_argument("--camera-calibration", required=True, type=Path)
    simulation_template = commands.add_parser(
        "simulation-training-template",
        help="Create a dynamics template pinned to the current CAD-derived MuJoCo model",
    )
    simulation_template.add_argument("model", type=Path)
    simulation_template.add_argument("mujoco", type=Path)
    simulation_template.add_argument("output", type=Path)
    simulation_preflight = commands.add_parser(
        "simulation-training-preflight",
        help="Report missing measurements before physics rollout generation or training",
    )
    simulation_preflight.add_argument("model", type=Path)
    simulation_preflight.add_argument("mujoco", type=Path)
    simulation_preflight.add_argument("--servo-calibration", required=True, type=Path)
    simulation_preflight.add_argument("--camera-calibration", required=True, type=Path)
    simulation_preflight.add_argument("--simulation-config", required=True, type=Path)
    physical_preflight = commands.add_parser(
        "physical-preflight",
        help="Check CAD, calibration, camera dependency, and Baseten endpoint readiness",
    )
    physical_preflight.add_argument(
        "--config", type=Path, default=Path("config/pi4_runtime.json")
    )
    baseten_status = commands.add_parser(
        "baseten-status",
        help="Inspect configured Baseten IDs and read live deployment status without inference",
    )
    baseten_status.add_argument(
        "--config", type=Path, default=Path("config/pi4_runtime.json")
    )
    baseten_status.add_argument(
        "--offline", action="store_true", help="Report configuration without calling Baseten"
    )
    physical_run = commands.add_parser(
        "physical-run",
        help="Run the live camera, specialist routing, prediction, control, and feedback path",
    )
    physical_run.add_argument(
        "--config", type=Path, default=Path("config/pi4_runtime.json")
    )
    input_group = physical_run.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--instruction")
    input_group.add_argument("--audio", type=Path)
    input_group.add_argument(
        "--record-seconds",
        type=int,
        help="Record a spoken WAV command from the Pi ALSA microphone",
    )
    physical_run.add_argument("--user-id", default="demo-user")
    physical_run.add_argument("--camera", help="USB camera index or device path")
    physical_run.add_argument(
        "--camera-url",
        help="Authenticated laptop camera endpoint, for example http://HOST:8765/v1/camera",
    )
    physical_run.add_argument("--camera-id")
    physical_run.add_argument("--audio-device", help="Optional ALSA capture device name")
    physical_run.add_argument("--workspace", type=Path, help="JSON workspace context")
    physical_run.add_argument("--robot-state", type=Path, help="Measured robot-state JSON")
    physical_run.add_argument("--execute", action="store_true")
    physical_run.add_argument("--speak", action="store_true")
    for command in ("route", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--experts", required=True, type=Path, help="Expert JSON manifest")
        sub.add_argument(
            "--interface",
            help=(
                "Restrict routing to experts that declare this compatibility interface; "
                "selection inside that pool remains semantic"
            ),
        )
        sub.add_argument(
            "--router",
            choices=("prototype", "embedding", "prompt", "hybrid"),
            default="prototype",
        )
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
    if args.command not in (
        "demo",
        "physical-demo",
        "pi-check",
        "robot-model-check",
        "servo-calibration-template",
        "servo-calibration-apply",
        "camera-calibration-template",
        "episode-dataset-init",
        "simulation-training-template",
        "simulation-training-preflight",
        "physical-preflight",
        "baseten-status",
        "physical-run",
    ):
        if args.router != "hybrid" and (args.fallback_model or args.fallback_revision):
            parser.error("--fallback-model and --fallback-revision require --router hybrid")
        if args.router != "hybrid" and args.min_margin is not None:
            parser.error("--min-margin requires --router hybrid")
        if args.router in ("embedding", "prototype") and args.examples:
            parser.error("--examples is only used by prompt or hybrid routing")
        if args.router in ("embedding", "prototype") and args.max_new_tokens is not None:
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
            print_package = None
            if model.print_file is not None:
                print_path = (Path(args.model).parent / model.print_file).resolve()
                print_package = inspect_3mf(print_path).as_dict()
            output = {
                "model_id": model.model_id,
                "source_file": model.source_file,
                "source_sha256": model.source_sha256,
                "servo_models": model.servo_models,
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
                "print_package": print_package,
                "motion_ready": model.motion_ready,
                "bimanual_motion_ready": model.bimanual_motion_ready,
                "blockers": model.blockers,
                "readiness_issues": model.readiness_issues,
                "bimanual_readiness_issues": model.bimanual_readiness_issues,
            }
        elif args.command == "servo-calibration-template":
            write_servo_calibration_template(args.model, args.arm, args.output)
            output = {
                "status": "template_created",
                "arm": args.arm,
                "output": str(args.output.resolve()),
                "hardware_commanded": False,
            }
        elif args.command == "servo-calibration-apply":
            updated = apply_servo_calibration_file(
                args.model,
                args.calibration,
                args.output,
                args.packaged_output,
            )
            model = load_robot_model(args.output)
            output = {
                "status": "calibration_applied",
                "robot_model_id": updated["model_id"],
                "output": str(args.output.resolve()),
                "packaged_output": (
                    str(args.packaged_output.resolve()) if args.packaged_output else None
                ),
                "motion_ready": model.motion_ready,
                "remaining_readiness_issues": model.readiness_issues,
            }
        elif args.command == "camera-calibration-template":
            model_value = _json_mapping(args.model, "robot model")
            write_json_atomic(
                args.output,
                create_camera_calibration_template(model_value, args.camera_id),
            )
            output = {
                "status": "template_created",
                "camera_id": args.camera_id,
                "output": str(args.output.resolve()),
                "hardware_commanded": False,
            }
        elif args.command == "episode-dataset-init":
            created_at = datetime.now(timezone.utc).isoformat()
            manifest = initialize_episode_dataset(
                args.root,
                args.model,
                args.servo_calibration,
                args.camera_calibration,
                created_at=created_at,
            )
            output = {
                "status": "dataset_initialized",
                "root": str(args.root.resolve()),
                "manifest": manifest,
            }
        elif args.command == "simulation-training-template":
            model_value = _json_mapping(args.model, "robot model")
            write_json_atomic(
                args.output,
                create_simulation_training_template(model_value, args.mujoco),
            )
            output = {
                "status": "template_created",
                "output": str(args.output.resolve()),
                "training_started": False,
            }
        elif args.command == "simulation-training-preflight":
            output = simulation_training_preflight(
                args.model,
                args.mujoco,
                args.servo_calibration,
                args.camera_calibration,
                args.simulation_config,
            )
        elif args.command == "physical-preflight":
            output = inspect_physical_runtime(load_physical_runtime_config(args.config))
        elif args.command == "baseten-status":
            output = inspect_baseten_deployments(
                load_physical_runtime_config(args.config), live=not args.offline
            )
        elif args.command == "physical-run":
            load_runtime_environment()
            config = load_physical_runtime_config(args.config)
            camera_id = args.camera_id or config.camera.camera_id
            workspace = (
                _json_mapping(args.workspace, "workspace") if args.workspace is not None else {}
            )
            workspace = prepare_physical_workspace(
                config,
                workspace,
                camera_id=camera_id,
            )
            state = _robot_state(args.robot_state)
            if args.execute and state is None:
                parser.error("--execute requires --robot-state with a timestamped observation")
            audio = None
            if args.audio is not None:
                audio_path = args.audio.resolve()
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
                audio = audio_path.read_bytes()
            elif args.record_seconds is not None:
                audio = AlsaCommandRecorder(device=args.audio_device).record(
                    args.record_seconds
                )
            camera_url = args.camera_url
            if camera_url is None and config.camera.transport == "lan_http":
                camera_url = os.environ.get(config.camera.url_env or "")
            if camera_url:
                token = os.environ.get(config.camera.token_env or "MOIRA_LAN_TOKEN")
                if not token:
                    required = config.camera.token_env or "MOIRA_LAN_TOKEN"
                    raise RuntimeError(f"Set {required} before using the LAN camera")
                camera = LanCameraSource(
                    camera_url,
                    token=token,
                    camera_id=camera_id,
                )
            else:
                if config.camera.transport == "lan_http":
                    required = config.camera.url_env or "MOIRA_CAMERA_URL"
                    raise RuntimeError(f"Set {required} before using the LAN camera")
                camera_device = args.camera or os.environ.get(
                    config.camera.device_env or ""
                )
                if not camera_device:
                    raise RuntimeError(
                        f"Set {config.camera.device_env} to the verified CO6 device "
                        "or pass --camera explicitly"
                    )
                camera = OpenCVCameraSource(
                    _camera_device(camera_device),
                    camera_id=camera_id,
                )
            with build_physical_session(config, (camera,)) as session:
                if args.execute:
                    conversation = HumanAwarePhysicalSession(
                        session,
                        user_id=args.user_id,
                        workspace=workspace,
                    )
                    interaction = conversation.plan(
                        instruction=args.instruction,
                        audio=audio,
                        robot_state=state,
                        speak=args.speak,
                    )
                    while isinstance(interaction, (ClarificationResult, PlanProposal)):
                        prompt = (
                            interaction.question
                            if isinstance(interaction, ClarificationResult)
                            else interaction.prompt
                        )
                        response = input(prompt + "\nResponse: ")
                        if isinstance(interaction, ClarificationResult):
                            interaction = conversation.plan(
                                instruction=response,
                                robot_state=state,
                                speak=args.speak,
                            )
                        else:
                            stop_monitor = None
                            if conversation.is_confirmation_response(response):
                                stop_recorder = AlsaCommandRecorder(device=args.audio_device)
                                stop_monitor = SpokenEmergencyStopMonitor(
                                    lambda recorder=stop_recorder: recorder.record(1),
                                    lambda payload: session.system.transcribe_audio(
                                        payload, allow_empty=True
                                    ),
                                    lambda: conversation.emergency_stop(speak=False),
                                )
                                stop_monitor.start()
                            try:
                                interaction = conversation.respond(
                                    confirmation_id=interaction.confirmation_id,
                                    instruction=response,
                                    robot_state=state,
                                    speak=args.speak,
                                )
                            finally:
                                if stop_monitor is not None:
                                    stop_monitor.close()
                    result = interaction
                else:
                    result = session.run(
                        user_id=args.user_id,
                        instruction=args.instruction,
                        audio=audio,
                        workspace=workspace,
                        robot_state=state,
                        execute=False,
                        speak=args.speak,
                    )
            if isinstance(result, EmergencyStopResult):
                output = {
                    "status": "emergency_stop",
                    "transcript": result.transcript,
                    "response": result.response_text,
                    "stop_issues": list(result.stop_issues),
                    "journal": str(config.journal_path),
                }
            elif isinstance(result, ClarificationResult):
                output = {
                    "status": "clarification",
                    "transcript": result.transcript,
                    "question": result.question,
                    "routed_components": [item.component_id for item in result.routing],
                    "journal": str(config.journal_path),
                }
            else:
                output = {
                    "status": result.outcome.status if result.outcome else "unknown",
                    "transcript": result.intent.transcript,
                    "candidates": [item.id for item in result.candidates],
                    "simulations": [asdict(item) for item in result.simulations],
                    "selected_plan": result.plan.candidate.id,
                    "routed_components": [item.component_id for item in result.routing],
                    "executed": result.control.executed,
                    "camera_verified": result.world_after is not None,
                    "outcome": asdict(result.outcome) if result.outcome else None,
                    "prediction_error": (
                        asdict(result.prediction_error) if result.prediction_error else None
                    ),
                    "response": result.response_text,
                    "journal": str(config.journal_path),
                }
        else:
            registry = ExpertRegistry.from_json(args.experts)
            if args.interface:
                registry = registry.for_interface(args.interface)
            router = _router(args, registry)
            if args.command == "route":
                output = asdict(router.route(args.instruction))
            else:
                samples = load_samples(args.samples)
                labels = [e.id for e in registry.snapshot()]
                excluded_samples = sum(sample.expert_id not in labels for sample in samples)
                if args.interface:
                    samples = [sample for sample in samples if sample.expert_id in labels]
                    if not samples:
                        raise ValueError(
                            f"No evaluation samples belong to interface: {args.interface}"
                        )
                output = evaluate_routing(
                    router,
                    samples,
                    labels,
                )
                if args.interface:
                    output["interface"] = args.interface
                    output["excluded_samples"] = excluded_samples
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
