# MoIRA Physical AI Router

This repository is an OpenRouter-style gateway for physical AI. A Raspberry Pi
4B captures sensors and owns the local safety/control boundary, while a remote
router selects exact specialized components for perception, scene-grounded
voice NLP, 6-DoF grasping, task planning, robot-specific manipulation policies,
structured world prediction, outcome verification, and speech. Baseten can host
those models independently. Personal memory, inverse kinematics, trajectories,
collision checks, tactile reflexes, hard safety, feedback, and bimanual hardware
control remain on the Pi.

The production component path does not substitute a different model after a
router or inference failure. It stops before motor execution and reports the
error.

A Python implementation of [MoIRA: Modular Instruction Routing Architecture for
Multi-Task Robotics](https://arxiv.org/html/2507.01843v2) (arXiv:2507.01843v2).
This repository implements the routing and serving method and provides a
specialist-training interface. It does **not** bundle trained robot policies or
claim to reproduce the paper's benchmark results.

The core package has no third-party dependencies. Pretrained routing and LoRA
training are optional installations.

## GPU training environment

Windows training uses a separate `.venv-training` environment so CUDA and
robot-learning dependencies do not enlarge or destabilize the Raspberry Pi
runtime. Create or repair it from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_training_env.ps1
```

The script installs the verified CUDA 12.8 PyTorch and torchvision builds,
OpenCV, MuJoCo, Gymnasium, LeRobot with its dataset stack, and a compatible
FFmpeg 8 shared build for TorchCodec. It finishes by running real CUDA matrix
multiplication, OpenCV image encoding, MuJoCo stepping, a Gymnasium environment,
LeRobot ACT/CLI loading, TorchCodec loading, and a MoIRA import. Run the verifier
directly with:

```powershell
.\.venv-training\Scripts\python.exe tools\verify_training_env.py
```

## Raspberry Pi quick start

On 64-bit Raspberry Pi OS with Python 3.10+, run from the repository root:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[hardware]"
./.venv/bin/moira pi-check --strict
./.venv/bin/moira physical-demo
```

For development on Windows PowerShell, use `.\.venv\Scripts\python.exe` in
place of `./.venv/bin/python`; omit `--strict` on a non-Pi host.
`physical-demo` exercises camera objects, persistent personal context, a
2.5-second structured prediction horizon, grasp generation, single-arm and
bimanual policies, four world-model specialists, reward/risk scoring, local IK,
validated action chunks, tactile sensing, execution feedback, and voice output
without commanding hardware. See the
[Pi 4B + Baseten architecture](docs/pi4-baseten-architecture.md) for production
wiring and deployment contracts.

The active SolidWorks arm is tracked in
[`robot_models/four_dof_desktop_arm/model.json`](robot_models/four_dof_desktop_arm/model.json).
It uses MG996R servos at the base and shoulder and SG90 servos at the elbow and
end effector. The assembly, every CAD dependency, and `Robotic+Arm.3mf` are
hash-pinned. The 3MF contributes millimetre print geometry but contains
print-plate placement rather than assembled transforms. The configuration is
the single source for link geometry, coordinate frame, bimanual mounting,
gripper limits, joint speed, control frequency, clearance, payload, safety
thresholds, and controller timing. Values from the retired robot were removed;
unknown values remain `null` and prevent robot-backed components from being
constructed. Run `moira
robot-model-check robot_models/four_dof_desktop_arm/model.json
--verify-source` to inspect its readiness. Exact Fusion export and calibration
steps are documented in
[`robot_models/four_dof_desktop_arm/README.md`](robot_models/four_dof_desktop_arm/README.md).
The hardware path uses one PCA9685 and the confirmed external 6 V/10 A supply.
The new arm's installation state, channel map, pulse endpoints, and SG90 voltage
compatibility are deliberately unconfirmed. These values live in the robot
configuration rather than driver code. Physical execution and dual-arm startup
remain blocked until the corresponding installation records are calibrated.

The older `demo` routes a toy instruction to a policy and completes a three-step
line world. Its manually defined encoder is explicitly a plumbing fixture, not
MiniLM and not a robot benchmark.

## Local Baseten credentials

Copy `.env.example` to `.env` and place the Baseten key in the ignored local
file:

```powershell
Copy-Item .env.example .env
```

```dotenv
BASETEN_API_KEY=your-key-here
```

Cloud clients load `.env` from the current working directory. An existing
`BASETEN_API_KEY` process environment variable takes precedence. Set
`MOIRA_ENV_FILE` when launching outside the repository and the secret file is
stored elsewhere. Never commit `.env`.

### Deploy the speech-to-text specialist

The repository includes a Whisper Large V3 Turbo Truss fixed to the Baseten
`L4:4x16` instance that is available to this workspace. From the repository
root:

```powershell
py -3.12 -m venv .venv-deploy
.\.venv-deploy\Scripts\python.exe -m pip install --upgrade pip truss==0.18.30
.\.venv-deploy\Scripts\truss.exe login --browser
.\.venv-deploy\Scripts\truss.exe push deploy\baseten_whisper --watch
```

After the development deployment is ready, set `BASETEN_STT_MODEL_ID` in the
ignored `.env` file and test a real command recording:

```powershell
.\.venv\Scripts\python.exe tools\test_baseten_stt.py path\to\command.wav
```

See `deploy/baseten_whisper/README.md` for the endpoint contract and deployment
details.

## General semantic routing with typed compatibility

MoIRA remains the general task-to-specialist router described by the paper.
Every specialist declares short and abstract natural-language descriptions plus
representative routing phrases. The frozen MiniLM router embeds the current
request and ranks each expert by its best metadata-prototype similarity. Adding
or replacing an expert does not require a route-table edit or router training.
`examples/physical_ai_specialists.json` is the current catalog.

`ComponentRegistry` is the robot-side compatibility and authority gate. Exact
capabilities reject components with the wrong request/response schema, runtime,
or safety role before semantic routing. This gate does not decide which
compatible specialist best matches the task. The Baseten Chain receives only
allow-listed compatible IDs and performs the paper-style semantic choice. If a
contract has one provider, there is no choice to infer and that provider is
selected directly.

The layered workflow composes specialists by making several typed routing
requests: speech, perception, planning, manipulation policy, short-horizon
world prediction, reward, and verification. This orchestration is an extension
around MoIRA; the routing rule itself remains architecture-agnostic and frozen.
The full flow is implemented by `PhysicalAI`, while local collision checks,
hard safety, and motor control retain authority on the Pi.

On the authored routing test, the unrestricted 20-expert pool scored 19/20. The
actual typed world-model pool scored 6/6. A raw-instruction policy
pool scored 5/6, while the production-shaped query containing the grounded goal,
planner action, arms, and target scored 6/6. These are small repository
regressions, not a robotics benchmark; run
`tools/evaluate_physical_router.py` after every catalog change.

The physical path uses these exact capability groups:

| Stage | Capabilities |
| --- | --- |
| Scene and language | `perception.scene`, `voice.transcribe`, `voice.ground` |
| Physical grounding | `grasp.pose_6d`, `personal.recall` |
| Task and skill policy | `planning.candidates`, `manipulation.bimanual`, `manipulation.skill.waypoint`, `.pour`, `.insert`, `.open_lid`, `.handover` |
| Local feasibility | `kinematics.inverse`, `motion.trajectory`, `motion.collision_check` |
| Prediction | `dynamics.predict`, `world.rigid_dynamics`, `world.grasp_contact`, `world.bimanual_coordination`, `world.deformable_dynamics`, `world.human_motion` |
| Model-based choice | `reward.task_progress`, `safety.risk`, `planning.select` |
| Contact and control | `tactile.contact`, `tactile.slip`, `tactile.force`, `tactile.grasp_stability`, `control.single_arm`, `control.bimanual` |
| Learning loop | `outcome.verify`, `failure.classify`, `load.estimate`, `feedback.prediction_error`, `feedback.learn`, `personal.record` |

Every candidate policy is converted to a local joint trajectory before remote
prediction. Relevant world models predict typed future object poses, joints,
contact forces, slip, collision probability, success, and uncertainty through
the configured 2–3 second horizon. A reward model scores progress; the Pi-side
safety component applies hard limits. The task planner can select only a
candidate that was generated, validated, predicted, and marked safe. After
execution, predicted success is compared with the observed outcome per world
model so calibration errors can be persisted for retraining.

`src/moira/specialists.py` contains bounded analytic/state-space implementations
for offline validation. Its generic numeric fixtures are not arm calibration.
Production local components use `from_robot_model(...)`, and `PhysicalAI` uses
`from_robot_model(...)`, so every physical limit comes from the validated robot
configuration. Production neural model IDs in
`examples/pi4_components.json` are explicit deployment placeholders and must be
replaced with robot-specific trained endpoints. An unavailable endpoint fails
the request; offline implementations are never used as production substitutes.
The [robot model training audit](docs/training-audit.md) identifies every
component that needs training, pretrained deployment, sensor data, or calibration.
The [arm #1 training playbook](docs/training-playbook.md) defines the initial
voice-conditioned dataset, ACT policy, short-horizon dynamics, and deployment flow.

## Pretrained routers

Install the desired backend, then route an instruction:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[embeddings,prompt]"
.\.venv\Scripts\python.exe -m moira route --experts examples/physical_ai_specialists.json --interface manipulation.policy.v1 "Use both arms to carry the box."
```

The CLI and production Baseten Chain default to the frozen MiniLM prototype
router. For physical AI, `--interface` first limits the pool by a data/schema
contract such as `manipulation.policy.v1` or `world.predict.v1`; MiniLM then
compares the request with each expert's description and representative phrases.
Compatibility filtering and routing prototypes are metadata-driven and do not
encode task-to-expert rules in application code.

The default embedding model is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Expert descriptions are cached, task embeddings are normalized, and the expert
with maximum cosine similarity is selected. The scores are similarities, not
probabilities. Registry additions, removals,
or changed descriptions invalidate the cache automatically. MiniLM inherits its
model's input truncation limit; keep individual descriptions short.

The paper's two exact strategies remain available with `--router embedding`
(description-only cosine argmax) and `--router prompt`. Production's
`--router prototype` extends the embedding strategy with expert-owned example
phrases, retaining frozen weights and add-with-metadata behavior. The repository
also retains an explicit `--router hybrid` research extension. Production has
no automatic fallback: an unavailable or invalid router stops the request.

```powershell
.\.venv\Scripts\python.exe -m moira route --experts examples/directional_experts.json --router embedding "Travel toward the right end of the line."
.\.venv\Scripts\python.exe -m moira route --experts examples/directional_experts.json --router hybrid "Travel toward the right end of the line."
```

For this regression, the baseline selects `left`; hybrid selects `right` through
`hybrid_prompt`. Output includes the original `embedding_expert_id`, embedding
`scores`, and top-two `margin` even when the LM overrides the initial choice.
`hybrid_embedding` indicates no LM call was needed. A single-expert registry has
no runner-up (`margin: null`) and bypasses disambiguation.

The margin threshold is an implementation heuristic, not a calibrated confidence
or a paper hyperparameter. It does not guarantee that larger-margin predictions
are correct. Validate it on representative tasks before choosing a deployment
threshold. `HybridRouter` combines the paper's two routers and is an extension
to the published method.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[prompt]"
.\.venv\Scripts\python.exe -m moira route --experts examples/libero_experts.json --router prompt --examples examples/prompt_examples.jsonl "Open the top drawer."
```

The prompt backend uses
[`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).
It formats the expert pool and optional labeled examples using the tokenizer's
chat template, generates deterministically, and validates the selected index.
Invalid or ambiguous output raises `RoutingError`; no expert is executed.
The complete original prompt is not provided in the paper; this implementation
uses a reconstructed prompt in `src/moira/routing.py`.

All routing modes support `--style simple|abstract`, `--device cpu|cuda`, `--model`,
`--revision`, and `--offline`. In hybrid mode, `--model` and `--revision` configure
MiniLM; `--fallback-model` and `--fallback-revision` configure the LM separately.
`--examples` supplies few-shot examples for prompt mode or hybrid disambiguation.
`--max-new-tokens` bounds prompt generation and defaults to 64. Router-specific
options used with an incompatible mode are rejected instead of being ignored.
The first pretrained call downloads model files;
the prompt model needs substantially more memory than MiniLM. `--offline` only
uses previously cached files. Pin `--revision` to a model commit for repeatable
experiments. CPU is the CLI default.

The example descriptions and labels are newly authored illustrative data, not
the paper's original metadata or a held-out benchmark. Keep few-shot examples
separate from evaluation samples.

## Connect policies

A policy implements `reset(instruction)` and `act(observation)`. Its wrapper owns
image preprocessing, embodiment information, normalization statistics, action
decoding, and any action-chunk queue. `reset` must clear episode state.

```python
from moira import EmbeddingRouter, ExpertRegistry, InMemoryServer, MoIRA


# Supply independently loaded policy wrappers implementing reset() and act().
def build_controller(spatial_policy, goal_policy):
    registry = ExpertRegistry.from_json("examples/libero_experts.json")
    router = EmbeddingRouter(registry)
    server = InMemoryServer({"spatial": spatial_policy, "goal": goal_policy})
    return MoIRA(registry, router, server)


def control_episode(controller, observations):
    with controller.episode("Pick up the bowl to the right of the cup.") as episode:
        print(episode.decision.expert_id)
        for observation in observations:
            action = episode.act(observation)
            yield action
```

Routing uses only instruction and expert text. Visual observations go to the
selected policy. Routing happens once per episode; observations do not trigger
rerouting. The original instruction is passed unchanged to the policy.

Always close an episode with its context manager. `InMemoryServer` permits
parallel episodes when the selected experts use distinct policy objects and
rejects overlap on the same object. `AdapterServer` rejects all overlapping
episodes because its adapters share mutable backbone and episode state. Use
separate adapter servers/backbones for parallel adapter rollouts. Registry
updates are synchronized; an active episode retains the immutable expert record
selected at its boundary. Policy handles returned by either server expire when
their session context closes.

## Adapter serving

`InMemoryServer` retains separate, fully instantiated policies.
`AdapterServer` shares one backbone and supports:

| Mode | Resident adapters | Switching behavior |
| --- | --- | --- |
| `disk` | One | Unload the previous adapter, then load the selected checkpoint |
| `multi` | All loaded so far | Activate a resident adapter; load unseen adapters once |

Use `server.preload(registry.snapshot())` in `multi` mode to load all adapters
before latency measurements, and `server.close()` to unload them afterward.
Preload rejects duplicate IDs and replacement paths, and rolls back adapters it
added if a later load fails. Close attempts every unload and reports any adapter
that remains loaded.
This is ordinary PEFT adapter switching, not S-LoRA/LoRAX optimized kernels;
the paper's reported latency is not promised here. All adapters on one server
must target the same compatible backbone.

Add `adapter_path` to each expert in a JSON manifest. Local paths resolve
relative to the manifest directory. Use `hf://organization/repository` for
Hugging Face adapter IDs. IDs are unique; both `simple` and `abstract`
descriptions are required. An omitted adapter path is valid for routing-only
use and fully instantiated policies.

```python
from moira import AdapterServer, ExpertRegistry, HybridRouter, MoIRA
from moira.adapters import PeftAdapterBackend


def build_adapter_controller(base_model, action_fn, reset_fn, manifest):
    registry = ExpertRegistry.from_json(manifest)
    # action_fn(model, observation, original_instruction) returns an action.
    # reset_fn(model, instruction) clears robot-policy/chunk state.
    backend = PeftAdapterBackend(base_model, action_fn, reset_fn=reset_fn)
    server = AdapterServer(backend, mode="multi")
    server.preload(registry.snapshot())
    return MoIRA(registry, HybridRouter(registry), server)
```

Install `.[adapters]` for the PEFT backend. `base_model` must be a fresh,
unwrapped compatible PyTorch model placed on its inference device. PEFT-format
adapters use inference mode and remain frozen after switching. For models with
native adapter formats, implement the five `AdapterBackend` methods in
`src/moira/serving.py` instead. GR00T and OpenPI checkpoints are not assumed to
be PEFT-compatible, and no native robot-specific wrapper is bundled.

## Train specialists

`train_specialist` trains and saves an independent LoRA adapter. Supply a fresh
backbone for each specialist, its native loss function, device-ready batches,
and correct target module names:

```python
from moira.training import LoraTrainingConfig, train_specialist


def fit_specialist(base_model, dataloader, policy_loss, output_dir, target_modules):
    config = LoraTrainingConfig(
        target_modules=tuple(target_modules),
        steps=5000,
        rank=8,
        alpha=16,
        learning_rate=1e-4,
    )
    return train_specialist(base_model, dataloader, policy_loss, output_dir, config)
```

`policy_loss(adapted_model, batch)` must return a differentiable scalar tensor.
The helper freezes the backbone, trains adapter parameters, clips gradients,
and atomically publishes PEFT weights after a complete save. A reiterable
dataloader restarts at epoch boundaries. Its internal seed does not change the
caller's CPU or CUDA random-number streams.
The seed controls adapter initialization; seed data shuffling and base-model
initialization in your training entry point too. Nonempty output directories
are rejected to avoid overwriting checkpoints.

These rank/optimizer settings are implementation defaults, not recovered paper
hyperparameters. For native VLA training use the backbone's own trainer and
implement its adapter backend; this helper is not a replacement for native
GR00T/OpenPI training pipelines.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m moira evaluate --experts examples/libero_experts.json --samples examples/routing_samples.jsonl
```

This emits JSON containing accuracy, macro-F1, per-class F1, the confusion
matrix, invalid predictions, strategy counts, full routing decisions,
predictions, and elapsed time. Each input JSONL
line contains `instruction` and `expert_id`. Macro-F1 includes every expert in
the supplied manifest; absent classes contribute zero. Invalid LM selections
count as misses. Model-loading errors propagate instead of being counted as
classification errors. Timing includes model loading on the first call.

For execution metrics, `moira.evaluation` provides:

- `active_joint_mse(predicted, target, active_joints)`: rectangular time-by-joint
  arrays with explicit joint selection.
- `rollout(controller, environment, instruction, seed=..., max_steps=...)`:
  a Gymnasium-style reset/step loop with the selected expert, actions, reward,
  success, termination, and truncation recorded.
- `success_rate(results)`: empirical success fraction over rollout results.

The environment wrapper must emit `info["success"]` or you must provide
`success_fn(environment, info)`. Termination and positive reward are not
automatically treated as task success. The caller closes the environment.
Policies must return a single environment-compatible action; unwrap action
chunks in the policy wrapper. Run each required seed/trajectory separately.

## Paper coverage and reproduction limits

The implementation follows section 3's external text routing, simple/abstract
descriptions, and episodic expert execution. It includes all three serving
regimes and the routing F1, joint MSE, and rollout success metrics. The paper
trains GR00T-N1 embodiment specialists for 5,000 steps and π₀ LIBERO specialists
for 30,000 steps. Its evaluation uses robot datasets, checkpoints, and simulator
configurations absent from this repository. See the
[paper's methods](https://arxiv.org/html/2507.01843v2#S3) for the experimental protocol
and the [engineering audit](docs/audit.md) for resolved findings and remaining
deployment boundaries.

Full reproduction additionally requires native VLA loaders/trainers, dataset
splits and preprocessing, action normalization, original expert descriptions
and prompt examples, robot environments, and the authors' exact training
settings. Statistical significance tests and optimized serving kernels are not
included. Synthetic examples and tiny-model integration tests validate this
implementation's behavior; they do not establish robotics performance.

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,prompt,adapters]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

The normal test suite does not download pretrained models. Optional integration
tests train two tiny real adapters, reload and switch them in both serving modes,
and generate text using a locally created tiny language model. Without the ML
extras those integration tests skip. CI runs the core on Windows/Linux and
Python 3.10/3.12, plus ML integration tests on Linux/Python 3.12.

The opt-in E2E suite downloads pinned MiniLM and SmolLM2-1.7B weights, trains two
LoRA specialists, reloads their saved manifest/checkpoints, and runs complete
closed-loop episodes with hybrid and prompt routing in all three serving modes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pretrained_e2e.py --run-e2e -q -o junit_logging=all --junitxml=outputs/e2e.xml
```

It tests both literal descriptions and new paraphrases in a synthetic line
environment. All actions come from trained adapters; routing outputs are not
mocked. The quality assertions require every episode to succeed and deliberately
report semantic misrouting as failures. A separate regression reproduces the raw
embedding misroute and requires hybrid to correct the unchanged instruction,
both with and without few-shot examples. Additional directional/negation wording
is checked separately. Without `--run-e2e`, these tests skip;
the full language model requires a download of several GB and can run slowly
on CPU. Use `-k hybrid` or `-k prompt` to select a router. See
[E2E results](docs/e2e-results.md) for measured outcomes and limitations.

Implementation API references:
[Sentence Transformers](https://github.com/huggingface/sentence-transformers),
[Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating),
[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
