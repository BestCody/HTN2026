# End-to-end validation

Run date: 2026-09-19. Windows, Python 3.12.10, CPU-only PyTorch 2.14.0,
Transformers 5.17.0, Sentence Transformers 5.7.0, PEFT 0.21.0.

## Diagnosis

The original E2E run found a reproducible semantic routing failure. MiniLM sent
“Travel toward the right end of the line.” to the `left` specialist in every
serving mode. Direct comparison against PyTorch's cosine similarity confirmed
that the implementation was numerically correct:

| Candidate | Cosine similarity |
| --- | ---: |
| `left` | 0.5907557267 |
| `right` | 0.5889922977 |

The incorrect lead was only `0.0017634289`. The two expert descriptions were
themselves extremely close (`0.9915475692` cosine similarity). Pure embedding
argmax therefore lacked enough semantic separation to reliably distinguish the
opposing directions. Before the fix, MiniLM completed 21/24 episodes; SmolLM2
completed 24/24. Adapter loading and policy execution behaved correctly after
the wrong MiniLM selection, confirming that the defect was isolated to routing.

## Fix

The directional regression is fixed by the optional `HybridRouter`. It computes
MiniLM scores first and delegates to SmolLM2 when the top-two gap is below
`0.05`. When users do not
supply few-shot examples, it dynamically turns each expert description into one
neutral labeled demonstration. This gives the small LM the label mapping and
required output format without task-specific keyword rules. The demonstrations
refresh when registry metadata changes.

The margin is an ambiguity heuristic, not a calibrated probability. `0.05` is
an implementation default and should be tuned on representative deployment
tasks. The hybrid strategy is an extension that composes the paper's two routing
options. The CLI and physical-AI production chain now default to the frozen
prototype-embedding router; the paper's exact description-only baseline remains
`--router embedding`. `--router hybrid` must be requested explicitly and still
exists for this historical regression.

For the unchanged failing instruction, CLI output now records:

- `expert_id: right`
- `strategy: hybrid_prompt`
- `embedding_expert_id: left`
- `margin: 0.0017634289`

An invalid or unavailable LM response aborts routing before an episode opens;
the ambiguous embedding choice is never silently used as a fallback.

## Fixed E2E result

The fixed suite runs two trained LoRA specialists through a multi-step synthetic
environment. It loads the saved JSON manifest and adapter checkpoints, uses real
pretrained text models, routes once per episode, forwards the unchanged
instruction, and verifies learned actions, termination, task success, and
adapter load/eviction behavior. Router responses and policy actions are not
mocked.

| Router | Instruction set | Resident | Disk | Multi-adapter |
| --- | --- | --- | --- | --- |
| Hybrid (no user examples) | Literal | 4/4 | 4/4 | 4/4 |
| Hybrid (no user examples) | Paraphrased | 4/4 | 4/4 | 4/4 |
| SmolLM2 (few-shot) | Literal | 4/4 | 4/4 | 4/4 |
| SmolLM2 (few-shot) | Paraphrased | 4/4 | 4/4 | 4/4 |

All 48 fixed E2E episodes pass across the three serving modes. A separate
regression test first proves that raw MiniLM still selects `left`, then requires
the zero-example hybrid to select `right` for the exact same instruction. It also
checks unseen right-hand/left-hand wording and negated directions.

The final PCA9685 robot-configuration audit, including the recorded 6 V/10 A
servo-supply rating, arm #1 channel map, and unbuilt arm #2 state, reports all
157 tests passed, including the 13 pretrained E2E tests, in 106.41 seconds. JUnit output is in
`outputs/e2e-deep-audit.xml` (SHA-256
`37033b0ac989ac49bc9912afbb6570ea52a25f8e7e7911caf710bbbf2ddff002`).
The regular suite reports 144 passing unit/integration tests with 13 pretrained
E2E cases skipped unless explicitly enabled. Its JUnit output is in
`outputs/unit-deep-audit.xml` (SHA-256
`eb7da9c99ae1ea9c026b908402efdd4e356978cd7425d8abb71a66ef4f3c363c`).
That historical wheel was installed into an isolated target and ran
`physical-demo` outside the repository; it predates the four-DOF desktop-arm
migration. The wheel
is `outputs/wheel-arm1-single-control-final-20260919/moira_robotics-0.1.0-py3-none-any.whl`
(SHA-256
`ecca25075cf1470cb30b4f4c0515334735ad6f15807e9fe57f0c3728feb4faca`).

## Four-DOF arm migration regression

On 2026-09-19, the post-migration local suite completed with 169 passed and 13
optional pretrained E2E cases skipped in 10.11 seconds. Ruff passed across
`src`, `tests`, and `tools`. `physical-demo` completed the full offline layered
flow, and `robot-model-check --verify-source` validated all 18 CAD/3MF source
artifacts while correctly reporting the new physical profile as not
motion-ready.

## Live workflow regression

On 2026-09-19, the live-session pass completed with 176 passed and 13 optional
pretrained E2E cases skipped in 11.97 seconds. Ruff passed across `src`,
`tests`, `tools`, and `deploy`; a wheel built successfully. The offline physical
E2E now performs a second perception pass after execution and reports
`camera_verified: true`. The production preflight correctly reports the active
robot as awaiting CAD/calibration and separates the missing endpoints required
for the single-arm pick/place demo from optional bimanual and task-specific
specialists. The connected CO6 USB camera was opened through the production
OpenCV source and returned a valid 43,488-byte JPEG from device index 0.

## CAD and MuJoCo regression

On 2026-09-19, the Fusion exporter captured 38 occurrences, 76 per-body mesh
references, analytic cylindrical faces, transforms, physical properties, and
metric bounds. Named CAD feature selectors recovered four revolute axes without
manual Fusion joints. The merged model contains five canonical link-local STL
meshes and three gripper mechanism meshes, with 36 of 37 occurrence bounds
matching within 0.1 mm at a validated Fusion STL scale of 0.001 m.
`Sharma_Ishaan_rotateBase:1` remains recorded as a bounds outlier because Fusion
did not write one non-meshable body.

MuJoCo 3.13 compiled the kinematic validation model as seven bodies, seven
meshes, five hinge coordinates, and one gear equality. The equality makes the
two gripper groups counter-rotate from one commanded J4 degree of freedom. All
four hierarchy checks kept the upstream body fixed while changing the intended
downstream pose. A zero-gravity 100-step run remained finite, and the home plus
per-joint renders were visually reviewed. The full local suite completed with
181 passed and 13 optional pretrained E2E cases skipped in 8.50 seconds; Ruff
passed. A separate opt-in run completed all 13 pretrained routing E2E cases in
116.35 seconds.

## Calibration and dataset regression

On 2026-09-19, the calibration-independent training pass added strict Arm #1
servo and CO6 camera records, atomic calibration application, simulation input
preflight, and an atomic episode recorder. Episode manifests pin the CAD source,
robot model, servo calibration, and camera calibration by SHA-256; frames carry
monotonic timestamps, byte counts, and content hashes. Incomplete or stale
records fail before dataset initialization or simulation training. The expanded
local suite completed with 190 passed and 13 optional pretrained E2E cases
skipped in 11.94 seconds; Ruff passed.

## Baseten deployment regression

On 2026-09-19, the read-only Baseten Management API audit confirmed that the
configured Whisper entity is a real development deployment on `L4:4x16` and
can scale from zero to one active replica. The first synthetic spoken-command
request reached the live model and exposed a container defect:
`libcublas.so.12` was missing. The Truss now pins CTranslate2 4.6.3, CUDA 12
cuBLAS, and cuDNN 9 and configures their loader path. After repushing the same
development model, the cloud endpoint transcribed the generated WAV exactly as
`Pick up the red block and place it in the blue tray.`

The Baseten client now uses the platform's special `/development/predict` and
`/development/run_remote` routes for development deployments. A new read-only
`baseten-status` command checks configured entities without exposing keys or
IDs. The router package now uses concrete Pydantic I/O schemas accepted by
Truss Chains; its actual `--dryrun` code generation passes. The expanded local
suite completed with 197 passed and 13 optional pretrained E2E cases skipped in
37.12 seconds; Ruff passed across source, tests, tools, and deployment packages.
The same router was then deployed as a development Chainlet on a `1x2` CPU
instance. Its live routing test received a two-policy allow-list and selected
`baseten-bimanual-act` for a coordinated two-arm lift request. The ignored local
environment now contains both verified development entity IDs.

## Additional checks

Both real models ran through `python -m moira evaluate` on the four illustrative
`examples/routing_samples.jsonl` entries and returned accuracy 1.0, macro-F1 1.0,
and zero invalid predictions. Direct CLI regression checks confirm that explicit
embedding mode selects `left`, while explicit zero-example hybrid mode selects
`right`.

The 3,422,777,952-byte SmolLM2 checkpoint was verified against SHA-256
`f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8`.
Pinned model revisions are:

- MiniLM: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- SmolLM2: `31b70e2e869a7173562077fd711b654946d38674`

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,prompt,adapters]"
.\.venv\Scripts\python.exe -m pytest tests/test_pretrained_e2e.py --run-e2e -q -o junit_logging=all --junitxml=outputs/e2e-fixed.xml
```

Use `-k hybrid` or `-k prompt` to select a router. The test environment is a
synthetic horizontal line designed to make a wrong expert observable. It does
not simulate robot dynamics, vision, or manipulation and does not establish
GR1/LIBERO performance or physical-robot readiness.

## Pre-calibration software integration

On 2026-09-19, the non-actuating physical workflow passed with the current
four-DOF CAD model ID and `execute=false`. Pinned Kokoro-82M generated the WAV
command, pinned Whisper Large V3 Turbo transcribed it exactly on the RTX 5070,
and GLM-5.3-Flash detected `red-block-1` and `blue-tray-1` in the controlled
scene and grounded their manipulated/destination roles. The production MiniLM
Router Chain selected the waypoint specialist for all three candidates. The
deterministic Planner Chain generated three candidates, consumed three parallel
2.5-second prediction sets, and selected a safe plan.

The motor component reported `executed=false` and its simulated driver received
zero commands. Outcome verification reported `planned` and did not claim a
post-action camera observation. Personal memory, feedback, and the run journal
were written, and Kokoro returned a valid spoken WAV response. The run exposed
and fixed two integration defects: the Router Chain now bakes MiniLM weights
into its image instead of downloading them after readiness, and Planner Chain
JSON round trips now compare list/tuple containers canonically while still
rejecting changed plan content.

The connected Windows camera at index 0 returned a 640x480 JPEG and reached the
GLM Model API; the live frame contained neither configured demo object at probe
time. The Realtek microphone captured a valid one-second 16 kHz mono WAV. The
final local suite completed with 227 passing tests and 13 optional pretrained
E2E cases skipped in 30.40 seconds; Ruff and `git diff --check` passed. Physical
camera verification and motor movement remain blocked until the measured servo,
camera, payload, and collision calibration records are complete.

## Human-error recovery integration

On 2026-09-19, the judge workflow completed a real two-turn voice recovery run
in 13.84 seconds. Kokoro generated “Move the red block,” Whisper transcribed it,
and scene-grounded NLP requested the missing destination instead of borrowing a
blue-tray reference from stored memory. A second synthesized and transcribed
answer, “Put it in the blue tray,” resolved the reference through active
dialogue. The Router Chain selected the manipulation specialist, the Planner
Chain evaluated three parallel 2.5-second futures, and the program reached
`DEMO COMPLETE` with zero motor commands.

The audit also added exact plan-bound confirmation, correction-driven
replanning, pre-execution and between-step scene revalidation, a parallel
spoken-stop monitor, and a latched PCA9685 emergency stop. Historical comments remain available for
preferences and accommodations but no longer authorize current physical
targets. The final repository suite completed with 246 passing tests and 13
hardware or opt-in model cases skipped in 8.45 seconds. Ruff and
`git diff --check` passed.
