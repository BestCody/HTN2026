# Raspberry Pi 4B and Baseten architecture

The Raspberry Pi captures the attached CO6 webcam and owns the PCA9685
arm-control boundary. The RTX laptop retains personal memory, enforces the
component allow-list, runs orchestration and local GPU voice models, and calls
the remote router and specialist models. The production path
has no automatic model fallback: a camera, router, network, schema, or inference
failure is reported and physical execution does not begin.

```mermaid
flowchart LR
    MIC[Microphone] --> LAPTOP[RTX laptop Charlie brain]
    CAM[CO6 webcam] --> PI[Pi camera + robot service]
    PI -->|authenticated JPEG| LAPTOP
    LAPTOP -->|task text + compatible allow-list| ROUTER[Fine-tuned MiniLM router Chain]
    ROUTER -->|component ID only| LAPTOP
    LAPTOP --> STT[Local RTX Whisper]
    LAPTOP --> VNLP[Baseten scene-grounded voice NLP]
    LAPTOP --> VISION[Baseten vision / pose]
    LAPTOP --> GRASP[Baseten 6-DoF grasp model]
    LAPTOP --> PLAN[Baseten task planner]
    LAPTOP --> POLICY[Baseten task-specific manipulation policy]
    LAPTOP --> WORLD[Baseten specialized world models]
    LAPTOP --> REWARD[Baseten task reward model]
    LAPTOP --> OUTCOME[Baseten outcome verifier]
    LAPTOP --> TTS[Local RTX Kokoro]
    LAPTOP --> MEM[(Laptop SQLite personal memory)]
    LAPTOP --> LOCAL[IK + trajectory + collision + hard safety]
    LOCAL -->|authenticated bounded action chunks| PI
    PI --> CTRL[Local limit checks + watchdog]
    CTRL --> PCA[PCA9685 PWM controller]
    PCA --> ARM[Current arm: base + shoulder + coupled gripper]
    ARM --> SENSORS[Future feedback sensors + stop input]
    SENSORS --> PI
```

## Why this split

Raspberry Pi 4 Model B has a quad-core Cortex-A72 CPU, 1–8 GB RAM depending on
the board, one exposed CSI camera connector, and a 5 V / 3 A supply requirement.
It is a good sensor and control gateway, but it is not the right host for all of
the vision, speech, planning, and simulation models at once. The official Pi 4
documentation also limits downstream USB current to about 1.1 A in aggregate.
Arm motors need their own correctly sized supply and controller; never power
them from the Pi's USB or GPIO rail. The Pi powers only PCA9685 logic and
communicates over IÂ²C; a separate regulated servo supply feeds `V+`, with a
shared ground.

Baseten supports separately deployed models and Chains with independently
scaled components and parallel calls. That maps directly to this project's
exact capabilities. The laptop stores deployment clients and orchestration
state; the GPU model memory remains in Baseten and local RTX services.

Official references:

- [Raspberry Pi 4 product brief](https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-product-brief.pdf)
- [Raspberry Pi 4 datasheet](https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-datasheet.pdf)
- [Baseten Chains architecture](https://docs.baseten.co/development/chain/design)
- [Baseten Chain invocation](https://docs.baseten.co/development/chain/invocation)

## Request flow

1. The laptop records bounded audio and requests CO6 JPEG frames, camera
   identity, and capture timestamps from the authenticated Pi endpoint.
2. STT returns a transcript.
3. Perception returns object IDs, labels, confidence, 3D positions, estimated
   masses, and hazards.
4. Personal memory returns preferences, accommodations, recent comments,
   workspace context, and masses measured in earlier tasks.
5. The DUM-E-style voice NLP component grounds the transcript to the perceived
   object IDs using the current transcript and active dialogue. Personal memory
   supplies preferences and accommodations, but historical comments cannot
   authorize a physical target. An unresolved object produces a spoken
   clarification and waits for a bounded follow-up turn.
6. A 6-DoF grasp specialist returns scored gripper poses and collision
   probabilities for every grounded target.
7. The planner proposes a small set of typed candidates. The compatibility gate
   forms the manipulation-policy pool, and the general text router chooses the
   best waypoint, bimanual ACT, pour, insert, lid, or handover specialist for
   each candidate. Policies return bounded action chunks, never raw motor
   authority.
8. Laptop-side IK, trajectory generation, and collision checks validate each policy
   against current joint state, geometry, joint limits, and workspace clearance.
9. Each viable policy is evaluated concurrently by the learned forward-dynamics
   model and the relevant rigid, contact, bimanual, deformable, or human-motion
   world models. Each returns future poses, joints, forces, slip, collision
   probability, success, uncertainty, and risks through a 2–3 second horizon.
10. A task-reward model scores progress. Laptop hard safety fuses world-model,
    collision, and tactile results; the Pi independently enforces calibrated
    command envelopes. If every candidate is unsafe, execution stops.
11. The planner selects only a candidate that was generated and evaluated. The
    laptop never receives raw PWM authority; the Pi rejects unbound, duplicate,
    mismatched, uncalibrated, or disabled motion requests.
12. Before physical execution, the system reads the interpreted command and
    selected candidate back to the user. A fresh confirmation is bound to that
    exact intent and plan. Corrections trigger full replanning; a changed plan
    requires confirmation again. A fresh camera frame blocks moved or missing
    targets and new hazards. Further camera checks run between action steps,
    keeping stationary destinations under observation and watching the
    manipulated object until planned lift or transfer begins.
13. The laptop sends only validated action chunks to the authenticated Pi robot
    endpoint. The Pi checks the robot ID, CAD digest, request uniqueness, chunk
    binding, timing, calibrated joint and gripper envelopes, then drives the
    installed hardware concurrently and stops every arm if one reports failure.
    A parallel bounded microphone listener recognizes local stop phrases during
    movement and calls the interruptible PCA9685 driver. The stop remains
    latched until the operator checks the workspace and restarts the runtime.
14. Outcome verification, failure classification, load estimation, and per-model
    prediction-error measurement close the model-based feedback loop.
15. Current/load sensing records measured object mass and execution issues.
    SQLite makes those facts available to later voice grounding, world-model,
    policy, and planning requests.
16. TTS speaks the status or clarification.

This is task-level orchestration. Network inference must never be placed inside
a servo-rate loop. Joint control, limits, collision interlocks, watchdogs, and
the emergency stop belong on the local motor controller and Pi-side driver.

## Component contract

[`examples/pi4_components.json`](../examples/pi4_components.json) is the edge
allow-list. Every component declares one or more exact, layer-prefixed
capabilities, and wildcards are rejected. These contracts define which experts
can exchange the requested typed data and which operations must stay local.
They do not encode a task-to-expert decision.

The compatible remote experts are described in
[`examples/physical_ai_specialists.json`](../examples/physical_ai_specialists.json).
Each declares one or more versioned interfaces, descriptions, and representative
routing phrases. The router receives task text plus only the IDs allowed by the
laptop registry, embeds the task and expert-owned prototypes with the accepted
MiniLM checkpoint, and returns the ID with the highest normalized prototype-
centroid cosine similarity. The
laptop maps that ID to its configured endpoint and validates it again. It never
accepts a URL or credential from the router.

The current capabilities are:

| Layer | Exact capability | Location |
| --- | --- | --- |
| Perception | `perception.scene` | Baseten vision/pose deployment |
| Personal intelligence | `personal.recall`, `personal.record` | Laptop SQLite |
| Voice | `voice.transcribe`, `voice.ground`, `voice.synthesize` | Laptop RTX STT/TTS; Baseten grounding |
| Planning | `planning.candidates`, `planning.select` | Baseten planner Chain |
| Grasp | `grasp.pose_6d` | Baseten grasp deployment |
| Manipulation | `manipulation.skill.waypoint`, `.pour`, `.insert`, `.open_lid`, `.handover`; `manipulation.bimanual` | Separate Baseten policies |
| Kinematics | `kinematics.inverse` | Laptop, robot-specific geometry |
| Motion | `motion.trajectory`, `motion.collision_check` | Laptop planning; Pi calibrated envelopes |
| Dynamics | `dynamics.predict` | Baseten learned forward model |
| World | `world.rigid_dynamics`, `.grasp_contact`, `.bimanual_coordination`, `.deformable_dynamics`, `.human_motion` | Separate Baseten models |
| Reward and safety | `reward.task_progress`; `safety.risk` | Baseten reward; laptop fusion; Pi command gate |
| Tactile | `tactile.contact`, `.slip`, `.force`, `.grasp_stability` | Robot sensors through Pi telemetry |
| Control | `control.single_arm`, `control.bimanual` | Pi hardware driver only |
| Outcome | `outcome.verify`, `failure.classify`, `load.estimate` | Baseten verifier; laptop telemetry models |
| Feedback | `feedback.prediction_error`, `feedback.learn` | Laptop memory from Pi telemetry |

The typed JSON decoder in `moira.contracts.decode_physical_response` validates
remote vision, voice, grasp, policy, world-model, reward, outcome, planner, and
TTS responses before downstream use. A voice model cannot invent an object ID
that perception did not return. World models must cover the configured horizon,
and every returned plan ID must match its candidate.

## Baseten setup

Deploy each model independently, or group tightly coupled model work in a
Chain. Keep the
router a small CPU Chain; a deployable template is in
[`deploy/baseten_router/router.py`](../deploy/baseten_router/router.py).

```bash
cd deploy/baseten_router
truss chains push router.py --dryrun
truss chains push router.py --promote
```

On the laptop, set the API key outside the manifest. The Pi does not receive
Baseten credentials:

```bash
export BASETEN_API_KEY='...'
```

The production config creates the laptop registry, cloud clients, local camera,
and authenticated Pi controller without placing model credentials on the robot:

```python
import os

from moira import OpenCVCameraSource
from moira.production import build_physical_session, load_physical_runtime_config

config = load_physical_runtime_config("config/pi4_runtime.json")
camera_device = os.environ[config.camera.device_env]
camera_source = int(camera_device) if camera_device.isdecimal() else camera_device
camera = OpenCVCameraSource(camera_source, camera_id=config.camera.camera_id)
session = build_physical_session(config, (camera,))
```

The router remains a required Baseten Chain. No local LLM or automatic model
fallback is selected if the router, specialist, or Pi robot endpoint fails.

## Active fixed-elbow desktop arm

The SolidWorks assembly remains the geometry source. The physical elbow motor
has failed mechanically closed and holds the link rigidly, so the active profile
uses MG996R base yaw on channel 0, MG996R shoulder pitch on channel 1, and the
SG90 gripper on channel 2. Channel 3 is unused and the failed elbow is disconnected. Charlie emits two positioning
angles plus gripper aperture.

The Fusion export supplies five assembled link meshes, the original joint
frames, a Y-up frame, and 154.14 mm/100.10 mm link spacing. The active MuJoCo
model removes the elbow hinge and keeps the CAD assembly transform rigid. It
compiles with four generalized coordinates: base, shoulder, and two coupled
gripper fingers. Automated hierarchy checks, a 100-step finite-state run, and
visual renders pass. The active model and report are
`physical_three_actuator_validation.xml` and
`physical_three_actuator_validation.json`.

The Arduino calibration used `Adafruit_PWMServoDriver` at 50 Hz with a global
102-to-512 tick mapping. The recorded base, shoulder, and gripper command ranges
are stored separately from physical joint angles in
`config/arm1_servo_observations.json`. The future two-arm simulator can still
exercise coordination, but physical bimanual control remains unavailable until
arm #2 is built and independently calibrated.

The robot remains motion-locked while physical velocity, gripper aperture and
force, payload, collision clearance, power compatibility, camera workspace,
and stop-path checks are incomplete. `moira robot-model-check` reports every
remaining value rather than substituting measurements.

## Pi bring-up

Use 64-bit Raspberry Pi OS. The Pi installs only the core and PCA9685 hardware
dependencies; it does not install OpenCV, router weights, model SDKs, or cloud
credentials.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[hardware]"
./.venv/bin/moira pi-check --strict
./.venv/bin/moira-pi-controller \
  --host 0.0.0.0 \
  --port 8770 \
  --model robot_models/four_dof_desktop_arm/physical_three_actuator_model.json
```

This command is intentionally locked. It exposes health and emergency-stop
handling but returns HTTP 423 for motion. Add `--enable-motion` only after the
calibrated model passes readiness checks; the service still opens I2C lazily on
the first authenticated execution request.

For the current `client` Pi account, install the repository's locked systemd
unit so the controller starts after networking and restarts after a process
failure:

```bash
sudo install -m 644 deploy/pi/moira-robot-controller.service \
  /etc/systemd/system/moira-robot-controller.service
sudo systemctl daemon-reload
sudo systemctl enable --now moira-robot-controller.service
```

The checked-in unit deliberately omits `--enable-motion`. Calibration completion
does not silently unlock the arm; enabling physical motion remains a separate,
reviewable deployment change.

The CO6 camera is attached to the Pi at the stable UVC path
`/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM-video-index0`. The same locked Pi
service exposes authenticated `/v1/camera`, `/v1/control`, and `/v1/stop`
endpoints; enabling the camera does not enable motor motion.

The voice design follows the same high-level path as the Hack the North 2025
[DUM-E project](https://devpost.com/software/dum-e-kgx6at): speech, computer
vision, fast language reasoning, and robot control. This implementation extends
that path with typed grounding, dialogue and personal context, parallel
simulation, bimanual steps, measured-load feedback, and strict local motor
authority.
