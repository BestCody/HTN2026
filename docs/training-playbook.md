# Twelve-hour hackathon training plan

This plan targets one credible physical demonstration within twelve hours while
preserving the final dual-arm architecture. A spoken request causes physical
arm #1 to pick up a lightweight rigid object from a calibrated tabletop and
place it in a visible target region. Arm #2 remains a first-class simulated
embodiment until it is built, calibrated, and physically validated.

The hardware profile for this plan is:

- Raspberry Pi 4B for CO6! USB webcam capture, PCA9685 control, watchdogs, and
  hard safety;
- active four-servo arm design with MG996R base/shoulder and SG90
  elbow/end-effector actuators; installation and PCA9685 channels must be
  confirmed before collection;
- an external regulated 6 V, 10 A servo supply;
- one monocular CO6! USB webcam fixed above and diagonally across the workspace;
- an RTX 5070 Laptop GPU with 8 GB VRAM and 16 GB system RAM for training;
- GLM-5.3-Flash Model API for scene semantics and grounded voice NLP;
- Baseten CPU Chains for frozen MiniLM routing and deterministic planning;
- the RTX 5070 LAN service for pinned Whisper Large V3 Turbo and Kokoro TTS;
- independently hosted robot-specific endpoints after training.

The lightweight `.venv` intentionally retains CPU-only PyTorch. The dedicated
`.venv-training` environment is installed and verified with CUDA 12.8 PyTorch,
OpenCV, MuJoCo, Gymnasium, LeRobot, its dataset stack, TorchCodec, and FFmpeg 8
shared libraries. Recreate and verify it with
`powershell -ExecutionPolicy Bypass -File tools/setup_training_env.ps1`.

## Demo boundary

The physical demo supports known lightweight rigid objects on one calibrated
table plane, one installed arm, and fixed gripper orientation. It does not claim
arbitrary monocular 3D reconstruction, direct force measurement, tactile slip
sensing, arbitrary 6-DoF grasp orientation, or physical bimanual operation
before arm #2 exists.

The complete dual-arm configuration, routing contracts, synchronized action
chunks, collision checks, and simulator remain part of the design. Bimanual
weights trained without the final arm #2 geometry and real calibration would
not be credible demo artifacts, so they are not in the twelve-hour critical
path.

## Training decision

Train at most three compact robot-specific models:

1. **Markerless arm keypoint estimator.** A small vision head predicts visible
   base, shoulder, elbow, wrist/gripper keypoints and visibility confidence from
   a CO6! frame. Calibrated camera geometry and the robot kinematic model convert
   those keypoints into a joint-state belief. Recent commanded actions are a
   temporal prior, never mislabeled as measured state.
2. **Arm #1 structured-state waypoint policy.** A small behavior-cloning policy
   maps robot state, object pose, target region, object dimensions, grounded
   intent, and personal constraints into short reachable XYZ/gripper action
   chunks. Local IK, trajectory generation, collision checking, and hard safety
   remain deterministic.
3. **Arm #1 dynamics residual ensemble.** Several small MLP/GRU models predict
   the difference between calibrated MuJoCo rollouts and observed arm motion
   across the configured 2-3 second horizon. Ensemble spread represents
   uncertainty.

Do not train a large VLA or image-to-PWM policy. It would consume the time
needed for calibration and physical evaluation, require substantially more
data, and weaken the local safety boundary.

## Use pretrained components as primary specialists

Deploy and evaluate pretrained models for:

- speech-to-text;
- scene-grounded voice NLP with typed structured output;
- text-to-speech;
- object detection and segmentation;
- a frozen visual backbone for the keypoint head, when compatible;
- optional generic grasp proposals constrained to this arm's reachable
  workspace.

These are the selected production specialists, not failure fallbacks. A missing
required endpoint aborts the task before motor execution.

## Configure or calibrate without training

Do not spend the twelve-hour window training components whose behavior can be
defined and tested directly:

- MoIRA capability routing;
- personal memory retrieval and recording;
- structured task decomposition;
- camera intrinsics and table-to-robot calibration;
- tabletop pixel-to-XY projection;
- forward and inverse kinematics;
- trajectory interpolation;
- collision geometry and joint/workspace limits;
- MuJoCo rigid-body physics;
- deterministic reward and risk terms;
- PCA9685 pulse generation and watchdog behavior;
- final outcome verification from the observed object region.

## Defer until the observations or hardware exist

Do not train these for the first demonstration:

- physical bimanual policy and coordination residuals, until arm #2 exists;
- force, tactile, slip, and grasp-stability heads, until those sensors exist;
- learned object-mass measurement; monocular motion permits only an explicitly
  uncertain load estimate;
- pouring, deformable-object, insertion, lid-opening, and human-handover
  policies;
- learned human-motion, learned reward, and learned outcome-verification models.

## Required observations and labels

### Camera and workspace calibration

The CO6! must have a stable Linux `/dev/v4l/by-id/` identity. Enumerate its
actual V4L2 modes on the Pi and store the selected mode in configuration. Never
assume the laptop's integrated-camera modes describe this webcam and never bind
production capture to a mutable `/dev/videoN` index.

Keep the camera fixed after calibration. Record:

- camera intrinsics and distortion;
- camera-to-robot-base transform;
- tabletop plane and visible workspace polygon;
- arm-base location;
- capture resolution, frame rate, exposure behavior, and calibration hash.

No AprilTags or link-mounted fiducials are required. A temporary calibration
pattern displayed on an existing screen may be used for lens calibration; it is
not a runtime observation or a policy input.

### Keypoint data

Generate synthetic RGB frames from the exported arm meshes and exact joint
frames. Each frame receives perfect 2D keypoints, joint angles, visibility, and
camera metadata. Randomize:

- camera pose within the measured mounting tolerance;
- table/background appearance and lighting;
- exposure, white balance, motion blur, compression, and image noise;
- arm surface appearance;
- partial occlusion by the arm, gripper, and known objects.

Add a small real set of CO6! frames covering the joint range and relevant
occlusions. Label the visible keypoints manually. Split by pose sequence and
camera session so neighboring frames do not leak into validation.

### Policy demonstrations

Use calibrated IK and trajectory generation as the simulation expert. Randomize
reachable object and goal positions, object sizes, permitted mass classes,
servo lag, backlash, friction, and camera-state uncertainty. Save successful
and failed episodes in an episode-aware format:

- structured grounded task and target identity;
- image and calibration IDs;
- estimated robot state and state confidence;
- object pose, target region, and confidence;
- proposed and executed waypoint/gripper chunks;
- simulator parameters;
- collision, intervention, uncertainty, and outcome labels.

The policy predicts reachable XYZ plus gripper commands. It does not predict an
arbitrary end-effector quaternion because the current arm has three positioning
joints and no independently controlled wrist orientation.

### Dynamics data

First pretrain the residual ensemble using domain-randomized simulator
transitions. Then collect low-speed physical sweeps and guarded pick/place
attempts. Each transition window contains:

- estimated joint state and confidence;
- commanded trajectory and timing;
- observed future joint/end-effector state;
- object pose and coarse load class;
- simulator prediction;
- execution outcome and any safety stop.

Train the residual, not a replacement physics model. If there is insufficient
real data to beat calibrated MuJoCo on held-out physical sweeps, do not use the
residual in action selection and do not claim it improved prediction.

## Model shapes sized for the available GPU

The exact dimensions come from the dataset and robot configuration. The initial
models should remain deliberately compact:

- keypoint model: pretrained lightweight image encoder plus a heatmap or
  coordinate head at a cropped input resolution;
- waypoint policy: structured-state MLP or small temporal transformer producing
  a short action chunk;
- dynamics: three-to-five independently seeded compact MLP/GRU members.

Use mixed precision, frozen visual features for the first run, small batches,
gradient accumulation when necessary, and early stopping on held-out episodes.
Do not unfreeze a full visual-language backbone during the hackathon window.

## Twelve-hour execution schedule

Work streams overlap where the dependency permits it.

| Time | Required result |
| --- | --- |
| 0:00-1:00 | CUDA PyTorch works on the RTX 5070; MuJoCo and vision dependencies import; the physical demo task and success geometry are frozen. |
| 0:30-2:00 | Fusion exports exact meshes, joint origins, link lengths, and mass data; arm #1 pulse endpoints, direction, home, safe speed, and stop path are measured. |
| 1:00-3:00 | CO6! capture works on the RTX laptop; stable device identity, camera calibration, tabletop transform, and synchronized frame timestamps are recorded. |
| 2:00-5:00 | Synthetic keypoint images, expert policy demonstrations, and randomized dynamics transitions are generated. Real keypoint frames and low-speed sweeps are recorded in parallel. |
| 3:30-6:00 | Keypoint head, structured waypoint policy, and initial dynamics ensemble train on the RTX GPU. |
| 5:30-8:00 | Held-out offline evaluation, checkpoint export, typed inference wrappers, and router registration complete. |
| 7:00-10:00 | Plan-only shadow runs and low-speed physical trials expose calibration and integration errors. Fine-tune only models whose held-out or physical metrics show a specific deficit. |
| 10:00-12:00 | Rehearse unseen start positions and paraphrased voice commands; freeze working checkpoints/configuration; retain the last period for physical reliability fixes. |

## Stop/go gates

These gates prevent training time from hiding missing fundamentals:

1. **Hour 1:** CUDA must be active. CPU-only training is outside the schedule.
2. **Hour 2:** Exact geometry and safe arm calibration must exist. Synthetic
   data generated from guessed geometry is rejected.
3. **Hour 3:** CO6! frames, timestamps, camera calibration, and tabletop mapping
   must be repeatable. Moving the camera invalidates calibration.
4. **Hour 6:** Each trained checkpoint must beat its declared non-learned
   baseline on held-out data. Training loss alone is insufficient.
5. **Hour 8:** The complete request must run in shadow mode without any
   production placeholder or offline fixture being selected.
6. **Hour 10:** Low-speed physical trials must show repeatable success before
   presentation behavior is frozen.

Acceptance thresholds belong in an evaluation configuration tied to object
size, image dimensions, and workspace scale. They must not be scattered as
robot-specific literals through the implementation. The demo target is at least
eight successes in ten held-out, low-speed tabletop pick/place trials, with no
uncaught safety violation. Uncertain perception or state is counted as a failed
trial, not success.

## Demo evidence

The presentation should expose the complete trace:

1. spoken request and transcript;
2. visible objects and grounded target;
3. retrieved personal accommodation or preference;
4. exact policy/world components chosen by the router;
5. multiple candidate action chunks;
6. 2-3 second physics and learned-residual predictions;
7. selected plan and local safety decision;
8. physical arm motion;
9. observed final object region and outcome;
10. saved prediction error for later improvement.

The physical claim is single-arm for the current hardware. The product and
simulator remain dual-arm, and the same training interfaces accept arm #2 after
its geometry, mounting, channels, power budget, calibration, and real rollouts
are available.

## Repository blockers found by the audit

The GPU training environment is complete. Robot training still cannot begin
without adding or completing:

1. exact Fusion export and a valid MuJoCo digital twin;
2. live integration between the implemented UVC/OpenCV CO6! source and episode
   recorder;
3. measured values for the implemented camera/workspace calibration record;
4. laptop orchestration for the implemented synchronized, atomic episode
   recorder, using Pi telemetry over the authenticated robot link;
5. markerless keypoint dataset generation, labels, trainer, and inference;
6. structured waypoint-policy dataset/trainer and inference wrapper;
7. simulator/residual-dynamics dataset/trainer and inference wrapper;
8. real endpoint packages and model IDs for every component used in the demo;
9. physical shadow and guarded-rollout commands.

The generic LoRA helper in `src/moira/training.py` is not any of these trainers.
It remains useful for compatible pretrained backbones but does not create robot
datasets, ACT policies, keypoint models, or dynamics models by itself.
