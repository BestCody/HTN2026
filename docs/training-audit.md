# Robot model training audit

Audit date: 2026-09-19.

## Conclusion

The repository has a strong typed orchestration and safety skeleton, but it is
not yet a trained physical-robot stack. The production catalog contains 31
components: 20 remote components, 10 local components, and one hardware
component. Nineteen remote entries are `DEPLOYMENT:` placeholders and the task
planner is a `CHAIN:` placeholder. No production model checkpoint is stored in
the repository. The only checked-in datasets are small JSONL corpora for text
routing calibration and regression; there are no camera, robot-state, action,
tactile, load, or task-outcome datasets.

The existing LoRA helper is real and tested, but it is a generic PEFT loop. Its
caller must supply the backbone, batches, and loss. It does not implement an
ACT, diffusion-policy, VLA, vision, tactile, or world-model data loader or
trainer. The pretrained end-to-end tests train two tiny linear left/right
policies in a synthetic line world. They verify routing and adapter serving,
not robot manipulation.

For the twelve-hour voice-controlled arm #1 demonstration, train no more than
three compact robot-specific models: a markerless arm-keypoint head, a
structured-state waypoint policy, and a short-horizon dynamics residual
ensemble. Use calibrated MuJoCo as the primary physics world model. Most voice
and object-perception models should start from pretrained weights. Kinematics,
trajectory generation, collision enforcement, memory, outcome geometry, the
PCA9685 driver, and hard safety should be calibrated or configured rather than
trained. Arm #2 remains part of the simulator and final architecture, but real
bimanual adaptation and validation wait for the second physical arm.

## Blocking findings

### Critical: real training data cannot be collected yet

1. The robot configuration is not motion-ready. Exact Fusion geometry, joint
   origins, mesh scale, pulse endpoints, velocity limits, controller timing,
   safety limits, payload validation, PCA9685 address/frequency, power validation,
   and an independently tested stop path remain unresolved.
2. Physical execution requires a timestamped measured `RobotState`, but the
   PCA9685 hobby-servo path supplies commands only. No encoder, potentiometer reader,
   or camera-based joint-state estimator exists. Treating commanded pulse widths
   as measured state would violate the current execution contract.
3. There is no teleoperation interface, demonstration recorder, synchronized
   episode format, dataset versioning, or train/validation/test split tooling.
4. Camera capture is a single JPEG through CSI-oriented `rpicam-still`. The
   confirmed sensor is a monocular CO6! USB webcam, but there is no UVC/V4L2
   stream source, stable device discovery, synchronized frame pipeline, camera
   intrinsic/extrinsic calibration loader, tabletop mapping, or markerless
   robot-pose tracker.
5. The production Baseten model services do not exist in this repository. A
   general frozen-MiniLM router Chain is implemented, but it has not yet been
   deployed. Model endpoint identifiers and the factory wiring shown in
   documentation remain placeholders.

### Critical: some learning contracts exceed the current hardware

1. The grasp contract carries a full 6D pose, while the physical arm has three
   positioning joints and a gripper. The current analytic IK consumes target
   position and does not control the requested quaternion. A robot policy must
   be trained for reachable XYZ targets with the arm's fixed orientation
   constraints, or the hardware must gain more controllable axes.
2. Policy chunks include a force limit, but the PCA9685 driver controls gripper
   width only and cannot measure or close the loop on force. A load cell, force
   sensor, motor-current sensor, or calibrated compliant mechanism is required
   before force is a trustworthy observation or command constraint.
3. MG996R and SG90 command pulses do not reveal actual joint position, stall,
   backlash, or deflection. Robot-specific policy and dynamics training therefore
   require external pose sensing or added joint feedback.
4. The replacement arm's installed state has not been confirmed, and arm #2 is
   still a future build. Bimanual policy and coordination-model training can use
   simulation, but no real bimanual dataset can be collected or validated yet.

### High: model labels currently overstate implementations

1. `local-tactile-sparsh` names a SPARSH-style shared encoder, but no SPARSH
   dependency, checkpoint, preprocessing pipeline, or trained head is present.
   The offline implementation computes contact, force, slip, and stability from
   already-structured normal/shear samples with deterministic equations.
2. The six offline world-model routes share one deterministic state-space
   fixture. Its object lift trajectory, collision probabilities, uncertainty,
   and task success are hand-authored formulas rather than learned predictions.
3. The offline manipulation specialists convert semantic steps to action chunks
   with rules. They do not infer robot actions from images or demonstrations.
4. The offline outcome, failure, load, reward, and feedback components are
   telemetry rules. They are useful safety and integration baselines, but they
   are not trained models.
5. The Baseten router now performs frozen prototype-cosine routing over a
   compatible, allow-listed pool using task text, specialist descriptions, and
   representative phrases. It requires metadata evaluation rather than weight
   training. It does not optimize latency/cost or adapt weights from outcomes.

The frozen router's small authored test set scored 19/20 over all 20 experts,
6/6 inside the compatible world-model interface, and 5/6 inside the policy
interface when given raw instructions. With the production-shaped grounded
candidate query (goal, planner action, arms, and target), the policy interface
scored 6/6. These checks validate the software path and metadata, not general
robot-task accuracy; catalog changes require a new held-out evaluation.

## Complete component disposition

| Layer/component | Current repository reality | Training disposition |
| --- | --- | --- |
| Baseten component router | Frozen MiniLM semantic selection inside the Pi-supplied compatible allow-list | No training. Maintain expert descriptions and interface metadata; evaluate every catalog change on held-out task paraphrases. |
| Scene perception | Remote placeholder; offline component accepts object dictionaries rather than pixels | Deploy a pretrained detector/segmenter and derive tabletop pose from calibrated monocular geometry. Fine-tune only if target objects or the workspace fail evaluation. |
| 6D grasp pose | Remote placeholder; offline top/side geometric fixture | Start from a pretrained grasp model, then constrain/evaluate it for this gripper and 3-DOF reachable workspace. Robot-specific fine-tuning is optional until pretrained grasps fail. |
| Personal memory | Real local SQLite profile/event store | No training. Define retention and retrieval tests. |
| Speech-to-text | Remote placeholder; offline path decodes UTF-8 bytes | Use a pretrained STT deployment. Fine-tuning is optional for workshop noise, accents, or robot-specific vocabulary. |
| Scene-grounded voice NLP | Remote placeholder; offline keyword/object-name fixture | Use a pretrained language model with structured output and scene context. Build an evaluation set first; fine-tune only if prompting does not meet grounding accuracy. |
| Text-to-speech | Remote placeholder; offline tagged-byte fixture | Use pretrained TTS. Train only for optional voice cloning or a custom voice. |
| Personalized task planner | Remote Chain placeholder; offline rule planner | Prompt/tool engineering and evaluation are sufficient initially. Training is optional after real plan/outcome data exists. |
| Waypoint manipulation policy | Remote placeholder; offline rule converter | **Train or robot-adapt for arm #1. This is the first required policy model.** An analytic scripted pick/place controller is a valid deliberate alternative, but is not a learned policy. |
| Bimanual ACT policy | Remote placeholder | Train in simulation if useful; real fine-tuning and validation must wait for arm #2. |
| Pour policy | Remote placeholder | Train only if pouring is in the hackathon demo. It also requires liquid/deformable observations and outcome labels. |
| Insert policy | Remote placeholder | Train only if insertion is in scope; requires tight pose/force data the current hardware does not provide. |
| Open-lid policy | Remote placeholder | Train only if included in the demo; requires suitable gripper mechanics and task demonstrations. |
| Handover policy | Remote placeholder | Postpone unless human handover is a demo requirement; needs human-pose and safety data. |
| Analytic IK | Implemented local two-link solver | No training. Export and validate exact geometry and reachable frames. |
| Trajectory planner | Implemented bounded interpolator | No training. Calibrate joint/gripper limits, rates, and timing. |
| Collision checker | Implemented conservative checker | No training. Supply validated meshes, scale, transforms, workspace geometry, and clearance. |
| Learned forward dynamics | Remote placeholder; deterministic offline fixture | **Train a robot-specific short-horizon residual/state model** if learned prediction is a core claim. Requires measured state, actions, and observed next states. |
| Rigid-body world model | Remote placeholder; deterministic offline fixture | Prefer a physics engine plus system identification first. A learned residual can be trained from rollout errors. |
| Grasp-contact world model | Remote placeholder; deterministic offline fixture | Train after contact/slip sensing exists. Requires grasp attempts with contact, force, slip, and success labels. |
| Bimanual coordination world model | Remote placeholder | Simulation pretraining is possible; real training waits for arm #2 and synchronized state feedback. |
| Deformable/liquid world model | Remote placeholder | Train or deploy a pretrained model only for pour/deformable tasks. Not required for basic pick/place. |
| Human-motion world model | Remote placeholder | Start with a pretrained human pose/trajectory model. Fine-tune only after collecting consented workspace data and evaluating safety coverage. |
| Task reward model | Remote placeholder; deterministic weighted score offline | Keep deterministic task metrics for the first demo. Train preference/reward models only after labeled outcomes exist. |
| Hard safety risk | Real deterministic local checks | Never replace the hard boundary with a learned-only model. Calibrate thresholds and retain deterministic enforcement. |
| Tactile encoder/heads | Manifest placeholder; deterministic structured-sample equations | Select/install tactile hardware first. Then deploy a pretrained encoder and train/calibrate contact, slip, force, and stability heads for the sensor/gripper. |
| PCA9685 control | Implemented, gated hardware driver | No training. Complete pulse, frequency, power, timing, and stop-path calibration. |
| Outcome verifier | Remote placeholder; telemetry rule offline | A pretrained vision model may be enough initially. Fine-tune a binary/multistage verifier from before/after images and task labels if reliability is insufficient. |
| Failure classifier | Deterministic telemetry rules | No training initially. Train later from balanced failure logs if rule coverage becomes limiting. |
| Load estimator | Averages `measured_mass_kg`; no physical measurement source | Install and calibrate a sensor. Train a regression model only if mass must be inferred from current/deflection/vision rather than directly measured. |
| Prediction-error tracker | Deterministic comparison of predicted and observed success | No training. Persist errors as future world-model training examples. |
| Feedback learner | Deterministic memory update and recommendation rules | No training initially. It needs trustworthy outcome, failure, and load inputs. |

## Models required for the first credible hackathon demo

Assume the demo is voice-commanded single-arm pick-and-place of known rigid
objects in a bounded workspace.

### Must be supplied or trained

1. A markerless robot-state estimator trained from synthetic arm renders and a
   small labeled CO6! set. Calibrated kinematics and command history turn its
   visible keypoints into an explicitly uncertain state belief. Commanded PWM
   alone is not measured state.
2. One arm #1 manipulation policy conditioned on the grounded task, monocular
   object pose, and estimated robot state. It should be a compact
   structured-state policy that outputs bounded Cartesian waypoints or action
   chunks; local IK and trajectory code retain motor authority.
3. One short-horizon forward/residual dynamics model if the demo claims learned
   parallel prediction. Otherwise a deliberately selected physics simulator can
   provide the initial prediction component, but it must not be presented as a
   trained world model.
4. A deployed perception model and outcome verifier. These can begin with
   pretrained weights, but must be evaluated on the actual camera, objects, and
   lighting.

### Use pretrained models without custom training initially

- speech-to-text;
- scene-grounded language model;
- text-to-speech;
- object detection/segmentation plus calibrated tabletop pose estimation;
- grasp proposal generation, after reachability constraints are applied;
- optional human detection and short-horizon motion estimation.

### Keep deterministic and calibrate

- frozen semantic router and deterministic compatibility/authority gate;
- SQLite personal memory;
- IK, trajectories, collision checks, and hard safety;
- PCA9685 control and emergency stop;
- initial task reward, failure classification, prediction-error tracking, and
  memory feedback.

### Defer

- bimanual policy and world model until arm #2 exists;
- pour/deformable, insertion, lid, and handover specialists unless one is the
  chosen demo task;
- learned reward/personalization until enough labeled executions exist.

## Dataset that must be recorded

Each episode needs an immutable robot/calibration version and synchronized:

- user ID or anonymized profile ID, audio, transcript, grounded intent, and task;
- CO6! RGB frames with camera timestamps and calibration IDs;
- camera intrinsics, camera-to-base extrinsics, coordinate frame, and workspace;
- detected object IDs, labels, dimensions, poses, and confidence;
- measured joint positions, gripper width, and observation timestamp;
- commanded Cartesian target, joint target, pulse width, duration, and
  end-effector command for the calibrated per-joint channels;
- contact, force, slip, motor current, supply voltage, and load measurements when
  those sensors exist;
- predicted world states and selected plan;
- final object pose, success, failure class, human intervention, and safety stop;
- model/checkpoint IDs, code revision, and the robot-model configuration hash.

Do not mix data collected under different pulse calibration, camera extrinsics,
link geometry, or sensor scaling without recording those versions. Split
evaluation by episode and initial object arrangement rather than randomly
splitting adjacent video frames, which would leak nearly identical observations.

## Missing training/deployment engineering

The following code must be added before production training can start:

1. UVC/V4L2 CO6! capture plus a Raspberry Pi episode recorder with monotonic
   timestamps and atomic episode completion.
2. Safe teleoperation UI or controller that emits the same bounded action
   representation used by deployed policies.
3. A `perception.robot_state` component contract, markerless keypoint inference,
   and a `RobotState` representation that records source, confidence/uncertainty,
   and age instead of labeling every supplied value as measured.
4. Camera/table calibration storage plus a dataset schema, integrity validator,
   calibration/version manifest, and
   train/validation/test split tool.
5. Markerless keypoint dataset renderer/labeler, trainer, evaluation metrics,
   checkpoint metadata, and serving package.
6. Compact structured-state policy backbone wrapper, loss, data loader,
   evaluation metrics, checkpoint metadata, and Baseten serving package.
7. MuJoCo environment plus a residual-dynamics backbone, rollout loss,
   uncertainty calibration, multi-step evaluation, and serving package.
8. Perception, grasp, voice, planner, reward, and outcome Baseten model packages
   or explicit integrations with deployed pretrained endpoints.
9. Production registry bootstrap that replaces every placeholder with an actual
   endpoint ID and constructs all local factories.
10. Offline replay, shadow planning, low-speed hardware evaluation, and regression
   gates for every checkpoint.
11. Dataset and checkpoint lineage so feedback samples cannot silently train a
    model against stale robot geometry or calibration.

## Recommended execution order

1. Choose one single-arm rigid-object task and define objective success from
   observable state.
2. Finish arm, camera, state-sensing, and stop-path calibration.
3. Implement the recorder and teleoperation path; validate recordings before
   spending time training.
4. Collect a small pilot dataset and replay every episode through the typed
   contracts. Fix synchronization and labeling errors.
5. Collect a diverse task dataset, keeping failed and human-intervened episodes
   with explicit labels.
6. Train the single-arm policy and evaluate it entirely offline.
7. Run plan-only shadow evaluation, then guarded low-speed trials.
8. Train the short-horizon dynamics residual from the resulting transitions and
   calibrate its uncertainty against held-out rollouts.
9. Deploy the policy/world models on Baseten, enter their exact IDs in the
   production factory configuration, and run the complete Pi-to-cloud-to-Pi E2E.
10. Add specialized skills only after the base pick/place loop is reliable.

This sequence preserves the no-fallback design: a missing, invalid, or
unavailable production component remains an error rather than silently selecting
an offline fixture.

The concrete collection, training, evaluation, and deployment procedure is in
the [twelve-hour hackathon training plan](training-playbook.md).
