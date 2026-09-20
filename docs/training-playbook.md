# Hackathon training and validation record

This plan targets one honest physical demonstration with the current arm. A
spoken request is grounded against the CO6 webcam scene, routed to exact
specialists, expanded into candidate plans, checked with parallel 2.5-second
predictions, executed by the Raspberry Pi, verified from a new camera frame,
and recorded for later improvement.

## Frozen hardware boundary

- Raspberry Pi 4B owns the CO6 webcam, motor authority, watchdogs, and the PCA9685.
- The RTX 5070 laptop owns voice services, orchestration, router, and simulation.
- Arm #1 has two positioning actuators: base yaw on channel 0 and shoulder pitch
  on channel 1. The failed elbow is rigid at the CAD assembly pose. The coupled
  gripper is on channel 2. Channel 3 is unused; the failed elbow is disconnected.
- The servo supply is an external regulated 6 V, 10 A source.
- Arm #2 remains a supported future installation and is not represented as
  physically available during this demo.

## Model decision

The semantic specialist router is a fine-tuned MiniLM bi-encoder:
project-specific task text is pulled toward the correct specialist description
and away from a hard negative inside the same typed interface. This preserves
MoIRA's metadata-driven routing instead of turning it into a hardcoded task
table.

Whisper STT, scene understanding, scene-grounded voice NLP, and Kokoro TTS use
existing pretrained specialists. The deterministic Planner Chain generates
candidate task steps. Two robot-specific networks are trained from 60,000
fixed-elbow randomized simulation samples: a waypoint action model and a 2–3
second command-space dynamics model. Their accepted checkpoint is hash-bound to
the CAD-derived MJCF, servo observations, dataset, and weight files. Contact,
friction, slip, force, and calibrated real-world dynamics are not learned from
simulation-only evidence.

## Router acceptance

Training, validation, holdout, and test utterances must be text-disjoint. Report
full-pool and interface-scoped accuracy, macro-F1, invalid predictions, and every
confusion. A checkpoint is accepted only when:

- validation, independent holdout, and independent test accuracy are each at
  least 95%;
- manipulation-policy and world-model accuracy are each 100% inside their
  typed production pools; and
- there are zero invalid selections.

The accepted run used 548 hard-negative triplets generated from 88 source
examples with no text overlap against any evaluation split. It reached 97.5%
validation, 95% holdout, and 95% test accuracy across the deliberately
unrestricted 20-specialist pool; both production-critical typed pools reached
100%. A rejected checkpoint never replaces the production router.

## Fixed-elbow simulator

`physical_three_actuator_model.json` is the active robot profile.
`fixed_elbow_learning.xml` contains base yaw, shoulder pitch, and
two counter-rotating gripper hinges. The forearm is attached rigidly at the
Fusion-exported elbow transform and has no elbow joint. Validation requires:

1. MuJoCo compilation with exactly four generalized coordinates;
2. base and shoulder hierarchy motion checks;
3. counter-rotation of the coupled gripper;
4. a finite 100-step run;
5. visual inspection of home and moved poses; and
6. an exact SHA-256 binding from `config/simulation_learning.json` and the
   accepted checkpoint report to the MJCF.

The learning generator randomizes response time, actuator delay, backlash, and
payload slowdown across deliberately broad configured intervals. Those values
are simulation assumptions, not physical measurements. Mass, inertia, friction,
torque, contact, and force remain uncalibrated, so the checkpoint is described
as simulation-trained command-space prediction rather than calibrated
rigid-body physics.

## Current status and remaining work

| Status | Result |
| --- | --- |
| Complete | Router generation, training, independent evaluation, and accepted checkpoint. |
| Complete | Fixed-elbow MJCF, 60,000 simulation samples, waypoint/dynamics training, held-out evaluation, and accepted checkpoint. |
| Complete | Software-only voice, vision, memory, routing, three candidates, parallel prediction, selection, MuJoCo execution, camera placement verification, and feedback journal. |
| Complete | Pi/CO6 service connectivity, camera intrinsics, PCA9685 channels, Arduino command observations, and locked motor endpoint. |
| Waiting for construction | Final camera extrinsics, exact physical workspace, gripper aperture, shoulder speed/range, and guarded no-load trials. |
| Final physical step | Enable motor authority only after the Pi readiness report is clear, then rehearse the single-arm demo. |

## Demo evidence

The terminal presentation must expose:

1. spoken request and transcript;
2. visible objects and grounded target;
3. personal context used for the plan;
4. exact specialists selected by the router;
5. multiple candidate actions;
6. parallel 2.5-second predictions;
7. the selected safe action;
8. physical execution by the Pi;
9. camera-verified outcome; and
10. stored prediction error and feedback.

The physical claim is limited to the current single fixed-elbow arm. The product
architecture can add arm #2 after its geometry, mounting transform, channels,
power budget, calibration, and guarded trials are available.
