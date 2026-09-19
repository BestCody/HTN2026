# Four-DOF desktop arm audit

Audit date: 2026-09-19.

The active robot changed from the retired lightweight Fusion arm to the
SolidWorks design in `4+DOF+Robotic+Arm+(SG90,+MG996R,+Arduino)`. Its primary
assembly is `Sharma_Ishaan_robotAssem.SLDASM`, SHA-256
`bc5ad54ea950fa068b1c95d26e907497d43277aab97aa67eb58ef610bfd86df5`.
The repository also contains `Robotic+Arm.3mf`, SHA-256
`e693d19f0b99ef43a45988a602a4d2f35290945013fd3775e795f8ca8e9d6814`.
All 18 source artifacts are pinned in the model's checksum manifest and are
verified by `robot-model-check --verify-source` and the test suite.

## Confirmed design facts

The published model description and supplied assembly agree on this actuator
layout:

- MG996R turntable/base servo;
- MG996R shoulder servo;
- SG90 elbow servo;
- SG90 wrist/gripper mechanism;
- three positioning axes plus an end-effector actuator;
- open-loop hobby servos with no position feedback; and
- an approximate 15–20 cm workspace stated by the designer.

The designer also reports that inexpensive servo dimensions vary from the CAD
models and that the SG90 elbow struggles with the arm and payload. Those are
material calibration constraints, not generic warnings. The active profile now
records servo type per joint and blocks motion until actual fit and payload are
validated. Source description: [4 DOF Robotic Arm (SG90, MG996R, Arduino)](https://3dgo.app/models/makerworld/3016491).

## 3MF findings

Read-only inspection of the 3MF found:

- millimetre model units;
- 12 unique printable mesh names and 41,638 triangles;
- 54 build items spread over six print plates;
- no mesh repairs reported by the slicer metadata; and
- no joint metadata.

The 3MF's transforms arrange copies for printing. They are not assembly poses.
Using them for MuJoCo would place links across six plates and yield meaningless
joint geometry. The 3MF is useful for printing, visual geometry, units, and
cross-checking component names. The SolidWorks/Fusion assembly remains the
source for link transforms, joint origins, axes, and mass properties.

## Removed stale assumptions

The migration removed facts that belonged only to the previous robot:

- four MG996R servos;
- base/shoulder/elbow/gripper channels 0–3;
- the old joint limits and home offsets;
- Y-up coordinates and five old Fusion component names;
- 25 g simulation payload and 0.5 N m torque evidence; and
- the claim that physical arm 1 was this CAD design.

No replacement value was guessed. Unknown axes, limits, link lengths, payload,
wiring, installed state, and collision geometry are `null` and appear as
readiness issues.

## Electrical state

The user-confirmed controller remains a PCA9685 with an external 6 V/10 A
supply. That rating is present in the profile, while power validation remains
false. The exact SG90 variants must explicitly permit 6 V; their label or
manufacturer data takes precedence over generic SG90 specifications. The new
joint-to-channel map and pulse endpoints are empty, so the driver cannot start.

This is also why the code does not carry the old channel 0–3 map into the new
arm. A channel number can be reused after physical confirmation, but reuse is a
calibration result rather than a property of the CAD.

## Software changes

Robot schema version 5 now stores `actuator_model` on each joint and permits
unknown payload, frame, axes, limits, and homes during design intake. Motion
readiness requires all of them. Error messages and component-count validation
are model-neutral, and the CLI reports the mixed-servo inventory plus the 3MF
inspection.

The Fusion exporter no longer searches for the retired component names. It
exports every assembly occurrence and only produces canonical link meshes when
the imported assembly exposes the five configured link names. It also requires
the four configured joint names before setting `complete: true`. The merge
copies joint axes as operational axes and accepts limits only when Fusion also
provides a rest position.

The PCA9685 driver remains data-driven. It receives channels, direction, pulse
endpoints, and bounds from the validated profile and has no MG996R-only logic.
The same profile supports the planned second arm through a separate per-arm
calibration map.

## Remaining blockers

Before simulation training, export the assembled geometry and build the MuJoCo
model. Before physical motion, additionally complete:

- exact joint axes, origins, limits, homes, link lengths, and mesh scale;
- current arm build state and both arms' eventual channel maps;
- SG90 voltage compatibility and simultaneous-load power test;
- slow pulse endpoint and speed calibration for all four actuators;
- end-effector aperture, direction, force, and grip-friction measurements;
- camera intrinsics and camera-to-base calibration;
- guarded payload tests focused on SG90 elbow deflection; and
- independent stop/output-enable validation.

Simulation training can begin after the digital twin exists. Its outputs still
cannot establish real pulse endpoints, backlash, camera extrinsics, electrical
brownout behavior, printed fit, or physical payload capacity.
