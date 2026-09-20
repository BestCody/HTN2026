# Four-DOF desktop arm audit

> This records the original four-actuator CAD audit. The active physical profile
> is `physical_three_actuator_model.json`; its elbow is rigid and channel 2 is
> disabled.

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

No replacement value was guessed. Fusion subsequently established the axes and
link lengths, and the user confirmed Arm #1's installed state and channel map.
Unknown limits, payload, pulse endpoints, and collision validation remain
`null` and appear as readiness issues.

## Electrical state

The user-confirmed controller remains a PCA9685 with an external 6 V/10 A
supply. That rating is present in the profile, while power validation remains
false. The exact SG90 variants must explicitly permit 6 V; their label or
manufacturer data takes precedence over generic SG90 specifications. Arm #1's
base, shoulder, elbow, and gripper are confirmed on PCA9685 channels 0, 1, 2,
and 3. Pulse endpoints remain empty, so the driver cannot start.

## Software changes

Robot schema version 5 now stores `actuator_model` on each joint and permits
unknown payload, frame, axes, limits, and homes during design intake. Motion
readiness requires all of them. Error messages and component-count validation
are model-neutral, and the CLI reports the mixed-servo inventory plus the 3MF
inspection.

The Fusion exporter no longer searches for the retired component names. It
exports every assembly occurrence and B-Rep body, then groups the original
occurrence names into five canonical links using `fusion_mapping.json`.
Analytic cylindrical faces provide named sources for the four revolute axes, so
manual Fusion joints are unnecessary. Missing or ambiguous source features fail
the fit. The merge validated the 0.001 m STL scale, recovered 154.14 mm and
100.10 mm link spacing, and produced five canonical link-local meshes plus
fixed, primary, and mirror gripper mechanism meshes.

The CAD-derived MuJoCo model compiles on MuJoCo 3.13 as seven bodies, seven
meshes, five hinge coordinates, and one gear equality. The equality couples the
two finger hinges at -1:1, leaving four commanded degrees of freedom. Its four
hierarchy motion checks, counter-rotation check, and 100-step zero-gravity
validation pass. This establishes the kinematic topology without claiming
calibrated physical dynamics.

The PCA9685 driver remains data-driven. It receives channels, direction, pulse
endpoints, and bounds from the validated profile and has no MG996R-only logic.
The same profile supports the planned second arm through a separate per-arm
calibration map.

## Remaining blockers

Before dynamics training or physical motion, complete:

- physical joint limits, homes, velocities, mass, inertia, friction, and servo response;
- Arm #2's eventual build state and channel map;
- SG90 voltage compatibility and simultaneous-load power test;
- slow pulse endpoint and speed calibration for all four actuators;
- end-effector aperture, direction, force, and grip-friction measurements;
- camera intrinsics and camera-to-base calibration;
- guarded payload tests focused on SG90 elbow deflection; and
- independent stop/output-enable validation.

Fusion omitted one non-meshable or hidden body under
`Sharma_Ishaan_rotateBase:1`; 36 of 37 mapped occurrences otherwise match their
metric bounds within 0.1 mm. Confirm that omitted body is non-physical before
marking collision geometry validated.

Kinematic dataset and rendering work can begin now. Physics policy and dynamics
training must wait for the physical values above. Simulation outputs cannot
establish real pulse endpoints, backlash, camera extrinsics, electrical
brownout behavior, printed fit, or physical payload capacity.
