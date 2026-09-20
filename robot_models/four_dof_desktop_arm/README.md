# Desktop arm CAD reference and active three-actuator build

The CAD reference is `model.json`. The active physical build is
`physical_three_actuator_model.json`. Both come from
`Sharma_Ishaan_robotAssem.SLDASM` in the supplied
`4+DOF+Robotic+Arm+(SG90,+MG996R,+Arduino)` directory. The accompanying
`Robotic+Arm.3mf` contains the printable meshes and slicer layout.
`source_manifest.json` pins the assembly, all 16 CAD dependencies, and the 3MF
by byte length and SHA-256 digest.

The CAD design has four actuators, but the physical elbow servo has failed and
holds the elbow rigidly. The active build therefore has three actuators:

| Joint | Function | Servo |
| --- | --- | --- |
| `J1_BASE_YAW` | turntable yaw | MG996R |
| `J2_SHOULDER` | shoulder pitch | MG996R |
| fixed elbow | rigid at the exported assembly pose | disabled; former channel 2 |
| `J3_GRIPPER` | gripper lever | SG90, channel 2 |

MoIRA exposes base yaw and shoulder pitch as the two positioning coordinates.
The coupled gripper is the third command. No runtime path can command channel 2
or create an elbow degree of freedom.

The profile remains motion-disabled. Arm #1 uses channels 0, 1, and 3 for base,
shoulder, and gripper. The Arduino calibration used the Adafruit PCA9685 driver
at 50 Hz with a 102-to-512 tick range; the exact mapping and observed commands
are in `config/arm1_servo_observations.json`. Motion stays locked until the
remaining speed, aperture, payload, collision, power, and stop checks complete.

## What the 3MF establishes

The checked 3MF uses millimetres and contains 12 unique printable meshes across
six slicer plates. Its mesh statistics report no slicer repairs. It does not
contain joints, and its build transforms place parts on print plates rather
than in the assembled robot. MoIRA therefore uses it as print geometry and
source evidence, never as kinematic transforms.

Inspect both the CAD manifest and 3MF report with:

```powershell
.\.venv\Scripts\moira.exe robot-model-check `
  .\robot_models\four_dof_desktop_arm\model.json --verify-source
```

## Fusion assembly export

1. Keep every `.SLDPRT` dependency beside
   `Sharma_Ishaan_robotAssem.SLDASM`, then import/open the assembly in Fusion.
2. Confirm that the imported assembly is assembled rather than laid out for
   printing.
3. Keep the imported component names and hierarchy unchanged. The checked
   `fusion_mapping.json` groups 37 non-duplicate occurrences into `fixed_base`,
   `turntable`, `upper_arm`, `forearm`, and `end_effector`; the nested SG90
   duplicate is deliberately excluded.
4. In **Utilities -> Scripts and Add-Ins**, add the entire
   `tools/FusionExportRobot` directory and run `FusionExportRobot`.

The exporter writes occurrence and per-body meshes plus transforms, bounds,
mass properties, parameters, and analytic cylindrical faces. The joint fitter
uses named CAD features from `fusion_mapping.json` to recover the four revolute
axes without manual component renaming, grounding, or Fusion joint creation.
It rejects missing or ambiguous features rather than substituting coordinates.

Fit the joint frames with:

```powershell
.\.venv\Scripts\python.exe .\tools\fit_fusion_joint_axes.py `
  .\four_dof_desktop_arm_export\fusion_robot_export.json `
  .\robot_models\four_dof_desktop_arm\fusion_mapping.json
```

The merge validates Fusion's unitless STL coordinates against metric occurrence
bounds, transforms every mapped part into its link frame, and combines the
parts into five canonical link meshes. It also divides the end effector into
fixed, primary-finger, and mirror-finger meshes using named CAD occurrences:

```powershell
.\.venv\Scripts\python.exe .\tools\merge_fusion_robot_export.py `
  .\robot_models\four_dof_desktop_arm\model.json `
  .\four_dof_desktop_arm_export\fusion_robot_export.json `
  .\robot_models\four_dof_desktop_arm\model.json
```

The merge fills link dimensions, joint origins, and axes. Joint limits remain
empty until measured on the physical build.

Generate and validate the active fixed-elbow MuJoCo hierarchy with:

```powershell
.\.venv\Scripts\python.exe .\tools\generate_mujoco_model.py `
  .\robot_models\four_dof_desktop_arm\physical_three_actuator_model.json `
  .\robot_models\four_dof_desktop_arm\physical_three_actuator_validation.xml

.\.venv-training\Scripts\python.exe .\tools\validate_mujoco_model.py `
  .\robot_models\four_dof_desktop_arm\physical_three_actuator_validation.xml `
  --output .\robot_models\four_dof_desktop_arm\physical_three_actuator_validation.json `
  --robot-model .\robot_models\four_dof_desktop_arm\physical_three_actuator_model.json `
  --visual-reviewed

.\.venv-training\Scripts\python.exe .\tools\render_mujoco_validation.py `
  .\robot_models\four_dof_desktop_arm\physical_three_actuator_validation.xml `
  .\robot_models\four_dof_desktop_arm\physical_three_actuator_validation.png `
  .\robot_models\four_dof_desktop_arm\physical_three_actuator_motion_validation.png
```

This MJCF is intentionally limited to kinematic validation. It uses zero
gravity and has no actuators or invented limits. The two CAD-derived gripper
hinges are linked by a -1:1 gear equality, so they counter-rotate from the one
physical `J3_GRIPPER` command. The forearm body has no elbow joint in the active
MJCF. Training dynamics must wait for physical
mass, inertia, friction, servo response, and joint-limit data.

## Physical calibration still required

The existing PCA9685 and external 6 V/10 A supply are recorded. Arm #1 is
installed on channels 0, 1, and 3; channel 2 is disabled. Confirm the actual SG90 label permits 6 V
before connecting the servo rail. Then record:

- safe lower, upper, and home pulse widths for Arm #1, found at low speed;
- measured joint velocity and end-effector travel;
- camera-to-base transform and workspace boundary;
- payload and elbow deflection under guarded lift tests; and
- output-enable or power-cut stop behavior.

The second arm can use the same model profile with its own channel and pulse
calibration. Bimanual hardware remains disabled until both installations are
present and the measured shoulder spacing is recorded.

## Calibration and training records

The checked-in [`config/arm1_servo_calibration.json`](../../config/arm1_servo_calibration.json)
is prefilled with the confirmed channels and leaves every measured value null.
Fill it only with observed values. A completed record is schema-checked against
the robot ID, CAD hash, installed physical arm, channel map, PWM period, joint
limits, home positions, speed, voltage compatibility, power test, and stop test.

Apply it atomically to both model copies with:

```powershell
.\.venv\Scripts\moira.exe servo-calibration-apply `
  .\robot_models\four_dof_desktop_arm\physical_three_actuator_model.json `
  .\config\arm1_servo_calibration.json `
  --output .\robot_models\four_dof_desktop_arm\physical_three_actuator_model.json
```

This command records the calibration hash and still leaves physical output
disabled when collision, payload, workspace, or other safety inputs are absent.

The CO6 record is in
[`config/co6_camera_calibration.json`](../../config/co6_camera_calibration.json).
After both calibration files are complete, `episode-dataset-init` creates an
immutable-lineage training dataset. Each published episode contains monotonic
timestamps, hashed camera frames, synchronized robot states and actions, a
train/validation/test split, and the exact robot, servo, and camera hashes.

The dynamics form is in
[`config/simulation_training.json`](../../config/simulation_training.json).
Check what still blocks physics rollout generation with:

```powershell
.\.venv\Scripts\moira.exe simulation-training-preflight `
  .\robot_models\four_dof_desktop_arm\physical_three_actuator_model.json `
  .\robot_models\four_dof_desktop_arm\physical_three_actuator_validation.xml `
  --servo-calibration .\config\arm1_servo_calibration.json `
  --camera-calibration .\config\co6_camera_calibration.json `
  --simulation-config .\config\simulation_training.json
```
