# Four-DOF desktop arm

This is the active MoIRA robot profile. Its primary source is
`Sharma_Ishaan_robotAssem.SLDASM` in the supplied
`4+DOF+Robotic+Arm+(SG90,+MG996R,+Arduino)` directory. The accompanying
`Robotic+Arm.3mf` contains the printable meshes and slicer layout.
`source_manifest.json` pins the assembly, all 16 CAD dependencies, and the 3MF
by byte length and SHA-256 digest.

The design has four actuators:

| Joint | Function | Servo |
| --- | --- | --- |
| `J1_BASE_YAW` | turntable yaw | MG996R |
| `J2_SHOULDER` | shoulder pitch | MG996R |
| `J3_ELBOW` | elbow pitch | SG90 |
| `J4_END_EFFECTOR` | wrist/gripper lever | SG90 |

MoIRA treats the first three as positioning joints and the fourth as the
end-effector aperture command. The original design description calls the last
stage a wrist/gripper mechanism; its exact servo-to-jaw travel must be measured
on this build before execution.

The profile is intentionally motion-disabled. Values inherited from the
retired arm were removed, including its joint limits, 25 g payload rating,
Y-up frame, channel assignment, and installed-arm state. The new configuration
will not command a servo until its geometry, joint limits, PCA9685 mapping,
pulse endpoints, speed limits, power compatibility, payload, collision model,
and stop path have been validated.

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
3. Create or rename the five top-level moving link components to
   `fixed_base`, `turntable`, `upper_arm`, `forearm`, and `end_effector`.
4. Create revolute joints named `J1_BASE_YAW`, `J2_SHOULDER`, `J3_ELBOW`, and
   `J4_END_EFFECTOR`. Set measured limits and a safe rest position for each.
5. In **Utilities → Scripts and Add-Ins**, add the entire
   `tools/FusionExportRobot` directory and run `FusionExportRobot`.

The exporter writes every occurrence mesh and all available transforms,
bounds, mass properties, parameters, and joints. It marks the export complete
only when all five canonical links and four named joints are present. This
prevents a convenient but incorrect mapping from silently becoming the robot
model.

Merge a complete export with:

```powershell
.\.venv\Scripts\python.exe .\tools\merge_fusion_robot_export.py `
  .\robot_models\four_dof_desktop_arm\model.json `
  .\four_dof_desktop_arm_export\fusion_robot_export.json `
  .\robot_models\four_dof_desktop_arm\model.json
```

The merge fills link dimensions, joint origins, axes, and enabled Fusion joint
limits, then copies collision meshes. It leaves explicit scale and parent-frame
validation blockers. Do not clear those blockers until the exported bounds
match physical measurements.

## Physical calibration still required

The existing PCA9685 and external 6 V/10 A supply are recorded, but the new
arm's wiring is not. Confirm the actual SG90 label permits 6 V before connecting
the servo rail. Then record, for each physical arm:

- installed state and PCA9685 channel for every joint;
- safe lower, upper, and home pulse widths found at low speed;
- measured joint velocity and end-effector travel;
- camera-to-base transform and workspace boundary;
- payload and elbow deflection under guarded lift tests; and
- output-enable or power-cut stop behavior.

The second arm can use the same model profile with its own channel and pulse
calibration. Bimanual hardware remains disabled until both installations are
present and the measured shoulder spacing is recorded.
