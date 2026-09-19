# Current lightweight arm audit

Source: `Current_Lightweight_Arm.f3d`  
SHA-256: `23eb62682a7ab3bf087719e6c215848484483c6941b31b7f69b02d8190699b0d`

The F3D archive was extracted read-only. It contains 32 payload files, ten B-rep
records, seven parameterized mesh records, five named moving link groups, and
four named as-built joint axes.

## Recovered design facts

- Coordinate convention: XZ ground plane, Y up.
- Actuation: four MG996R servos.
- Link groups: `01_FIXED_BASE`, `02_YAW_TURRET`, `03_UPPER_LINK`,
  `04_FOREARM_HEAD`, and `05_MOVING_FINGER`.
- Positioning joints: base yaw about Y, shoulder about Z, and elbow about Z.
- Gripper actuator: J4 about Y. It is not an additional wrist angle in IK.
- Base-yaw range: -65° to +65°.
- Design notes identify elbow range -70° to +30° and gripper range -6° to +13°.
- Target payload: 50 g per arm, pending physical validation.
- User parameters exist for `upper_length`, `forearm_length`,
  `shoulder_height`, `print_clearance`, and `target_payload`.

The binary archive exposes parameter names but not a stable public representation
of their evaluated values. Exact values and transforms are therefore exported
through Fusion's API rather than reverse-engineered from undocumented binary
records. Fusion reports design length values in centimetres, angles in radians,
and mass in kilograms.

## Corrected software assumptions

The previous offline fixture represented each arm as four positioning joints,
used 0.45 m links, assumed 0.8 kg per-arm capacity, and demonstrated a 1.2 kg
mug. Those values do not describe this design.

The current fixture now:

- emits three positioning angles per arm and a separate gripper command;
- reads the 0.05 kg design target from the arm configuration for prediction and
  feedback fixtures;
- uses a 0.08 kg demonstration object so bimanual handling is required;
- hash-checks the supplied F3D source;
- uses the arm's Y-up frame in model-backed IK and rejects Z-up perception;
- requires calibrated per-joint velocity limits before constructing motion
  components;
- requires measured dual-arm spacing, gripper aperture/force/speed, control
  frequency, clearance, safety thresholds, and controller timing instead of
  substituting generic values;
- fails closed when perception omits clearance, tactile input omits friction,
  or a policy omits measured dimensions, mass, pose, or gripper width;
- refuses model-backed physical motion until exact geometry and calibration are
  complete.

## Servo electronics

The assembled robot uses a PCA9685 servo controller and a confirmed external
regulated supply rated at 6 V and 10 A. Both values are represented in the
canonical and bundled robot configurations, so the hardware driver does not
substitute a generic supply assumption. The supply remains marked unvalidated
until its polarity, voltage at the board, common ground, and simultaneous-load
behavior are physically checked. Motion also remains blocked until the detected
I2C address, calibrated PCA9685 reference clock, PWM frequency, per-actuator
channel map, pulse endpoints, and output-enable stop behavior are recorded.

Physical `arm_1` is the only installed arm. Its base, shoulder, elbow, and
gripper signals are recorded on PCA9685 channels 0, 1, 2, and 3. It occupies the
software's primary `left` control slot so existing single-arm plans can address
it; this logical slot is separate from the still-unmeasured spatial mount.
Physical `arm_2` is explicitly marked unbuilt. Candidate plans that require the
unavailable control slot are removed, single-arm driver construction creates
drivers only for installed hardware, and bimanual driver construction is
blocked until both physical arms exist.

## Mechanical blockers recorded inside the CAD file

- The Futaba FSH-6S horn approximation says physical fit is unverified.
- The 24T module-2 gear pair says CAD verification is pending.
- Drive-pin fastener retention is not finalized.
- The 50 g payload target is pending physical validation.

These are hard readiness blockers. A successful software simulation does not
clear them.

## Export path

`tools/fusion_export_robot.py` uses Fusion's supported API to export:

- user-parameter expressions and evaluated database values;
- occurrence transforms, bounds, mass, volume, and centre of mass;
- joint axes, origins, positions, and enabled limits;
- one STL collision/visual mesh for each named link group.

`tools/merge_fusion_robot_export.py` converts those results into SI units,
updates the robot metadata, and copies the meshes. STL has no embedded unit, so
the merge records an unresolved mesh-scale blocker rather than assuming that a
consumer interprets the coordinates as centimetres. Exported joint frames also
remain blocked until they are validated in each parent-link frame. The model
remains `motion_ready: false` until those checks, joint/gripper calibration,
dual-arm mount measurement, payload testing, local safety calibration,
collision validation, controller timing, and the mechanical blockers are
resolved.

Autodesk references: [Fusion scripts and add-ins](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/WritingDebugging_UM.htm),
[STL export API](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/STLExport_Sample.htm),
[as-built joint transforms](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/fusion_AsBuiltJoint_transform.htm),
and [Fusion database units](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/Units_UM.htm).
