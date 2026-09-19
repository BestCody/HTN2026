# Current lightweight arm

`model.json` records only facts recoverable from `Current_Lightweight_Arm.f3d`
without inventing missing geometry. The source archive is hash-pinned and the
configuration deliberately reports `motion_ready: false` until Fusion exports
the exact link dimensions, joint origins, and collision meshes and the physical
assembly is calibrated. A raw export does not clear readiness: STL scale,
parent-link joint frames, kinematics, collision geometry, actuator mapping,
payload, joint velocity, dual-arm mount spacing, gripper aperture/force/speed,
control frequency, clearance, local safety thresholds, and controller timing
must each be validated.

The design contains three positioning joints (`J1`-`J3`) and one gripper
actuator (`J4`). MoIRA therefore generates three arm-joint targets and treats
gripper width as a separate actuator command.

## PCA9685 wiring and calibration

The confirmed controller is a PCA9685. The production driver is implemented in
`src/moira/pca9685.py`, and the optional Raspberry Pi packages are installed
with `pip install -e ".[hardware]"`.

Connect the Pi's 3.3 V logic, ground, SDA, and SCL to the PCA9685 `VCC`, `GND`,
`SDA`, and `SCL`. Power the servos through the PCA9685 `V+` terminal using the
confirmed external regulated **6 V, 10 A** supply, and join the supply ground to
Pi/PCA ground. Do not supply the MG996R motors from the Pi's 5 V rail. The
rating is recorded in `model.json`; `servo_power_validated` deliberately remains
false until voltage, polarity, common ground, and behavior under simultaneous
servo load have been checked on the assembled robot.

`model.json` now records the actual hardware inventory. Physical `arm_1` is
installed in the primary `left` control slot with base, shoulder, elbow, and
gripper signals on PCA9685 channels 0, 1, 2, and 3 respectively. `arm_2` is
recorded as not built. Here, `left` is the software control slot used by
single-arm plans; it does not claim a calibrated spatial mounting position.
The planner removes plans that require an uninstalled control slot, and the
dual-arm driver refuses initialization while `arm_2` is absent.

Before motion, record and validate:

- the detected I2C address;
- the measured PCA9685 reference clock and chosen servo PWM frequency;
- the PCA9685 channels for any subsequently installed actuators;
- the pulse width at each calibrated lower and upper endpoint;
- the separate servo supply under simultaneous load; and
- an output-enable or equivalent stop path that has been physically tested.

Reversing a servo is represented by reversing its two calibrated pulse
endpoints. The driver never assumes a channel number, pulse range, or direction.
The implementation follows the
[NXP PCA9685 register/frequency specification](https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf)
and [Adafruit Raspberry Pi wiring guidance](https://learn.adafruit.com/16-channel-pwm-servo-driver?view=all).

## Export exact geometry

1. Open `Current_Lightweight_Arm.f3d` in Autodesk Fusion.
2. Open **Utilities → Scripts and Add-Ins**.
3. Use the **+** command to add `tools/fusion_export_robot.py`, then run it.
4. Choose this repository as the export location. The script writes
   `current_lightweight_arm_export/fusion_robot_export.json` and five STL link
   meshes.
5. Merge the export into the checked configuration:

   ```powershell
   .\.venv\Scripts\python.exe .\tools\merge_fusion_robot_export.py `
     .\robot_models\current_lightweight_arm\model.json `
     .\current_lightweight_arm_export\fusion_robot_export.json `
     .\robot_models\current_lightweight_arm\model.json
   ```

6. Validate it:

   ```powershell
   .\.venv\Scripts\moira.exe robot-model-check `
     .\robot_models\current_lightweight_arm\model.json --verify-source
   ```

The merge removes the missing-export blocker and adds explicit blockers for the
unitless STL scale and parent-link joint-frame validation. Mechanical fit,
payload, actuator mapping, joint and gripper limits, bimanual mount spacing,
collision geometry, safety thresholds, controller timing, and calibration
blockers remain until they have been measured on the assembled arm. Do not fill
these fields from a generic servo datasheet; record measurements from this exact
assembly.
