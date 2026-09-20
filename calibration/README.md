# CO6 camera calibration

The CO6 is attached to the Raspberry Pi and served to the RTX laptop through
the authenticated Pi endpoint. Calibration runs on the laptop, so OpenCV has
GPU-computer resources while the Pi only captures native MJPEG frames.

1. Mount the camera rigidly with the complete arm and tabletop in view. Do not
   move the camera after calibration.
2. Print `co6_checkerboard.svg` at **100% / Actual Size**. Measure its 100 mm
   verification line. If the line is not exactly 100 mm, use the measured square
   size with `--square-size-m`.

   The current physical print measures **25.4 mm (1 inch) per square**. That
   measurement is recorded in `co6_checkerboard_measurement.json` and is the
   default used by the calibration tool.
3. Run intrinsic calibration:

   ```powershell
   .\.venv-training\Scripts\python.exe tools\calibrate_co6_camera.py intrinsics
   ```

   Hold the board at varied positions, distances, and angles. Capture twelve
   sharp views that cover the center, edges, and corners of the image. The run
   is rejected if reprojection RMS exceeds 1.5 pixels.
4. Lay the board flat in the robot workspace. Align its rows with the robot-base
   X and Z axes. Measure the red `ORIGIN` inner corner relative to the base-yaw
   axis in metres. Then run:

   ```powershell
   .\.venv-training\Scripts\python.exe tools\calibrate_co6_camera.py extrinsics `
     --origin-x-m MEASURED_X `
     --origin-z-m MEASURED_Z `
     --table-height-m MEASURED_TABLE_Y
   ```

   The preview marks OpenCV corner zero in red. It must match the printed red
   `ORIGIN` arrow. Rotate the board if needed, or use `--reverse-corners` when
   the two endpoints are reversed. The visible checkerboard interior becomes
   the initial calibrated demo workspace; keep the block and tray inside it.

The command writes `config/co6_camera_calibration.json` only after both solves
pass their reprojection thresholds. Moving the camera, changing resolution, or
changing the robot mount invalidates the calibration.
