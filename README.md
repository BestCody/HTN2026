# HTN2026 Robotic Arm

A 4-DOF robotic arm — base rotation, shoulder, elbow, gripper — driven by
MG996R servos, controlled by a Raspberry Pi + PCA9685, and steered from a
browser dashboard with a live 3D view.

The full runbook (hardware wiring, SD card flashing, calibration, dashboard
build) is in **[SETUP.md](SETUP.md)**. Start there before touching a wire.

## Repo layout

```
firmware/       Python service that runs on the Raspberry Pi
  arm_controller/   FastAPI + WebSocket + PCA9685 driver
  config.yaml       Per-joint calibration (channel, angle limits, µs range)
  systemd/          arm-controller.service unit file
dashboard/      Vite + React + react-three-fiber control panel
  src/              ArmViewer (3D scene), JointControls, WS client
  public/models/    Drop STL exports of the arm here (see SETUP §6)
arm_model.f3d   Fusion 360 source assembly
SETUP.md        Full setup guide — wiring, SD card, install, calibrate
```

## Quick start (once you've followed SETUP.md)

On the Pi:

```bash
cd ~/HTN2026/firmware
source .venv/bin/activate
arm-controller             # or: sudo systemctl start arm-controller
```

On your laptop:

```bash
cd dashboard
npm install
npm run dev
```

Then open the dev URL, point the dashboard at `ws://arm.local:8000/ws`, and
hit **Enable**.

## Current scope

Wired for **one arm** (channels 0–3 on the PCA9685). The second arm will occupy
channels 4–7 and reuse the same firmware — extend `firmware/config.yaml` with
its four joints when the hardware is ready.
