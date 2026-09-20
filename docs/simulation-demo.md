# MExT checkpoint-backed digital-twin demo

This is the hardware-free judge path. It loads the CAD-derived fixed-elbow arm in
MuJoCo with a red cube and green platform, renders a real RGB observation, and
feeds that image into the same perception component used by the physical
workflow. Speech, scene grounding, specialist routing, candidate planning,
parallel future prediction, safety selection, confirmation, outcome
verification, and journaling retain their production contracts. Only the
camera and final arm driver are replaced by MuJoCo adapters.

The animation contains an overview and a camera attached to the moving arm.
Physical PWM is never initialized.

## Run

Start the laptop voice server and configure the same Baseten Router and Planner
Chain IDs used by the judge demo. Then run from the repository root:

```powershell
.\.venv-training\Scripts\python.exe tools\run_simulation_demo.py --live
```

By default, Kokoro generates "Move the red cube onto the green platform," and
Whisper transcribes that generated waveform before routing. To use a real voice
recording instead:

```powershell
.\.venv-training\Scripts\python.exe tools\run_simulation_demo.py `
  --audio .\command.wav `
  --live
```

Use `--open-animation` to open the saved result after the run. Without `--live`,
the pipeline remains suitable for headless recording.

## Judge evidence

The terminal shows every real component route, three candidate plans, their
2.5-second predicted futures, the selected safe plan, scene revalidation, the
MuJoCo control route, and post-action camera verification. Artifacts are saved
under `outputs/mext_sim_demo/`:

- `mext_simulation.gif`: overview and arm-POV execution;
- `initial_overview.jpg`: exact frame supplied to visual perception;
- `final_overview.jpg` and `final_arm_pov.jpg`: post-action evidence;
- `response.wav`: spoken completion response;
- `runs.jsonl`: typed route, plan, prediction, control, and verification trace.

This demonstrates software and model integration, including the accepted
fixed-elbow waypoint and 2–3 second dynamics checkpoints. It does not establish
calibrated real-world robot performance. MuJoCo control is explicitly identified as simulation in
the component route and terminal presentation. RGB detections come from the
rendered cameras; metric object positions come from the digital twin, analogous
to a calibrated depth/pose source on the physical path.
