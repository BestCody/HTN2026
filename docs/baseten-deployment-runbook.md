# Baseten deployment runbook

The Raspberry Pi calls Baseten over HTTPS. It keeps memory, calibrated
kinematics, trajectory and collision validation, hard safety, and PCA9685 motor
authority locally. Baseten hosts independently addressable specialist models
and Chains. A missing specialist is an error; production does not substitute a
fixture or a general model.

## Current verified state

| Component | State | Evidence / blocker |
| --- | --- | --- |
| API access | Ready | The Management API accepts the ignored local API key. |
| Whisper Large V3 Turbo | RTX LAN, live-tested | CUDA transcription exactly recovered the Kokoro-generated pick/place command. |
| Kokoro-82M | RTX LAN, live-tested | The pinned CUDA model produced a valid 24 kHz WAV response. |
| MiniLM semantic router | Existing production Chain is live-tested; replacement checkpoint is local | The accepted fine-tuned checkpoint scored 100% in typed manipulation and world pools. It is staged locally but has not been repushed to the billable Chain. |
| Deterministic physical planner | Production Chain, live-tested | It generated three single-arm candidates and selected the highest-scoring safe simulation. |
| GLM vision and grounded voice NLP | Baseten Model API, live-tested | Strict structured outputs detected the red block/blue tray and grounded both roles without inventing a destination. |
| Grasp, policy, dynamics, world, reward, outcome | Pre-training integration stand-ins only | Their real endpoints require calibration or trained checkpoints. The stand-ins exist only in the non-actuating integration profile. |

Run the read-only inventory at any time:

```powershell
.\.venv\Scripts\moira.exe baseten-status
```

It reports configuration and live deployment state without printing API keys
or entity IDs. `SCALED_TO_ZERO` is healthy and will cold-start on the next
request.

## Deployment order for the single-arm demo

1. Start the RTX voice service and complete one real microphone WAV transcription.
2. Verify the production Router and Planner Chains.
3. Verify GLM Model API scene perception and grounded NLP.
4. Run `tools/run_software_integration.py` with execution disabled.
5. Calibrate the CO6 camera and Arm #1.
6. Deploy the arm-constrained grasp specialist after camera-to-base calibration
   and gripper limits exist. It may propose a 6D pose, but only orientations and
   positions executable by the current robot are admissible.
7. Train and deploy the arm #1 waypoint policy from simulation plus physical
   calibration data.
8. Package calibrated MuJoCo as the rigid-world component, then train and
   deploy the short-horizon dynamics residual from rollout errors.
9. Deploy the task reward and outcome verifier after their observable success
   definitions and evaluation examples are fixed.

The grasp, waypoint, dynamics, and world deployments cannot be made truthful by
creating blank Baseten models. Their weights or physics parameters must match
the calibrated arm, camera, gripper, and workspace. Arm #2 policy and bimanual
world deployment remain optional until the second physical arm exists.

## Current service commands

The deploy-only environment is pinned at `.venv-deploy` and ignored by Git.

RTX voice service:

```powershell
.\.venv-training\Scripts\python.exe tools\run_rtx_voice_server.py
```

After it becomes active:

```powershell
.\.venv\Scripts\python.exe tools\test_baseten_stt.py path\to\microphone-command.wav
```

Router validation and production deployment:

```powershell
$env:PATH = (Resolve-Path .\.venv-deploy\Scripts).Path + ';' + $env:PATH
Push-Location deploy\baseten_router
truss.exe chains push router.py --dryrun --non-interactive
truss.exe chains push router.py --promote --wait --non-interactive
Pop-Location
```

Store the returned Chain ID only in the ignored `.env`, then test it:

```dotenv
BASETEN_ROUTER_CHAIN_ID=<chain-id>
BASETEN_ROUTER_ENVIRONMENT=production
```

```powershell
.\.venv\Scripts\python.exe tools\test_baseten_router.py
```

Baseten's special development endpoints are
`/{development}/predict` for models and `/development/run_remote` for Chains.
Named and production environments use `/environments/<name>/...`. The runtime
constructs these forms explicitly.
