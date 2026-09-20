# Charlie Whisper Large V3 Turbo

This Truss deploys the voice transcription specialist on one Baseten
`L4:4x16` instance. It accepts the exact JSON representation produced by
MoIRA's `SpeechInput` contract:

```json
{"audio":{"base64":"<audio file bytes>"}}
```

The endpoint returns `text`, language metadata, duration, and timestamped
segments. English transcription, VAD, deterministic decoding, and the
Large V3 Turbo checkpoint are fixed deployment properties for the hackathon
demo.

The package pins CTranslate2 and the CUDA 12 cuBLAS/cuDNN runtime libraries.
Their library directories are present in `LD_LIBRARY_PATH` when the model
server starts; omitting them causes a live L4 replica to fail at inference with
`libcublas.so.12` missing.

## Deploy from the repository root

```powershell
py -3.12 -m venv .venv-deploy
.\.venv-deploy\Scripts\python.exe -m pip install --upgrade pip truss==0.18.30
.\.venv-deploy\Scripts\truss.exe login --browser
.\.venv-deploy\Scripts\truss.exe push deploy\baseten_whisper --watch
```

The command creates a development deployment. Copy the model ID printed by
Truss or shown in the Baseten dashboard into the ignored `.env` file:

```dotenv
BASETEN_STT_MODEL_ID=<model-id>
BASETEN_STT_ENVIRONMENT=development
```

Test it with a WAV, MP3, or M4A recording:

```powershell
.\.venv\Scripts\python.exe tools\test_baseten_stt.py path\to\command.wav
```

Do not promote the model until the real microphone command set passes.
