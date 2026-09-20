# Baseten specialist router Chain

This CPU Chain packages the project-trained MiniLM bi-encoder. The laptop first
filters the catalog by a typed interface or exact capability and sends only
compatible, allow-listed component IDs. For a multi-model pool, the Chain
embeds the task and each specialist's metadata, normalizes each specialist's
prototype centroid, and returns the highest cosine score. A singleton pool is
selected directly because there is no semantic choice.

There is no task-to-component route table or classification head. Training uses
triplet loss to move project task language toward the correct specialist
metadata and away from the hardest compatible specialist. Unknown IDs, missing
routing text, model failures, and invalid selections are errors; there is no
automatic model fallback.

The accepted checkpoint is generated under the ignored `checkpoints/` tree.
Stage it into the ignored deployment bundle before either dry-run or push:

```powershell
.\tools\stage_router_checkpoint.ps1
```

The deployment must contain `deploy/baseten_router/model/model.safetensors` and
`moira_training.json`. `router.py` loads only that local checkpoint with offline
model loading; it does not download or substitute the original MiniLM weights.
Keep `specialists.json` synchronized with
`examples/physical_ai_specialists.json`; tests enforce equivalent JSON data.

```bash
cd deploy/baseten_router
truss chains push router.py --dryrun
truss chains push router.py --promote
```

The Chain returns a component ID, routing strategy, model ID, and similarity
scores. It never returns a URL, credential, motor command, or model output. The
laptop validates the returned ID against its own catalog before invocation;
local safety and motor authority remain on the robot.
