# Baseten semantic router Chain

This CPU Chain implements the paper-style frozen MoIRA router. The Pi first
filters the catalog by a typed interface or exact capability and sends only
compatible, allow-listed component IDs. When that pool contains more than one
specialist, the Chain embeds the task plus each specialist's natural-language
description and representative routing phrases with
`sentence-transformers/all-MiniLM-L6-v2`, then returns the ID whose best
prototype has the highest cosine similarity. A one-item compatibility pool is
returned directly because there is no semantic choice to make.

There is no task-to-component route table and no trained routing head. Add or
replace a compatible specialist by editing its metadata in
`specialists.json`; the frozen router weights do not change. The caller must
supply `context.routing_text` whenever more than one component is eligible.
Unknown IDs, missing routing text, model failures, and invalid selections are
errors. This path has no automatic fallback.

Keep `specialists.json` synchronized with
`examples/physical_ai_specialists.json`; the test suite enforces byte-equivalent
JSON data. Install the Baseten/Truss tooling in a deployment environment,
authenticate, and run:

```bash
cd deploy/baseten_router
truss chains push router.py --dryrun
truss chains push router.py --promote
```

The Chain returns a component ID, routing strategy, model ID, and similarity
scores. It never returns a URL, credential, motor command, or model output. The
Pi validates the returned ID against its own catalog before invocation, and
local safety and motor authority remain on the robot.
