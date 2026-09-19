# Baseten router Chain

This CPU-only Chain returns one component ID from the Pi's allow-list. It never
returns a URL, credential, motor command, or model output.

Install the Baseten/Truss tooling in a deployment environment, authenticate, and
run:

```bash
truss chains push --watch router.py
```

Set `BASETEN_API_KEY` and construct the Pi client with
`RemoteComponentRouter.from_baseten_chain(...)`. Update `ROUTES` when deployment
component IDs change. A missing route, unavailable Chain, or invalid selection is
an error; this hackathon path has no automatic fallback.

The static table covers scene perception, 6-DoF grasping, separate manipulation
skills, learned forward dynamics, five specialized world models, task reward,
outcome verification, and Pi-side IK, motion, tactile, safety, load, feedback,
and control components. Returning a local component ID is intentional: the Pi
still validates the router decision against its catalog and invokes that local
implementation. The router never moves local safety authority into the cloud.
