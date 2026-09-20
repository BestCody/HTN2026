# Charlie calibrated vision

This Truss deploys Grounding DINO Tiny as an open-vocabulary object detector. The
request must provide object labels, metric object dimensions, and a calibrated
image-to-robot-plane homography for every camera. It deliberately rejects an
uncalibrated request rather than estimating fake monocular depth.

The response matches `PerceptionInput -> WorldState` in `src/moira/contracts.py`.
