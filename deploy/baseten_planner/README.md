# MoIRA physical task planner

This Baseten Chain accepts the same flattened fields emitted by
`CandidatePlanningInput` and `PlanSelectionInput`. It generates several typed
plans from grounded objects and grasp poses, then selects only a candidate marked
safe by the downstream 2–3 second simulators.

`planner_profile.json` records the current robot installation. It currently
allows only the installed left arm; adding arm 2 is a profile change after its
channels and calibration are supplied.
