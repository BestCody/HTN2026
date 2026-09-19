# Engineering audit

Audit date: 2026-09-19. The review covered routing, model backends, registry
mutation, serving, PEFT adapter state, specialist training, evaluation, the CLI,
tests, packaging, and documentation.

## Resolved findings

| Severity | Area | Finding | Resolution |
| --- | --- | --- | --- |
| High | Adapter state | A failed activation or policy reset could leave stale episode state available to inference. | Activation and reset now clear state before calling fallible hooks and only commit the new state after success. |
| High | Concurrency | Lazy model loading, routing caches, and policy sessions had races. The resident server also treated two IDs pointing to the same policy object as independent. | Model inference and router state are locked. Registry snapshots are stable across route-and-lookup. Resident policies receive per-object leases, while independent policy objects may run concurrently. |
| High | Adapter preload | A multi-adapter preload failure left the successful prefix resident, and shutdown stopped at its first unload error. | Preload validates the complete request and rolls back adapters added by a failed call. Shutdown attempts every unload, retains accurate state for failures, and reports all affected IDs. |
| High | Training output | Checkpoints were written directly to their final directory, so interrupted saves could publish partial adapters. | Training saves to a sibling staging directory and renames it only after a complete PEFT save. Existing nonempty output directories are rejected. |
| Medium | Routing output | Public routing records accepted inconsistent IDs, duplicate or non-finite scores, and invalid margins. Prompt responses were not explicitly size-bounded. | Routing records validate their invariants. LM responses are limited to 4,096 characters and one final bounded integer selection. Non-text responses fail as routing errors. |
| Medium | API consistency | A controller could combine a router and a different registry, allowing an ID to be selected from metadata the controller did not own. | Registry-aware routers must share the controller's exact registry instance. Hybrid routing holds one registry snapshot across both stages. |
| Medium | Generation | The prompt backend defaulted to a larger output budget and did not check the model's context window before generation. | Generation defaults to 64 new tokens, checks model and tokenizer limits, validates chat messages, and uses a safe available pad token. |
| Medium | Training isolation | Configuration accepted booleans and non-finite numeric values, global RNG streams were changed, and malformed losses reached lower-level PyTorch failures. | Configuration and collaborators are validated, CPU/CUDA RNG streams are restored, and losses must be finite differentiable floating-point scalars. |
| Medium | Evaluation | Classification reports discarded the full routing decision. Rollouts accepted non-finite rewards and malformed metadata. | Reports include decisions and strategy counts. Rollouts validate seeds, limits, environment metadata, rewards, and result invariants. |
| Low | CLI behavior | Router-specific options could be silently ignored, making experiments differ from their command line. | Incompatible flags now produce a usage error. Hybrid thresholds and generation limits preserve explicitly supplied values before constructor validation. |
| Low | Input diagnostics | Invalid manifests and backend protocol objects often failed with indirect attribute or conversion errors. | Constructors and manifest loading now provide boundary-specific errors with the failing entry or protocol named. |

## Design after the pass

One immutable `Expert` record is selected per episode. Routing and metadata
lookup share a registry lock, then release it before the long-running policy
session. The episode retains that selected record, so later registry changes
only affect later episodes.

`InMemoryServer` gives each distinct policy object its own non-blocking lease.
This permits parallel episodes on independent resident backbones while
preventing aliases from concurrently mutating the same policy. `AdapterServer`
has one exclusive episode lease because all adapters share mutable backbone and
episode state. Its metadata uses a separate state lock so observations such as
`loaded_ids` are coherent.
Both servers return session-scoped policy handles that reject inference after
the context closes, preventing a retained handle from bypassing its lease.

The hybrid router first obtains normalized embedding scores. A single expert or
a top-two margin at or above the configured threshold finishes without loading
the LM. An ambiguous score invokes the prompt router against the same registry
snapshot. Any unavailable model, invalid response, or out-of-range selection
aborts before a policy session opens.

## Verification boundary

The automated suite covers public API validation, cache invalidation, registry
mutation, routing ambiguity, CLI behavior, adapter disk and multi-resident
switching, preload rollback, concurrent session leases, real PEFT training and
reload, local Transformers generation, evaluation metrics, and closed-loop
pretrained routing with learned adapter actions.

The final physical-audit run completed 144 normal tests, with 13 pretrained E2E
tests skipped. The opt-in run completed all 157 tests. The E2E cases exercise 48 complete episodes across resident,
disk-swapped, and multi-adapter policy serving; measured details are in
[E2E results](e2e-results.md).

## Raspberry Pi and component-routing extension

The production architecture treats the Pi 4B as an edge gateway rather than a
host for every model. `ComponentRegistry` filters candidates by exact typed
capabilities and authority. The frozen semantic router ranks the compatible
specialists from their text descriptions, and Baseten model or Chain endpoints
perform inference. Router and inference errors propagate; the hackathon path
has no automatic component substitution.

The layered `PhysicalAI` flow adds camera perception, scene-grounded DUM-E-style
voice NLP, persistent personal and workspace context, 6-DoF grasp generation,
robot-specific action-chunk policies, local IK/trajectory/collision validation,
and parallel structured 2–3 second world-model predictions. Task reward and
hard safety consume those predictions before plan selection. Shared local
tactile inference provides contact, slip, force, and grasp stability; outcome,
failure, and load models close the feedback loop. Remote components are
prohibited from holding motor, IK, motion, tactile-reflex, or hard-safety
authority. See
[Pi 4B and Baseten architecture](pi4-baseten-architecture.md).

The active SolidWorks arm is now hash-pinned and gated by validated robot-model
metadata. Tests cover every CAD/3MF digest, read-only 3MF inspection,
incomplete-model refusal, per-joint MG996R/SG90 inventory, and Fusion-export
merge. The new profile has no payload rating until this assembly passes a
guarded physical lift test. The 25 g result from the retired arm is no longer
used as robot configuration.
See the [arm audit](four-dof-desktop-arm-audit.md).

The repository still does not contain GR00T/OpenPI policy wrappers, the paper's
training data, original metadata, trained production weights, a physical
robot-state sensor adapter, or a certified independent emergency-stop implementation.
The margin is an uncalibrated heuristic, and an LM can still make a valid-format
but semantically wrong selection. Deployments need representative calibration,
checkpoint-to-backbone compatibility checks, policy action limits, collision
avoidance, and an independent stop path. The synthetic E2E suite establishes
software integration behavior; it does not establish benchmark reproduction or
physical-robot readiness.

## Deep physical execution audit

The 2026-09-19 physical pass traced untrusted model output through local motor
control and repaired the following faults:

- infeasible IK can no longer enter trajectory planning; it marks only that
  candidate unsafe so another feasible candidate may still be selected;
- policy chunks must preserve every selected step's arms, object, and total
  duration, while allowing multiple bounded chunks per high-level step;
- every trajectory point is labeled with its action chunk, and the controller
  sends a driver only that chunk's time-local segment;
- kinematics must cover every policy chunk and commanded arm with stable joint
  counts, and execution requires measured, timestamped robot state;
- trajectory limits apply to both samples and carried state, and calibrated
  per-joint velocity limits reject commands that move too far for their time;
- driver telemetry must identify the commanded arm and step; command deadlines
  stop both arms, and a failed emergency stop does not prevent attempting the
  other arm;
- local payload safety is computed for each chunk's actual arm count, so a
  later bimanual step cannot hide an earlier single-arm overload;
- exact world-model capability routes must return the corresponding model kind,
  and outcome, memory, reward, safety, plan, and control identities are checked;
- forged router decision metadata, non-standard NaN JSON, contradictory safety
  reports, and malformed boundary objects now fail before physical authority;
- model-backed IK rejects perception in a coordinate convention that differs
  from the calibrated robot frame;
- the arm configuration now owns bimanual spacing, control rate, clearance,
  gripper aperture/force/speed, local slip/collision/stability limits, and
  controller timing; robot-backed factories cannot override them with literals;
- guessed clearance, friction, object dimensions/mass, target poses, and
  gripper widths were removed from the physical path, and missing observations
  now fail closed;
- the packaged Pi wheel contains the same arm metadata as the canonical config,
  while the runnable demo identifies its remaining numeric geometry as an
  offline generic fixture;
- the hardware layer creates drivers only for physical arms marked installed,
  converts validated joint/gripper trajectories through per-channel pulse
  calibration, rejects frequency or mapping mismatches, and latches emergency
  stops until an explicit rearm; the replacement arm has no active channel map
  until its wiring is confirmed.

The arm gate now requires more than a successful Fusion export. Motion readiness
also requires validated mesh scale, parent-link joint frames, kinematics,
collision geometry, actuator mapping, payload, and joint velocity limits. STL
coordinates are explicitly treated as unitless until their bounds are checked.

The remaining physical blockers are real inputs rather than silent software
defaults: the Fusion export must still be run, payload and elbow deflection must
be measured, servo pulse endpoints and speeds must be calibrated, collision
geometry must be validated, and an independent stop path must be physically
tested. The Baseten identifiers remain deployment placeholders and
the offline state-space models remain deterministic integration fixtures.

The complete model and dataset disposition is recorded in the
[robot model training audit](training-audit.md).
