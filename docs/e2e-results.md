# End-to-end validation

Run date: 2026-09-19. Windows, Python 3.12.10, CPU-only PyTorch 2.14.0,
Transformers 5.17.0, Sentence Transformers 5.7.0, PEFT 0.21.0.

## Diagnosis

The original E2E run found a reproducible semantic routing failure. MiniLM sent
“Travel toward the right end of the line.” to the `left` specialist in every
serving mode. Direct comparison against PyTorch's cosine similarity confirmed
that the implementation was numerically correct:

| Candidate | Cosine similarity |
| --- | ---: |
| `left` | 0.5907557267 |
| `right` | 0.5889922977 |

The incorrect lead was only `0.0017634289`. The two expert descriptions were
themselves extremely close (`0.9915475692` cosine similarity). Pure embedding
argmax therefore lacked enough semantic separation to reliably distinguish the
opposing directions. Before the fix, MiniLM completed 21/24 episodes; SmolLM2
completed 24/24. Adapter loading and policy execution behaved correctly after
the wrong MiniLM selection, confirming that the defect was isolated to routing.

## Fix

The production default is now `HybridRouter`. It computes MiniLM scores first
and delegates to SmolLM2 when the top-two gap is below `0.05`. When users do not
supply few-shot examples, it dynamically turns each expert description into one
neutral labeled demonstration. This gives the small LM the label mapping and
required output format without task-specific keyword rules. The demonstrations
refresh when registry metadata changes.

The margin is an ambiguity heuristic, not a calibrated probability. `0.05` is
an implementation default and should be tuned on representative deployment
tasks. The hybrid strategy is an extension that composes the paper's two routing
options. Explicit `--router embedding` preserves the paper's pure cosine-argmax
baseline and still reproduces the original failure.

For the unchanged failing instruction, CLI output now records:

- `expert_id: right`
- `strategy: hybrid_prompt`
- `embedding_expert_id: left`
- `margin: 0.0017634289`

An invalid or unavailable LM response aborts routing before an episode opens;
the ambiguous embedding choice is never silently used as a fallback.

## Fixed E2E result

The fixed suite runs two trained LoRA specialists through a multi-step synthetic
environment. It loads the saved JSON manifest and adapter checkpoints, uses real
pretrained text models, routes once per episode, forwards the unchanged
instruction, and verifies learned actions, termination, task success, and
adapter load/eviction behavior. Router responses and policy actions are not
mocked.

| Router | Instruction set | Resident | Disk | Multi-adapter |
| --- | --- | --- | --- | --- |
| Hybrid (no user examples) | Literal | 4/4 | 4/4 | 4/4 |
| Hybrid (no user examples) | Paraphrased | 4/4 | 4/4 | 4/4 |
| SmolLM2 (few-shot) | Literal | 4/4 | 4/4 | 4/4 |
| SmolLM2 (few-shot) | Paraphrased | 4/4 | 4/4 | 4/4 |

All 48 fixed E2E episodes pass across the three serving modes. A separate
regression test first proves that raw MiniLM still selects `left`, then requires
the zero-example hybrid to select `right` for the exact same instruction. It also
checks unseen right-hand/left-hand wording and negated directions.

The final PCA9685 robot-configuration audit, including the recorded 6 V/10 A
servo-supply rating, arm #1 channel map, and unbuilt arm #2 state, reports all
157 tests passed, including the 13 pretrained E2E tests, in 106.41 seconds. JUnit output is in
`outputs/e2e-deep-audit.xml` (SHA-256
`37033b0ac989ac49bc9912afbb6570ea52a25f8e7e7911caf710bbbf2ddff002`).
The regular suite reports 144 passing unit/integration tests with 13 pretrained
E2E cases skipped unless explicitly enabled. Its JUnit output is in
`outputs/unit-deep-audit.xml` (SHA-256
`eb7da9c99ae1ea9c026b908402efdd4e356978cd7425d8abb71a66ef4f3c363c`).
The final wheel was installed into an isolated target and ran `physical-demo`
outside the repository; it includes the bundled current-arm metadata. The wheel
is `outputs/wheel-arm1-single-control-final-20260919/moira_robotics-0.1.0-py3-none-any.whl`
(SHA-256
`ecca25075cf1470cb30b4f4c0515334735ad6f15807e9fe57f0c3728feb4faca`).

## Additional checks

Both real models ran through `python -m moira evaluate` on the four illustrative
`examples/routing_samples.jsonl` entries and returned accuracy 1.0, macro-F1 1.0,
and zero invalid predictions. Direct CLI regression checks confirm that explicit
embedding mode selects `left`, while the zero-example default hybrid selects
`right`.

The 3,422,777,952-byte SmolLM2 checkpoint was verified against SHA-256
`f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8`.
Pinned model revisions are:

- MiniLM: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- SmolLM2: `31b70e2e869a7173562077fd711b654946d38674`

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,embeddings,prompt,adapters]"
.\.venv\Scripts\python.exe -m pytest tests/test_pretrained_e2e.py --run-e2e -q -o junit_logging=all --junitxml=outputs/e2e-fixed.xml
```

Use `-k hybrid` or `-k prompt` to select a router. The test environment is a
synthetic horizontal line designed to make a wrong expert observable. It does
not simulate robot dynamics, vision, or manipulation and does not establish
GR1/LIBERO performance or physical-robot readiness.
