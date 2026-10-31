# fast-harness: FastHarness (arXiv:2610.31427) as a HarnessStudio controller

"Experience-Driven Review Scheduling for Vision-Language-Action Policies". Code: github.com/FastHarness/fast-harness
@ `dfe070f` (MIT). Project page: fastharness.github.io.

A frozen policy (pi0.5) proposes a fresh action chunk every decision; an LLM reviewer supervises it. A **fast path**
releases chunks without a review once a stage has a record of reviewed successes (Wilson lower bound -> review
interval, commit spans, random audits); unfamiliar / ambiguous / contact / near-past-failure moments are always
reviewed. A **recovery path** reviews every chunk after an error (repair in place -> return to the latest verified
waypoint -> a reviewer-selected earlier one, bounded), then routine skipping resumes. Failure experience is kept.

Fidelity: **adapted** (see `hs_fast_harness/paper.yaml` substitutions and `docs/results/fast_harness.md`).

| File | Ports |
|---|---|
| `hs_fast_harness/scheduler.py` | `scheduler.py` + `recovery.py` (line for line) |
| `hs_fast_harness/memory.py` | `memory.py` (same rules, in memory; JSON artifact instead of SQLite) |
| `hs_fast_harness/reviewer.py` | `reviewer.py` (instructions, strict Review schema, validation, retry feedback, images) |
| `hs_fast_harness/controller.py` | `engine.py` (the loop) on HarnessStudio's `ctx` |
| `hs_fast_harness/kinematics.py` | the unreleased RoboDojo runtime's FK / EEF takeover, from the X5A URDF |
| `hs_fast_harness/prepare.py` | cross-episode experience -> frozen `experience.json` (`hs prepare`) |

## What is adapted

- Reviewer: **gpt-6.1-sol** (OpenAI chat completions, strict `json_schema`) instead of GPT-6 Astra. Astra is never used.
  The card's operator needs `temperature: 1.0`; `max_tokens` is renamed to `max_completion_tokens` by a temporary shim
  in `controller.py` (core request 1 in the results doc).
- Protocol: experience is built by `hs prepare` on development episodes and frozen. **cold** = empty experience
  (every chunk reviewed, as the paper's first episode); **warm** = the frozen artifact. The paper accumulates across
  the evaluated episodes themselves.
- The model does not echo `request_id` (the harness binds it); no terminal audit review; compact request JSON;
  reviewer failures (transport, or 2 schema retries) execute the fresh student prefix unreviewed and force the next
  review instead of aborting.
- `edit` / `eef` takeovers: DLS IK on the URDF chain, linear joint interpolation over the response's steps (<= 5),
  gripper switched at once. The paper's runtime is not released.

## Run

```bash
uv pip install --python .venv/bin/python -e methods/fast_harness
hs verify fast-harness --tier 0 && hs verify fast-harness          # tier 1, CPU, scripted reviewer
```

RoboDojo (native Isaac Sim env with harnessstudio overlay, pi0.5 T1 server on :8311, key in the environment):

```bash
# development: cold candidate on layout group seed 1 -> frozen experience
hs prepare fast-harness --config methods/fast_harness/configs/robodojo_build_tower_dev.yaml --case-namespace seed1 \
   --policy-url ws://127.0.0.1:8311 --contract robodojo --profile robodojo-pi05-hs --no-docker --out runs/fh_prep
# held-out: baseline / null / cold candidate, then warm candidate
hs eval fast-harness --config methods/fast_harness/configs/robodojo_build_tower.yaml --policy-url ws://127.0.0.1:8311 \
   --contract robodojo --profile robodojo-pi05-hs --no-docker --out runs/fh_eval_cold
hs eval fast-harness --arms candidate --prepared runs/fh_prep ... --out runs/fh_eval_warm
python methods/fast_harness/tools/summarize.py cold=runs/fh_eval_cold/candidate warm=runs/fh_eval_warm/candidate
```

`configs/run_stage.sh`, `run_stage_d.sh` (chained development on a repeated layout) and `run_stage_c.sh` are the
exact drivers used on 2026-10-31 (GPU 4, overlay env `/bigdata/sejain/hs2-cache/envs/hs2_robodojo_env.sh`).
`tools/summarize.py` prints per-episode reviews / tokens / skip rate; `tools/check_port.py <fast-harness clone>`
re-runs the differential check against the original code.

Results and every deviation: `docs/results/fast_harness.md`. In short: tier 2 passes on RoboDojo build_tower; under
HarnessStudio's protocol (frozen experience, held-out layouts) warm vs cold reviewer tokens were -11% and -16% per step
on build_tower and +26% on stack_bowls, far from the paper's -57%: the stage features rarely match experience from
other layouts, and stored failure memory makes each review larger.

The key must only be in the child environment (`set -a; . <env file>; set +a` in the launching shell); it is never
written to the card, traces or logs.
