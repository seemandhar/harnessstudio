# Run X (scaleagent-codex-20261103): NEGATIVE development results

**Status: NEGATIVE. Both methods were dropped at their predeclared development pilots. Neither method ever
entered a frozen Campaign phase, and no method was exposed to validation (seed 7 init 15-24) or final
(seed 7 init 25-49) scenes.** These are valid negative development results, not infrastructure failures.
They are not a benchmark score and must not be cited as one.

Setting: frozen OpenVLA `openvla-7b-finetuned-libero-10` (checkpoint 80970322) on LIBERO libero_10, official
horizon 520. Benchmark: `libero_rgbd` contract through the adopted RGB-D benchmark (`libero_subtask_rgbd`, Docker
image sha256:3946acda…, vla-eval 0.8.0). Policy server: pinned serving JSON
`robometaharness/integrations/harnessstudio_v2/policies/openvla-libero10.json` (probe 23d4ad8ab4182245).
The goal was to beat SCALE (arXiv:2602.04208) with a training-free harness whose runtime planner is codex
(gpt-6.1-sol). Both methods decode with SCALE at every step, using the exact sampler of the plain `scale` method
under the same common-random-numbers key. Runtime: GPUs 4-7.

## 1. witnessed-rescue (x-r1) — `methods/witnessed_rescue`, NEGATIVE

**Mechanism.** The controller watches public proprio for grasp-failure triggers: H2 empty close and H1 settled
unlifted clamp. The trigger rules were ported from the SCALE++ development replays and checked 400/400 against them.
On a trigger:
1. The simulator pauses.
2. Codex points at the instruction's target.
3. The harness runs a geometric witness gate on metric depth, then a same-frame marked-crop confirmation by codex.
4. The harness runs an analytic grasp with a measured custody test (gripper width plus lift).
5. Codex confirms custody.
6. The harness places the object analytically into an open-top destination (after a free-column check), or hands back to SCALE.

**Reviews.** Proposal: `runs/scaleagent-codex-20261103/reviews/proposal_r1_review.json`. Implementation:
`r1_impl_review.json`. Fixes: fix1 (`impl/witnessed_rescue/fix1/`) and fix2b (`fix2_review.json`,
`fix2b_k1_review.json`):
- non-vacuous free-column check;
- unique evidence paths;
- the card declares `send_state`;
- the custody prompt compares same-window before/after crops, and an abstain falls back to measured custody;
- `min_steps_left` 170;
- the witness prompt carries the measured object size;
- K1: a drawer or other enclosed destination never receives an analytic place.

**Predeclared re-pilot** (criterion: `fix2_review.json` repilot_test; manifest hashed before the first
candidate episode). PASS required at least one SCALE-failed scene turned into a success, and no new loss.
Result (`eval/r1-repilot/repilot_result.json`): **FAIL — 0 rescues in 10 scenes.**
- **Guards that held:** 0 losses. Respondoff (the SCALE-shadow null) matched plain `scale` bitwise on 10/10
  scenes, and every unfired scene was bitwise SCALE. 17 planner calls, 0 transport failures.
- **Why no rescue:**
  - On t3 (bowl into drawer), custody was reached but the drawer destination forbids an analytic place, and the hand-back did not finish the task.
  - On t8i8, one moka pot was placed analytically, but the two-object task did not finish.
  - On t2, grasps did not keep custody.
  - On t2i7, the planner declined.
- **Earlier pilot:** `eval/r1-pilot/`; analysis in `analysis/r1_pilot_analysis.json`.
- **Decision:** `control/research/decision-drop-x-r1.json`.

## 2. reseat-reentry (x-r2) — `methods/reseat_reentry`, NEGATIVE

**Mechanism.** The trigger is Scientist C's post-release gap rule on proprio: a carry ends (release or drop),
then no new carry starts for 140 steps, with at least 150 steps left. The online rule is made causal by
deferring the fire while a width run is in progress. On a fire:
1. Codex compares the step-0 frame with the current frame and points at the remaining movable instruction object.
2. A witness call with the measured-size cue confirms the point.
3. The open gripper moves to hover 0.10 m above the object. The re-seat budget is `steps_left − 105`
   clamped to [45, 90] steps (review `r2_impl_review.json` ruling 2), and the gripper is never closed.
4. SCALE resumes with the original instruction.

Articulated remaining work (drawer, knob, lid and the like) gets no motion.

**Reviews.** Proposal: `reviews/proposal_r2_review.json` (conditions K1-K10). Implementation:
`reviews/r2_impl_review.json` (rulings 1 and 2, conditions C1-C8).

**K1 trigger parity** (frozen code replayed offline):
- Seed 9 init 5-9: 7/15 failures and 1/35 successes. These counts are exact; two fire steps move by +8 and +7
  steps because of the causal deferral.
- Seed 7 devbase: 33/132 failures and 4/168 successes, against C's 36/132 and 4/168. Three failure fires are
  deferred past the steps-left limit.

**Predeclared pilot** (`eval/r2-pilot/pilot_manifest.json` sha256 c8ceef06…, written before the first candidate
episode). P1 required at least 2 rescues. Result (`eval/r2-pilot/verdict.json`, `readout.json`):
**FAIL — 0 rescues out of 10 fired failures** (15 candidate and 12 respondoff episodes).
- **Guards that held:**
  - P2: 0 losses.
  - P3: every unfired episode, and every fired episode up to its fire step, matched the SCALE reference
    bitwise; the online first-fire step equalled the offline prediction; respondoff matched SCALE bitwise on 12/12.
  - P5: 5 re-seats were executed, all reached the hover pose, at most 39 primitive steps, 0 closes by the primitive.
  - Planner: 21 codex calls, all ok.
- **Outcome:** the re-seat delivered the arm above the right object, but the language-blind policy did not
  go on to grasp it.
- **Decisions:** `control/research/decision-drop-x-r2-end-search.json`. Binding condition K10 (x-r2 was the last
  admissible hypothesis) ends hypothesis search for this run.

## 3. Shared components (reusable)

- **codex planner operator:** `runs/scaleagent-codex-20261103/impl/codex_operator/codex_planner.py`
  (sha256 a37c8f1a…; vendored verbatim into both packages).
  - **Isolation:** each call is a `codex exec` in an empty scratch HOME with a read-only sandbox. Skills, MCP, project
    docs and web search are off. Any tool item fails the call. The model id and effort are checked from codex's
    trace telemetry on every call.
  - **Accounting:** each call is logged as one JSONL line with hashes only.
- **Campaign RGB-D integration extension** (r3, merged 2026-11-03T13:20Z):
  `runs/scaleagent-codex-20261103/ext/campaign_rgbd/MERGED.json` (patch sha256 aca97114…, three reviews). It adds
  `spec --observation rgbd` and `send_state`, with a fail-closed check of the card's `benchmark_params`.
- **Sealed SCALE arm:** `runs/scaleagent-shared/scale-final` holds 250/250 episodes (SEAL.json 009f96dd…). It is
  **sealed and unread**, and was not used by Run X because no full phase ran. Future runs may use it under
  COORDINATION rule 4 with its key conditions.
- **Instruction-following probe:** within a scene, the OpenVLA libero_10 fine-tune ignores its instruction
  (language-blind). This is why subtask prompting was not pursued.

## 4. Costs

- **GPU:**
  - x-r1: tier 2, first pilot, and the re-pilot (13:56-14:20Z, 4 servers).
  - x-r2: pilot wave 16:25:36-16:42:12Z (4 servers).
  - Per-run totals: **pending** (`runs/scaleagent-codex-20261103/eval/final_costs.json` not yet written).
- **Codex:** all calls used the ChatGPT login, so paid API cost was $0.
  - Pilot calls are counted in the reports above: 17 (x-r1 re-pilot) and 21 (x-r2 pilot).
  - Coder contract checks and replays: 2 + 10 + 1 calls.
- **Training:** no model was trained or fitted (trainable parameters 0).

## 5. Why both failed (summary)

The triggers were precise. They fired on 0/35 (x-r1) and 1/35 (x-r2) seed-9 SCALE successes, and both had low
regression risk; every bitwise guard held. The responses never converted a SCALE failure:
- **x-r1:** the analytic grasp/place chain lacked custody or was blocked by enclosed destinations.
- **x-r2:** the spatial re-seat did not make the language-blind policy pick up the remaining object.

The predeclared pilots before freeze stopped both hypotheses before any Campaign GPU time or held-out exposure.
