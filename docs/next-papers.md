# First papers to implement in HarnessStudio v2

Pick papers that exercise different interfaces, so each one either fits what exists or tells us exactly
which interface to add. Sources: `gpt_report/readthis.md`, `gpt_report/compass_artifact_*.md`,
`RoboMetaHarness/knowledge/HARNESS_CATEGORIES.md`.

## Recommended order

| # | Paper | Category | Uses (exists today) | Still needed | Benchmark / policy | Reference code |
|---|---|---|---|---|---|---|
| 1 | **RTCF** (arXiv:2608.04527): progress-aligned retrieval of successful trajectories + clipped low-frequency DCT residual on the action chunk | memory_retrieval | `action_op`, `prepare` + `TrajectoryStore`, `bind(embodiment)` | progress alignment and DCT residual (method code only) | LIBERO (the paper uses pi0-FAST; vla-eval ships `pi0/libero_fast.yaml`) | none: equations only |
| 2 | **Harness VLA** (arXiv:2607.08448), zero-shot mode | agentic_orchestration + skill_composition | controller, LLM tool (NRP, `adapted`), `eef_delta` skills, `vla_act` | perception: RGB-D + camera parameters from the benchmark (spec 16.11, a HarnessStudio subclass of the LIBERO benchmark) and a SAM3 tool server; rotation skills | LIBERO-Pro, pi0.5 (RLinf `pi05_libero130` or openpi `pi05_libero`: pin one) | RPent (Apache-2.0), local copy at `cvpr27/RPent` |
| 3 | **FastHarness** (arXiv:2610.31427): experience decides when the expensive VLM review is needed | uncertainty_compute + agentic_orchestration | controller, LLM tool, `prepare` (cross-episode experience) | nothing new expected | pi0.5 | github.com/FastHarness/fast-harness (per the report; verify) |

Why this order: RTCF validates the development stage and memory path cheaply (CPU-only, no LLM);
Harness VLA is the flagship agent baseline and forces the perception interface; FastHarness reuses both.

## After these

- **VLS** (sampler_guidance): needs T2 guidance hooks in the GR00T server (velocity readout exists;
  on-step steering does not). pi0.5's flow loop is jitted, so GR00T first.
- **Knowing When to Stop** (execution_monitoring): attention readout (T3) exists in v1 `ext/attn`; port it
  into the GR00T server as a readout.
- **SCALE** (uncertainty_compute): needs a logits/attention-temperature hook in an autoregressive policy
  server (OpenVLA in vla-eval).
- **Gated GeoBoN** (world_model_verification): needs a world-action-model server with predicted futures.
- **RTC / FreeBar**: need live mode (not in v2.0).

## Per-paper checklist

1. `hs new <slug> --category ... [--kind controller --llm]`; fill `paper.yaml` (fidelity, substitutions,
   operators with `original_model`, artifacts with sha256 + licence).
2. Port the mechanism; `hs verify <name>` (tier 0 + 1) until PASS.
3. Small real run: `hs eval` on a predeclared subset (official horizons), check null attribution (0 X).
4. If reference code exists: run it on the same cases and compare pair by pair.
5. Full benchmark only after the subset shows an improvement against baseline and null.
