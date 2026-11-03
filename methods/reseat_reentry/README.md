# reseat-reentry (hs-reseat-reentry)

> **Status: NEGATIVE development result.** x-r2 pilot: 0 rescues / 10 fired failures (decision-drop-x-r2-end-search.json). Never frozen into a Campaign; no validation/final exposure. Kept as a labelled negative; see `docs/results/scaleagent-x.md`.

Run X (`runs/scaleagent-codex-20261103`) idea `x-r2-reseat-reentry`, approved with conditions K1-K10 in
`runs/scaleagent-codex-20261103/reviews/proposal_r2_review.json` (Scientist C's H1 merged with D's xd2). Frozen OpenVLA
`openvla-7b-finetuned-libero-10` on LIBERO libero_10 through the RGB-D benchmark (`libero_rgbd`, `send_state: true`).

- **Base decoding:** one `ctx.policy.sample(1, sampler=S)` per decision with the original instruction, `S` = the plain
  `scale` method's sampler dict; unfired episodes (and every fired one up to its fire) are bitwise plain `scale`.
- **Trigger** (`triggers_gap.py`, proprio only): after a carry ends (release or loss, as C's measured script), no new
  carry for W = 140 steps, steps_left >= 150, <= 2 fires, each carry fires once. Made causal: while a width run is in
  progress at the window end the fire waits until it lifts (no fire) or ends (fire). `scripts/gap_replay.py` runs the
  same code offline.
- **Response:** pause; codex call 1 (step-0 + current agentview, original instruction) names the remaining movable
  instruction object and a pixel, or `none`; depth back-projection (one replan only after a failed geometric/workspace
  check); call 2 = same-frame marked-crop witness with the measured-size cue; on confirm the OPEN gripper rises to
  max(z, table + 0.25), moves above the target and descends to its local top + 0.10 m (<= 45 steps, tolerance 0.02 m,
  orientation kept, never a close); SCALE resumes with the original instruction. Abstain / none / articulated part
  (planner `kind` or a generic fixture-word guard) / witness reject / transport failure / budget end -> no motion.
- **Planner:** isolated `codex exec`, gpt-6.1-sol, effort low, 60 s per call, <= 6 calls per episode, <= 3 per fire.
- **Entry points:** `reseat-reentry` (candidate), `reseat-reentry-respondoff` (SCALE-shadow null). The Campaign null is
  the framework's shadow null. The optional `-home` ablation is not built.
- **Evidence:** `<hs --out>/reseat_reentry_evidence/{packets,planner}` (per process/episode unique); accounting in
  `accounting.extra.reseat_reentry` (fires, escalations with every planner call, hover targets, re-seat legs,
  primitive steps, analytic share, `closes_commanded_by_primitive` = 0).

Vendored (with sha256 provenance in `hs_reseat_reentry/vendor/PROVENANCE.json`) from the witnessed-rescue package and
the r1 codex operator; that package is not imported, so the freeze closure is `methods/scale` only.

Tests: `cd harnessstudio/methods/reseat_reentry && ../../.venv/bin/python -m pytest -q -p no:cacheprovider tests/`
(seam tests through the real HarnessServer + K1 parity on the saved development traces), `hs verify NAME [--tier 0]`.
