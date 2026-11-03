# gpp-bdu (hs_gpp_bdu)

> **Status: NEGATIVE - not advanced** (run scalepp-20261102, see `runs/scalepp-20261102/REPORT.md` in RoboMetaHarness-claude).
> built and approved for tier 2, but never screened: after gpp-core's gate failed no admissible composition vehicle remained. No result exists for any of its entry points.
> Kept as a labelled record of a tried idea; it is not a recommended method and has no Campaign result.
> Committed in the scaleagent-20261103 housekeeping task; package files otherwise unchanged.


Round-3 bdu (blocked-descent unload) of RoboMetaHarness run scalepp-20261102 (proposal `ideation/idea-r3a.json`, review
`reviews/proposals-r3-review.json` BDU-RC1..RC5, R3-SEL). New training-free mechanism; screened only inside a composition
with its own lift-only sham.

- **Isolation:** `hs_gpp` v1.1 is imported by the running gpp-core / ge-gpp-core screens, so it is never edited. This package
  subclasses `hs_gpp.controller.GppController` (read-only) and overrides only `run()` (the parent loop plus bdu hooks).
  `check_parent_identity()` refuses to build unless hs_gpp's controller/rules/__init__/cards hash to the frozen v1.1 values
  and gpp-core (6c80527f), ge-gpp-core (91c88e6f) and greedy-escape (f03731f1) fingerprints match; greedy-escape's files
  stay pinned by hs_gpp's G3 guard.
- **bdu (BDU-RC1):** b_k = a_z < -0.2 AND gripper open AND realized dz > -0.5 mm; fire at 20 consecutive b. Window <= 10
  decisions: the policy's own action with a_z := +0.15 (sham: a_x = a_y := 0); ends on a policy close (executed natively),
  >= 2.5 cm lateral, or 10 steps; counter reset; <= 4 fires. Precedence (BDU-RC3): active GE window / primitive / bdu window
  first; then H2, H1, bdu, GE trigger; bdu not evaluated in GE windows (its counter resets there).
- **Entries:** `ge-gpp-bdu` / `ge-gpp-bdusham` (primary vehicle), `ge-bdu` / `ge-bdusham`, `gpp-bdu` / `gpp-bdusham`
  (cards generated from the frozen hs_gpp cards by `scripts/make_cards.py`).
- **Evidence:** `ctx.record("bdu", phase=fire|edit|end)`; bdu fires in `accounting.extra["gpp"]["fires"]` (module "bdu",
  frames fire/start/end/+10/+30, realized vs commanded lateral and cos for P1, P2 witness), `bdu_edited_steps`.

```bash
uv pip install --python .venv/bin/python -e methods/gpp_bdu
for m in ge-gpp-bdu ge-gpp-bdusham ge-bdu ge-bdusham gpp-bdu gpp-bdusham; do .venv/bin/hs verify $m --tier 0; .venv/bin/hs verify $m; done
(cd methods/gpp_bdu && ../../.venv/bin/python -m pytest -q tests)        # run separately from gpp / greedy_escape tests
.venv/bin/python ../runs/scalepp-20261102/eval/bdu/a5_bdu.py --out ../runs/scalepp-20261102/eval/bdu/a5_bdu.json
.venv/bin/python ../runs/scalepp-20261102/eval/bdu/make_plan.py --vehicle ge-gpp-bdu --eval-dir ../runs/scalepp-20261102/eval/bdu/ge-gpp-bdu
```
