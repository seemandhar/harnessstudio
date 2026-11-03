# fork30 (hs_fork30)

> **Status: NEGATIVE - not advanced** (run scalepp-20261102, see `runs/scalepp-20261102/REPORT.md` in RoboMetaHarness-claude).
> tier 2 passed (2026-11-03 02:23 UTC), but its bound 101-job development screen never ran (the run's Claude session hit the account spend limit) and the C3 selection window had passed, so it has no screen result and no Campaign standing. Its threshold, feature and fork time were selected in-sample on all 100 development scenes.
> Kept as a labelled record of a tried idea; it is not a recommended method and has no Campaign result.
> Committed in the scaleagent-20261103 housekeeping task; package files otherwise unchanged.


Round-4 candidate of RoboMetaHarness run scalepp-20261102 (cue from `analysis/fork-headroom.json` rep30_ge6; review
`reviews/fork-r4-review.json` F1-F4). New training-free mechanism (fidelity tf_variant): native greedy OpenVLA; at decision
30, if R = #{k in 1..29 : executed a[k] == a[k-1]} >= 6, decode with SCALE (`hs_scale.decoder:ScaleDecoder`, paper defaults,
read-only, guarded by the decoder sha256 619d9b96 and the scale fingerprint 73123974) for every later decision; otherwise
greedy. One policy call per decision; send_state false; shadow null; no sham (review). Threshold/feature/fork time were
selected in-sample on all 100 development scenes (disclosed).

```bash
uv pip install --python .venv/bin/python -e methods/fork30
.venv/bin/hs verify fork30 --tier 0 && .venv/bin/hs verify fork30
(cd methods/fork30 && ../../.venv/bin/python -m pytest -q tests)
.venv/bin/python methods/fork30/scripts/fire_set.py --out ../runs/scalepp-20261102/eval/fork30/fire_set.json
.venv/bin/python ../runs/scalepp-20261102/eval/fork30/make_plan.py --eval-dir ../runs/scalepp-20261102/eval/fork30/screen
```
