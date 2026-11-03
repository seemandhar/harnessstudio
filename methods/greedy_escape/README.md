# greedy-escape

> **Status: NEGATIVE - not advanced** (run scalepp-20261102, see `runs/scalepp-20261102/REPORT.md` in RoboMetaHarness-claude).
> DROP after its development screen: the escape windows rescue only early freezes (6/16 rescued when firing by step 70; 0/38 when firing later).
> Kept as a labelled record of a tried idea; it is not a recommended method and has no Campaign result.
> Committed in the scaleagent-20261103 housekeeping task; package files otherwise unchanged.


New training-free mechanism (RoboMetaHarness run scalepp-20261102; proposal `runs/scalepp-20261102/ideation/idea-r1g.json`,
review `runs/scalepp-20261102/reviews/proposals-r1-review.json` GE1-GE7). Not a reproduction of any paper.
Policy: OpenVLA `openvla-7b-finetuned-libero-10` on the `openvla-libero-hs` server (`harnessstudio/policies/openvla_hs.py`).

- **What it does:** a controller that decodes natively (greedy, `sampler=None`) every step and watches two public
  stall signals: (a) 15 identical executed actions with max|a[0:6]| < 0.01; (b) the 3rd empty close (close held
  >= 10 steps, finger width states[6]-states[7] <= 4 mm 10 steps after the close) since the last window. A fire opens a
  40-decision escape window decoded by the frozen escape operator (`escape` param; default SCALE,
  `hs_scale.decoder:ScaleDecoder` at the paper defaults, exactly the `scale` method's sampler), then greedy again.
  At most 4 windows per episode. Rules, re-arming and missing-proprio handling: `hs_greedy_escape/monitor.py`.
- **Exactly one `ctx.policy.sample` per decision**, so window decision d uses the CRN seed of the SCALE map record at d.
  Untriggered episodes are the native greedy requests bitwise.
- **Fidelity `tf_variant`** (substitutions in `paper.yaml`): SCALE only inside windows; SCALE's EMA/gamma state starts
  at the first window decision and persists across windows.
- **Null: shadow.** The controller runs and pays its calls on the shadow session; native actions execute.
- **Benchmark param needed:** `send_state: true` (the OpenVLA server does not request state; the monitor needs the
  finger width). Behaviour-neutral for OpenVLA; tier 2 checks it.
- **Activation evidence:** trace events `greedy_escape` (`phase` fire / window_end) and
  `accounting.extra["greedy_escape"]` (window starts, triggers, suppressed fires, escape decisions, empty closes).
- **Another escape operator** (e.g. scale-gtok) is a new card/entry point and a new frozen candidate (GE2).

```bash
uv pip install --python .venv/bin/python -e methods/greedy_escape
.venv/bin/hs verify greedy-escape --tier 0 && .venv/bin/hs verify greedy-escape
.venv/bin/python -m pytest -q methods/greedy_escape/tests
# evaluator: configs + exact argv for tier 2 / grid screen / shadow null / init 4-9 confirmation
.venv/bin/python methods/greedy_escape/scripts/write_screen_configs.py --eval-dir runs/scalepp-20261102/eval/greedy-escape
.venv/bin/python methods/greedy_escape/scripts/screen_summary.py --plan runs/scalepp-20261102/eval/greedy-escape/plan.json
```

The policy server needs only `hs_scale` importable (`PYTHONPATH=<hs2>:<hs2>/methods/scale` as for the SCALE map, or
`server_boot.py --append-path` roots in a Campaign); `hs_greedy_escape` runs in the harness process only.
