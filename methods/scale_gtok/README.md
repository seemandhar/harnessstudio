# scale-gtok

> **Status: NEGATIVE - not advanced** (run scalepp-20261102, see `runs/scalepp-20261102/REPORT.md` in RoboMetaHarness-claude).
> HOLD, hypothesis falsified on development scenes: it behaves as another SCALE re-roll (+13/-11 vs its CRN SCALE seed, exact McNemar p 0.84) and does not reduce gripper flicker (7.42 vs 7.60 flips per 100 steps; the flicker is state-induced).
> Kept as a labelled record of a tried idea; it is not a recommended method and has no Campaign result.
> Committed in the scaleagent-20261103 housekeeping task; package files otherwise unchanged.


SCALE (arXiv:2602.04208v2, faithful port `methods/scale`) with one token rule changed: OpenVLA's 7th action token
(the gripper) is decoded by masked argmax over the 256 action bins of the same SCALE logits, instead of SCALE's
uncertainty-tempered multinomial sample. Original RoboMetaHarness candidate of run `scalepp-20261102`
(GENERAL `scale-gtok` = DOMAIN `r1d-h1-gripmode-scale`), approved with required changes RC1-RC6 in
`runs/scalepp-20261102/reviews/proposals-r1-review.json`.

- **Slots:** proposer `hs_scale:SingleSample` (K = 1, reused); sampler
  `hs_scale_gtok.decoder:ScaleGripperArgmax` with SCALE's paper defaults
  (T0 1.0, epsilon 1e-12, num_logits 256, kappa 2.0, alpha 0.8, attn_sensitivity 0.3). Derived null = sampler off =
  native greedy (bitwise the baseline), as for `scale`.
- **What stays SCALE's:** tokens 1-6 (`ScaleDecoder.select_token`), every u_k, the reported tau_k (also for token 7,
  RC1), gamma and the EMA state. Token 7's logits are conditioned on SCALE's sampled motion prefix and
  gamma-tempered vision attention, so this is neither `component` nor "the native greedy gripper".
- **Random stream (RC3):** the server seeds one fresh generator per (decision, seed) and token 7 is last, so
  skipping its multinomial changes no other draw; under case namespace `''` the candidate is bitwise the
  eval/devbase SCALE record until its first gripper override.
- **Token-position counter (RC2):** reset in `attention_temperature` (once per `_decode`, before the token loop of
  `openvla_hs.py` e7c6a4c3, line 186); `select_token` raises before a reset, beyond `action_dim` calls, or on
  batched logits. This refuses the unmerged openvla-ksample K > 1 per_seed/batched paths. Pinned by
  `tests/test_server_pin.py` (server file sha256 e7c6a4c3..., `hs_scale/decoder.py` 619d9b96...).
- **Fidelity:** `tf_variant` of SCALE; the substitution text and disclosures (RC6: "deterministic mode-of-bins rule
  for the gripper token", not a K->infinity majority-vote limit; the DOMAIN blipdrop2 result D-F1) are in
  `hs_scale_gtok/paper.yaml`.
- **Server import path:** the OpenVLA server must import `hs_scale_gtok` and `hs_scale`. In a Campaign the frozen
  `hs2.server_pythonpath` (this root + the referenced root `methods/scale`) is appended after site-packages by
  `robometaharness/integrations/harnessstudio_v2/server_boot.py`. For screens use the same mode
  (`runs/scalepp-20261102/impl/impl-scale-gtok/launch_gtok_server.py`). Never install into the server's uv env and
  never `uv run` openvla_hs.py.

```bash
uv pip install --python .venv/bin/python -e methods/scale_gtok
.venv/bin/hs verify scale-gtok --tier 0
.venv/bin/hs verify scale-gtok            # tier 1 (fake-t2 ignores samplers: the decoder is tested below)

# decoder + real server seam (pinned openvla_hs.py on a fake model) + witness tests; CPU, nothing installed:
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 \
PYTHONPATH=$PWD:$PWD/methods/scale:$PWD/methods/scale_gtok \
  /home/sejain/.cache/uv/environments-v2/openvla-hs-73218eaab6c13068/bin/python -m unittest discover -s methods/scale_gtok/tests -v

# mechanism witness (RC4) + activation metric for evaluated arms (paired with the eval/devbase map records):
.venv/bin/python methods/scale_gtok/scripts/witness.py --candidate OUT/candidate [--null OUT/null] --out witness.json
.venv/bin/python methods/scale_gtok/scripts/witness.py --self-check-map --out selfcheck.json   # 3.90 / 7.68
```

The exact tier-2 and 40-scene screen commands are in `runs/scalepp-20261102/impl/impl-scale-gtok/report.json`.
