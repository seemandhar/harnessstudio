# scale

SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models
(Choi, Ahn et al., ICML 2026, arXiv:2602.04208; official code github.com/snumprlab/scale, MIT, commit b4ad2a6).

- **What is reproduced:** the whole method, for OpenVLA. Per action token, the self-uncertainty
  `u = E_p[log(q_high / q_low)]` over the 256 action-token logits sets the sampling temperature
  `tau = T0 * sigmoid(u)` (near-greedy below 1e-3); per step, the EMA of the step-level uncertainty sets the
  vision-encoder attention temperature `gamma = kappa ** tanh(s * mean(u_{t-1} - EMA_{t-2}))` applied to every
  attention layer of both ViTs. One forward pass with a KV cache, no training, no verifier.
  Paper defaults (official `configs/scale.yaml`): `T0 1.0, epsilon 1e-12, num_logits 256, kappa 2.0, alpha 0.8,
  attn_sensitivity 0.3`.
- **Where it runs:** inside the policy server. `hs_scale.decoder:ScaleDecoder` is the algorithm (arithmetic copied
  from the official `modeling_prismatic.py`); `harnessstudio/policies/openvla_hs.py` runs the token loop,
  applies gamma to the vision encoder and keeps the per-episode uncertainty history. The harness side is K = 1
  and the `sampler` slot; the derived null turns the sampler off (greedy), so baseline and null are the native
  OpenVLA policy.
- **Fidelity `faithful`:** token-level agreement with the official `predict_action(decoding_mode="scale")` on
  the same checkpoint, inputs and seed is in `docs/results/scale.md`. Disclosed differences (not substitutions):
  one torch generator per (decision, seed) instead of one global stream; vla-eval's OpenVLA preprocessing and
  attention kernel (shared by all arms).

```bash
uv pip install --python .venv/bin/python -e methods/scale
hs verify scale --tier 0
hs verify scale                     # tier 1 on the fake stack (fake-t2 accepts and ignores samplers)

# policy server (GPU), method package on its PYTHONPATH so it can import the decoder:
CUDA_VISIBLE_DEVICES=5 PYTHONSAFEPATH=1 PYTHONPATH=$PWD:$PWD/methods/scale \
  vla-eval serve --config hs2-gpu/configs/openvla_libero10_hs.yaml --address 127.0.0.1:8115
hs eval scale --config hs2-gpu/configs/libero10_scale.yaml --policy-url ws://127.0.0.1:8115 \
  --contract libero --profile openvla-libero-hs --out runs/scale
```
