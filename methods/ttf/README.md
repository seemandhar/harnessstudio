# ttf: TTF-VLA (Temporal Token Fusion) for OpenVLA (HarnessStudio v2 method)

## What TTF-VLA does

TTF-VLA is arXiv:2508.19257v3 (AAAI 2026); the code is PKU-XLab/TTF-VLA @ afbd6bf (MIT). At non-keyframe steps,
OpenVLA reuses the previous step's visual token for every patch that meets both conditions:
- it is static: its grayscale pixel difference is at most 0.03;
- it is outside the top-70 patches of the previous step's text-to-vision attention (LLM layer 15, prefill).

Every third policy step is a keyframe, where everything is recomputed.

## How this package expresses it

This is a K = 1 slot method. Its server-side `sampler` is the vision-token hook `hs_ttf.server_hook:TTFVisionHook`.
It needs the capability `vision_token_hooks` from EXT openvla-vision-token-hook (server
`harnessstudio/policies/openvla_ttf_hs.py`, profile `openvla-libero-ttf`); see
`docs/extensions/openvla-vision-token-hook.md`.

## Entry points

| Entry point | What it runs | Role |
|---|---|---|
| `ttf` | hybrid fusion, K = 3, τ = 0.03, top-k 70, layer 15, text attention | candidate |
| `ttf-paper-baseline` | the release's `aptube_enabled=False` path: stock tokens, decoded with `generate(output_attentions=True)` | the paper's baseline arm |
| `ttf-shadow` | the full TTF computation, with stock tokens sent to the projector | compute-matched overhead control; must equal `ttf-paper-baseline` bitwise |

## Fidelity: `adapted`

- The substitutions are listed in `hs_ttf/paper.yaml`.
- Under transformers 4.40.1, the release's decode call takes a prefill with **no causal mask**. Every paper arm
  reproduces that, and it is not fixed.
- The framework's derived null (no sampler) is the native causal decode. It is a determinism control only.

## Code identity

- This package is byte-identical to the scored package of run `repro-next-20261105` (`method/`, MANIFEST
  `a611af04…`, 8 files).
- Fingerprints: ttf `c507c724…`, ttf-paper-baseline `c125ad52…`, ttf-shadow `78f54245…`.
- Since then the card has changed only by one appended changelog string. The fingerprints are unchanged.
- The cards' header comments and the pyproject description still say "DRAFT". Those are frozen bytes of the scored
  package.

## Results (2026-11-05; details in `docs/results/ttf_vla.md`)

- **Headline: not reproduced (non-matching baseline regime; adapted).**
  - It was measured on the released-code (paper-faithful) decode path, and that path's paper-baseline arm does not
    reproduce the paper's baseline: 35.0/46.0/40.5/5.0% vs 82.0/66.5/77.0/48.0%.
  - Over 800 paired LIBERO cases: paper-baseline 253/800 → ttf 248/800, D −0.62 pp, Newcombe 95% CI
    [−3.92, +2.67], exact McNemar p 0.766.
  - The result does not test TTF in the paper's ~68% baseline regime.
- **The mechanism was active:** fusion rates were 0.44-0.46 (paper 0.41-0.45), and actions diverged in 793/800
  episodes. ttf-shadow is action-identical to paper-baseline (M 800 / R 0 / X 0).
- **Cost:** about 60.35 GPU-h (primary 57.55); $0 paid; 0 NRP calls. Run:
  `runs/repro-next-20261105` (RoboMetaHarness-claude).
