# EXT openvla-vision-token-hook: server-side vision-token hooks on the OpenVLA LIBERO server

Status: MERGED into main by fast-forward 32c86a6 -> 5a2ce3b on 2026-11-05T06:01:29Z (code commit 2d8ba3d; 5a2ce3b is
docs only), after G1/G2 passed and with `scripts/regress.py --inflight` empty; method package `methods/ttf` (hs-ttf 0.1.0)
added on branch docs/ttf-results. Reviewed by critic-ttf-1 (READY FOR GPU PRE-CHECKS) and used by the sealed TTF-VLA
primary (critic-ttf-final: ACCEPT WITH CONDITIONS); results: `docs/results/ttf_vla.md`. Run
`repro-next-20261105`, task conv-ttf-port-1 (rmh_converter). Proposal and evidence:
`runs/repro-next-20261105/ext/ttf/EXT_PROPOSAL.md`.

## 1. Paper mechanism
TTF-VLA (arXiv:2508.19257v3, github.com/PKU-XLab/TTF-VLA @ afbd6bf) changes OpenVLA's visual tokens between the
fused vision backbone (DINOv2 + SigLIP, 256 patches each) and the projector. At non-keyframe steps a patch reuses the
previous step's token unless its grayscale pixel difference exceeds a threshold or it is among the top-k patches of the
previous step's text-to-vision attention (LLM layer 15, prefill). It needs per-episode state (previous pixel values,
previous fused tokens, previous prefill attention, step counter) and a decode with `output_attentions=True`.

## 2. Why the existing hooks cannot express it
- `decoding_hooks` (openvla_hs.py) change token selection and the ViT attention temperature; they never see or
  replace patch tokens and get no LLM attention.
- `residual_hooks` (openvla_simpler_star_hs.py) rewrite decoder-layer residual updates on the SIMPLER server.
- Observation slots can edit pixels, not tokens, and cannot read the LLM's attention.

## 3. The change (new files except where noted)
| File | Change |
|---|---|
| `harnessstudio/policy_ext/vision_token_hooks.py` | NEW. Protocol (`VISION_TOKEN_HOOK`, `OUTPUT_ATTENTIONS`, `ATTENTION_LAYERS`, `bind`, `init_state`, `vision`, `after_decode`), `check_hook`, `vision_forward_hook` (swaps `vision_backbone.forward` for one request, restores in `finally`), `generate_with_hook` (the TTF release's predict_action call). |
| `harnessstudio/policies/openvla_ttf_hs.py` | NEW. `HarnessOpenVLATTFServer(HarnessOpenVLAServer)`: capability `vision_token_hooks`; no sampler or a decoding-hook sampler -> parent unchanged; a vision-token hook -> hooked path under the parent's inference lock; per-(session, episode, sampler) state through the parent's `_states` (dropped at EPISODE_START/END); one seed, no readouts; `activation_log` required (one JSON record per hooked request: tokens, float64 action + hex, seeds, input sha256, hook params, the hook's info, shape). |
| `harnessstudio/profiles/openvla-libero-ttf.yaml` | NEW. openvla-libero-hs + `vision_token_hooks`. |
| `harnessstudio/core/capabilities.py` | EDIT: register `vision_token_hooks` (+ docstring). |
| `harnessstudio/core/method.py` | EDIT (one hunk, `validate`): a sampler may also be justified by `vision_token_hooks`. |
| `harnessstudio/profiles/fake-t3.yaml` | EDIT: declare `vision_token_hooks` (route a, as EXT openvla-residual-hook did): the fake server ignores samplers, so tier 1 of such a method checks plumbing only. |
| `tests/test_openvla_logits.py` | EDIT: one amended assertion in `test_new_profiles` (fake-t3 minus fake-t2 now also excludes `vision_token_hooks`). |
| `tests/test_ext_openvla_vision_token_hook.py`, `tests/toys/hs_ext_vision_toy.py` | NEW. V1 static (capability, profiles, parent pins, validate), V2 protocol, V3 generate_with_hook on a fake model, V4 server dispatch/state/log/refusals. |
| `docs/extensions/openvla-vision-token-hook.md`, `docs/extensions/README.md` | NEW doc; one backlog line. |

## 4. Numerics of the hooked decode (a property of the reproduced release, not of this extension)
The hooked request calls `generate(..., output_attentions=True)` because the TTF release decodes every arm that way
(its baseline too, "to make sure fair comparison"). With transformers 4.40.1 and OpenVLA's default SDPA attention,
`LlamaModel._update_causal_mask` returns `None` for an all-ones mask in the prefill and `LlamaSdpaAttention` falls back
to the eager forward, which then applies no causal mask: the prefill is bidirectional. Verified on CPU
(`runs/repro-next-20261105/parity/c1_noncausal_prefill.json`; on the real OpenVLA classes in `e2e_parity.json`,
`openvla_mask_path`). The server reproduces this call; native requests keep the stock causal SDPA decode.

## 5. Compatibility claim
- Every parent stays byte-identical: `openvla_hs.py`, `policy_ext/{fake,protocol,mixin}.py`, `context/*.py` and the
  other R6 pins of `tests/test_ext_openvla_residual_hook.py`; checked again by V1.
- Native and baseline requests of the new server are the parent's code path (no hook installed, V4).
- No installed method's tier 0/1 changes except that methods requiring `vision_token_hooks` now get fake-t3.
- Regression gate on the exact code commit 2d8ba3d (critic-ttf-1, clean worktree, caches disabled): pytest 106 passed /
  40 skipped; 84 of 86 methods PASS tier 0/1; the 2 failures (tube-steer-vfb, -ortho) are an untracked asset
  (`visual_memory.npz`) absent from every worktree, and both pass on main. Re-run on the docs-only head:
  `runs/repro-next-20261105/ext/ttf/regress_<head>.json` (`git_head` = that commit). An earlier run
  (`regress.json`) was made on the uncommitted tree and is not evidence for any commit.

## 6. Flagged edits for the diff review
- `tests/test_openvla_logits.py::test_new_profiles`: amended assertion (fake-t3 gains `vision_token_hooks`).
- `core/method.py`, `docs/extensions/README.md` and `docs/writing-a-method.md` are also edited by
  `ext2/episode-feedback`: method.py merges cleanly (disjoint hunks); README backlog line and writing-a-method
  paragraph conflict textually: keep both. Merge order (critic-ttf-1): this branch first by fast-forward of main after
  G1/G2; episode-feedback after the TTF primary run (it changes server/eval/pairing code the TTF run imports).

## 7. GPU checks (evaluator, before any scored episode)
- G1: native path of `openvla_ttf_hs` bitwise equal to the stock vla-eval server (actions over a few episodes).
- G2: tier 2 with `ttf-shadow` vs `ttf-paper-baseline` all M (any X blocks).
- G3: per-step cost of the `output_attentions` path (critic-select-1 C9).
- G4: activation records present for every hooked request; fusion rate per suite near the paper's 41-45%.
