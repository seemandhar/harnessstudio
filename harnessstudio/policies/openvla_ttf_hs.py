# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "transformers==4.40.1",
#     "timm==0.9.10",
#     "tokenizers==0.19.1",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "pyyaml>=6",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "/home/sejain/research/cvpr27/vla-eval-v0.8.0", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""HarnessStudio OpenVLA LIBERO server with vision-token hooks (EXT openvla-vision-token-hook, capability
``vision_token_hooks``; docs/extensions/openvla-vision-token-hook.md).

Subclass of ``openvla_hs.HarnessOpenVLAServer`` (same dependencies, same model loading and preprocessing). Every
parent file stays byte-identical (tests/test_ext_openvla_residual_hook.py R6 pins openvla_hs.py). Behaviour:

- ``sampler`` None (baseline, derived null, native requests): the parent's path, unchanged (vla-eval
  ``predict_action``, stock vision forward). No hook is installed.
- ``sampler`` naming a decoding hook (e.g. SCALE): the parent's path, unchanged.
- ``sampler`` naming a vision-token hook (``VISION_TOKEN_HOOK = True``, protocol in
  ``harnessstudio.policy_ext.vision_token_hooks``): one request = the call the TTF-VLA release makes in
  ``OpenVLAForActionPrediction.predict_action`` (github.com/PKU-XLab/TTF-VLA@afbd6bf,
  ``prismatic/extern/hf/modeling_prismatic.py:714-800``): append token 29871 to ``input_ids`` if absent, then

      model.generate(input_ids, max_new_tokens=action_dim, output_attentions=hook.OUTPUT_ATTENTIONS,
                     return_dict_in_generate=True, attention_mask=..., pixel_values=..., do_sample=False)

  with the vision backbone's forward swapped for the duration (``vision_forward_hook``) so the hook chooses the
  patch tokens the projector receives. The prefill attention of ``hook.ATTENTION_LAYERS`` is passed to
  ``hook.after_decode``. Tokens are de-tokenized/un-normalized and the gripper mapped by the parent's ``_actions``
  (= vla-eval's native predict + gripper mapping).
- Shape assertion per hooked request: exactly one vision forward; ``action_dim`` generated tokens; one attention
  tuple per generated token when OUTPUT_ATTENTIONS; requested layers present with a square prefill map. A violation
  raises ``VisionTokenHookError`` (infrastructure) before an action is returned.
- Activation evidence: every hooked request appends one JSON line to ``activation_log`` (REQUIRED when a hooked
  request arrives): seq, session, episode, tokens, float64 action (+hex), seeds, input_sha256, hook (factory +
  params), info (the hook's per-request values, e.g. TTF keyframe / reused patches / fallback counters), shape.

Refused: readouts together with a sampler; more than one seed with a vision-token hook (the hook's state follows the
executed action); a hook factory outside ``sampler_prefixes``.

Numerics note (docs section 4): with transformers 4.40.1 and the model's default SDPA attention,
``output_attentions=True`` makes every Llama layer fall back to the eager attention, and for the prefill
``LlamaModel._update_causal_mask`` returns ``None`` (all-ones mask, query length = key length), so the eager
fallback applies no causal mask. This server reproduces the release's call as is; it does not fix it.
"""
from __future__ import annotations

import importlib
import json
import threading
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from harnessstudio.context.trace import fingerprint
from harnessstudio.policies.openvla_hs import HarnessOpenVLAServer
from harnessstudio.policy_ext.vision_token_hooks import check_hook, generate_with_hook, is_vision_token_hook

PATH = "vision_token_hook"


class HarnessOpenVLATTFServer(HarnessOpenVLAServer):
    harness_capabilities = HarnessOpenVLAServer.harness_capabilities | {"vision_token_hooks"}

    def __init__(self, *args: Any, activation_log: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._activation_log = activation_log
        self._vhooks: dict[str, Any] = {}
        self._log_lock = threading.Lock()
        self._log_seq = 0

    # ---- dispatch ------------------------------------------------------------------------------------------
    def sample_chunks(self, obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping],
                      readouts: Sequence[str], ctx) -> list:
        if not sampler or not self._is_vision_sampler(sampler):
            return super().sample_chunks(obs, seeds, sampler, readouts, ctx)
        if readouts:
            raise ValueError("vision_token_hooks: readouts cannot be combined with a vision-token hook sampler")
        if len(seeds) != 1:
            raise ValueError(f"vision_token_hooks: exactly one seed per request, got {len(seeds)}")
        if not self._activation_log:
            raise ValueError("vision_token_hooks: activation_log is required for hooked requests (activation evidence)")
        with self._infer_lock:
            key, hook = self._vhook(sampler)
            state_key = (getattr(ctx, "session_id", None), getattr(ctx, "episode_id", None), key)
            state = self._states.get(state_key)
            if state is None:
                state = hook.init_state()
            tokens, attn, info, shape = self._hooked_generate(hook, state, obs)
            info = hook.after_decode(state, attn, info)
            action = self._actions(np.asarray(tokens))
            a = np.ascontiguousarray(action, dtype=np.float64)
            self._write({"session": getattr(ctx, "session_id", None), "episode": getattr(ctx, "episode_id", None),
                         "path": PATH, "tokens": [int(t) for t in tokens], "action_f64": a.reshape(-1).tolist(),
                         "action_f64_hex": a.tobytes().hex(), "seeds": [int(s) for s in seeds],
                         "input_sha256": fingerprint(dict(obs)), "hook": json.loads(key), "info": info,
                         "shape": shape})
            self._states[state_key] = state
            return [(action[None, :], {"seed": int(seeds[0])})]

    # ---- hooks ---------------------------------------------------------------------------------------------
    def _factory(self, sampler: Mapping):
        factory = str(sampler.get("factory") or "")
        module, sep, attr = factory.partition(":")
        if not sep or not module.startswith(self._sampler_prefixes):
            raise ValueError(f"sampler factory {factory!r} is not allowed (prefixes {self._sampler_prefixes})")
        return getattr(importlib.import_module(module), attr)

    def _is_vision_sampler(self, sampler: Mapping) -> bool:
        return is_vision_token_hook(self._factory(sampler))

    def _vhook(self, sampler: Mapping) -> tuple[str, Any]:
        key = json.dumps({"factory": sampler.get("factory"), "params": sampler.get("params") or {}}, sort_keys=True)
        if key not in self._vhooks:
            hook = self._factory(sampler)(**dict(sampler.get("params") or {}))
            check_hook(hook)
            vb = self._model.vision_backbone
            dims = (int(vb.featurizer.embed_dim), int(vb.fused_featurizer.embed_dim))
            n_patches = int(vb.featurizer.patch_embed.num_patches)
            hook.bind(n_patches=n_patches, dims=dims, num_layers=len(self._model.language_model.model.layers))
            self._vhooks[key] = hook
        return key, self._vhooks[key]

    def _hooked_generate(self, hook, state, obs) -> tuple[list, dict, dict, dict]:
        import torch

        inputs = self._inputs(obs)                      # parent: the native predict's preprocessing and prompt
        action_dim = int(self._model.get_action_dim(self.unnorm_key))
        with torch.no_grad():
            return generate_with_hook(self._model, hook, state, inputs, action_dim)

    # ---- evidence ------------------------------------------------------------------------------------------
    def _write(self, rec: dict) -> None:
        with self._log_lock:
            self._log_seq += 1
            rec = {"seq": self._log_seq, **rec}
            with open(self._activation_log, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(HarnessOpenVLATTFServer)
