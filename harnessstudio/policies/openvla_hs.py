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
"""HarnessStudio server for OpenVLA (vla-eval 0.8.0's OpenVLAModelServer + HarnessPolicyMixin), with
server-side decoding hooks (capability ``decoding_hooks``).

Same dependencies as vla-eval's ``model_servers/openvla.py`` (plus pyyaml for the harnessstudio package).

Native requests (no ``harnessstudio`` key) go to vla-eval's server unchanged: greedy ``predict_action``
(``do_sample=False``), one 7-D action per observation.

HarnessStudio requests return one 1x7 "chunk" per seed:
- ``sampler: None``: vla-eval's native greedy prediction (the seed is not used: greedy is deterministic).
  Baseline and derived-null arms therefore equal the native server bitwise.
- ``sampler: {factory, params}``: the factory (``package.module:Class``, module under one of
  ``sampler_prefixes``) is imported in this process and drives a KV-cached token loop on the same model
  and the same inputs as the native path. Hook protocol (see methods/scale/hs_scale/decoder.py):

      bind(action_token_begin, action_token_end, action_dim)
      init_state() -> state
      attention_temperature(state) -> gamma            # None = stock vision attention
      select_token(step_logits [vocab] float, generator) -> (token_id, info)
      next_state(state, token_infos) -> state

  The loop is the official SCALE ``predict_action`` (github.com/snumprlab/scale, MIT, b4ad2a6): append the
  empty token 29871, one multimodal forward with ``use_cache=True``, then one cached forward per token; the
  vision-attention temperature gamma is applied to every attention layer of both ViTs (DINOv2 + SigLIP)
  as ``scale = head_dim**-0.5 / gamma`` with the official manual-attention forward, and the stock
  attention is restored afterwards, so native requests never see it. Tokens are de-tokenized and
  un-normalized as ``predict_action`` does, then the gripper is mapped like the native server.
  Each seed gets ``torch.Generator(device).manual_seed(seed)``.

  Decoder state (SCALE's per-episode uncertainty history) is kept per (session, episode, sampler) and
  reset at EPISODE_START, like the official evaluation loop resets it every episode. All seeds of one
  request start from the same state (isolated streams); the state then advances with seed 0's sample, so
  with K = 1 (SCALE) it follows the executed action exactly as in the official loop.

Launch (harnessstudio-v2 checkout and the method package on PYTHONPATH):

    CUDA_VISIBLE_DEVICES=5 HF_HOME=/bigdata/sejain/hs2-cache/hf PYTHONSAFEPATH=1 \\
    PYTHONPATH=<harnessstudio-v2>:<harnessstudio-v2>/methods/scale \\
      vla-eval serve --config <yaml with script: .../harnessstudio/policies/openvla_hs.py> --address 127.0.0.1:8115
"""
from __future__ import annotations

import contextlib
import importlib
import json
import threading
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vla_eval.model_servers.openvla import OpenVLAModelServer
from harnessstudio.policy_ext.mixin import HarnessPolicyMixin

EMPTY_TOKEN = 29871  # Llama "▁": OpenVLA appends it after "Out:" (predict_action)


def _manual_attention_forward(attn_module):
    """The official SCALE attention forward (timm's fused path ignores ``scale``), verbatim."""
    def custom_forward(x):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn_module.q_norm(q), attn_module.k_norm(k)
        q = q * attn_module.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = attn_module.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x
    return custom_forward


class HarnessOpenVLAServer(HarnessPolicyMixin, OpenVLAModelServer):
    harness_capabilities = frozenset({"act", "chunk", "seeded_sampling", "isolated_streams", "decoding_hooks"})

    def __init__(self, *args: Any, sampler_prefixes: Sequence[str] = ("hs_",), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._infer_lock = threading.RLock()   # one inference at a time; the vision patch is model state
        self._sampler_prefixes = tuple(sampler_prefixes)
        self._decoders: dict[str, Any] = {}
        self._states: dict[tuple, Any] = {}    # (session, episode, sampler key) -> decoder state

    # ---- native path: vla-eval's predict, serialized with seeded calls ----------------------------
    def predict(self, obs, ctx):
        with self._infer_lock:
            return super().predict(obs, ctx)

    async def on_episode_start(self, config: dict[str, Any], ctx) -> None:
        self._drop_states(ctx)
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx) -> None:
        self._drop_states(ctx)
        await super().on_episode_end(result, ctx)

    def _drop_states(self, ctx) -> None:
        with self._infer_lock:
            for key in [k for k in self._states if k[:2] == (ctx.session_id, ctx.episode_id)]:
                del self._states[key]

    # ---- HarnessStudio path -------------------------------------------------------------------------
    def sample_chunks(self, obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping],
                      readouts: Sequence[str], ctx) -> list:
        if readouts:
            raise ValueError(f"this OpenVLA server offers no readouts (asked for {list(readouts)})")
        with self._infer_lock:
            if not sampler:
                out = []
                for seed in seeds:  # greedy: the same native prediction for every seed (compute paid per seed)
                    action = np.asarray(OpenVLAModelServer.predict(self, obs, ctx)["actions"], dtype=np.float32)
                    out.append((action[None, :], {"seed": int(seed)}))
                return out
            key, decoder = self._decoder(sampler)
            state_key = (ctx.session_id, ctx.episode_id, key)
            state = self._states.get(state_key)
            if state is None:
                state = decoder.init_state()
            inputs = self._inputs(obs)
            out, first_infos = [], None
            for seed in seeds:
                tokens, infos, gamma = self._decode(decoder, state, inputs, int(seed))
                action = self._actions(np.array(tokens))
                out.append((action[None, :], {"seed": int(seed), "tokens": tokens, "gamma": float(gamma),
                                              "u": [i["u"] for i in infos], "tau": [i["tau"] for i in infos]}))
                if first_infos is None:
                    first_infos = infos
            self._states[state_key] = decoder.next_state(state, first_infos)
            return out

    def _decoder(self, sampler: Mapping) -> tuple[str, Any]:
        key = json.dumps({"factory": sampler.get("factory"), "params": sampler.get("params") or {}}, sort_keys=True)
        if key not in self._decoders:
            factory = str(sampler.get("factory") or "")
            module, sep, attr = factory.partition(":")
            if not sep or not module.startswith(self._sampler_prefixes):
                raise ValueError(f"sampler factory {factory!r} is not allowed (prefixes {self._sampler_prefixes})")
            decoder = getattr(importlib.import_module(module), attr)(**dict(sampler.get("params") or {}))
            n_bins = int(self._model.config.n_action_bins)
            decoder.bind(action_token_begin=int(self._model.vocab_size) - n_bins,
                         action_token_end=int(self._model.vocab_size),
                         action_dim=int(self._model.get_action_dim(self.unnorm_key)))
            self._decoders[key] = decoder
        return key, self._decoders[key]

    def _inputs(self, obs: Mapping[str, Any]) -> dict:
        """The native predict's preprocessing and prompt (vla-eval OpenVLAModelServer.predict)."""
        import torch
        pil_image = self._preprocess_image(obs)
        task_description = obs.get("task_description", "")
        prompt = f"In: What action should the robot take to {task_description}?\nOut:"
        return self._processor(prompt, pil_image).to(self._device, dtype=torch.bfloat16)

    def _decode(self, decoder, state, inputs, seed: int) -> tuple[list, list, float]:
        """Official SCALE predict_action loop (single forward pass + KV cache), with hooks."""
        import torch
        model = self._model
        input_ids = inputs["input_ids"]
        if not torch.all(input_ids[:, -1] == EMPTY_TOKEN):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([EMPTY_TOKEN]).long(), dim=0).to(input_ids.device)), dim=1)
        action_dim = model.get_action_dim(self.unnorm_key)
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        gamma = decoder.attention_temperature(state)
        tokens, infos = [], []
        with self._vision_temperature(gamma):
            past_key_values = None
            current_input_ids = input_ids
            for _ in range(action_dim):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = model(input_ids=current_input_ids, attention_mask=inputs.get("attention_mask"),
                                        pixel_values=inputs.get("pixel_values"), past_key_values=None,
                                        use_cache=True, return_dict=True)
                    else:
                        outputs = model(input_ids=current_input_ids, past_key_values=past_key_values,
                                        use_cache=True, return_dict=True)
                step_logits = outputs.logits[:, -1, :][0]
                past_key_values = outputs.past_key_values
                token, info = decoder.select_token(step_logits, generator)
                tokens.append(int(token))
                infos.append(dict(info))
                current_input_ids = torch.tensor([[token]], device=input_ids.device)
        return tokens, infos, (1.0 if gamma is None else gamma)

    @contextlib.contextmanager
    def _vision_temperature(self, gamma):
        """Official apply_visual_attention_temperature on both featurizers; the stock attention comes back after."""
        if gamma is None:
            yield
            return
        vb = self._model.vision_backbone
        featurizers = [vb.featurizer]
        if getattr(vb, "use_fused_vision_backbone", False) and hasattr(vb, "fused_featurizer"):
            featurizers.append(vb.fused_featurizer)
        saved = []
        try:
            for featurizer in featurizers:
                for block in getattr(featurizer, "blocks", []):
                    attn = getattr(block, "attn", None)
                    if attn is None or not hasattr(attn, "scale"):
                        continue
                    saved.append((attn, attn.scale, attn.__dict__.get("forward")))
                    original_scale = attn.head_dim ** -0.5 if hasattr(attn, "head_dim") else attn.scale
                    attn.scale = original_scale / max(gamma, 1e-8)
                    attn.forward = _manual_attention_forward(attn)
            yield
        finally:
            for attn, scale, forward in reversed(saved):
                attn.scale = scale
                if forward is None:
                    attn.__dict__.pop("forward", None)
                else:
                    attn.forward = forward

    def _actions(self, predicted_action_token_ids: np.ndarray) -> np.ndarray:
        """predict_action's de-tokenization + un-normalization, then the native server's gripper mapping."""
        model = self._model
        discretized_actions = model.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
        normalized_actions = model.bin_centers[discretized_actions]
        action_norm_stats = model.get_action_stats(self.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                           normalized_actions)
        action_arr = np.asarray(actions, dtype=np.float32)
        action_arr[..., -1] = -np.sign(2 * action_arr[..., -1] - 1)   # RLDS [0 close, 1 open] -> robosuite
        return action_arr


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(HarnessOpenVLAServer)
