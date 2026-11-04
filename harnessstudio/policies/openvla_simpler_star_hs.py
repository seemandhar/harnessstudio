# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval==0.8.0",
#     "torch==2.3.1",
#     "transformers==4.44.2",
#     "timm==0.9.10",
#     "tokenizers==0.19.1",
#     "pillow",
#     "numpy<2",
#     "accelerate",
#     "opencv-python-headless",
#     "pyyaml>=6",
# ]
# ///
"""HarnessStudio OpenVLA server with decoder-layer residual hooks (EXT openvla-residual-hook, capability
``residual_hooks``; docs/extensions/openvla-residual-hook.md).

Subclass of ``openvla_simpler_hs.HarnessOpenVLASimplerServer`` (mode simpler_pcd_authors). Every parent stays
byte-identical. Behaviour:

- ``sampler`` None (baseline, derived null, any native request): exactly the parent's ``sample_chunks`` /
  ``predict``. No hook is registered, so the module graph is untouched: STAR off == the parent server bitwise.
- ``sampler: {factory, params}`` with the factory's module under ``sampler_prefixes`` (default ``hs_``): a
  *residual hook* object, protocol

      READS: tuple[str, ...]                   observation keys the hook may read (subset of HOOK_READABLE)
      bind(total_layers)                        once, after loading
      init_state() -> state                     first request of an episode (state dropped at EPISODE_START/END)
      begin(state, obs_view) -> (mixer, info)   per request; obs_view = a NEW dict with only READS keys (R2)
      next_state(state, info) -> state          after the request

  The mixer protocol is ``harnessstudio.policy_ext.residual_hooks``. The decode is the parent's unchanged
  ``_generate`` (the authors' no-mask greedy generate) inside ``residual_hooks(model.language_model.model.layers,
  mixer)``, all under ``self._infer_lock`` (R3).
- Runtime shape assertion per hooked request (R4): one seed; ``action_dim`` forwards; forward 1 longer than 1 token,
  forwards 2..N one token each; every forward calls every layer once; mixed calls per forward = layers from the
  mixer's ``correction_start_layer``; one step context (beta) for all forwards. A violation raises
  ``ResidualHookShapeError`` (infrastructure) before an action is returned.
- Activation evidence (R1): proposal extras never reach the trace (context/policy.py keeps only readouts), so every
  hooked request writes one record to the pre-cast action log (path ``residual_hook``): the parent's fields (seq,
  session, episode, tokens, float64 action) plus ``seeds`` (the CRN seeds the hs trace logs in its ``sample`` event:
  the join key), ``input_sha256`` (the framework fingerprint of the observation as received; advisory join key),
  ``hook`` (factory + params), ``info`` (the hook's per-request values, e.g. STAR beta/r/c/v/j/w) and ``shape`` (the
  R4 counts). ``action_log`` is therefore REQUIRED for this server.
- SimplerDecodeShort (R9, supervisor ruling 2026-11-04): unchanged behaviour (the parent raises in every arm); this
  server first writes a ``decode_short`` record (seq, session, episode, path, seeds, input_sha256) so the evaluator can
  count the event per arm and classify the episode as a completed policy failure, never as infrastructure.

Refused: ``readouts`` together with a sampler; more than one seed with a sampler; unknown READS keys.
"""
from __future__ import annotations

import importlib
import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from harnessstudio.context.trace import fingerprint
from harnessstudio.policies.openvla_simpler_hs import HarnessOpenVLASimplerServer, SimplerDecodeShort
from harnessstudio.policy_ext.residual_hooks import ForwardRecord, residual_hooks

HOOK_READABLE = frozenset({"states"})       # public proprioception only (R2); never images/task text/sim keys
PATH = "residual_hook"


class ResidualHookShapeError(RuntimeError):
    """A hooked request did not have the expected decode shape (R4): infrastructure, never an action."""


class HarnessOpenVLASimplerStarServer(HarnessOpenVLASimplerServer):
    harness_capabilities = HarnessOpenVLASimplerServer.harness_capabilities | {"residual_hooks"}

    def __init__(self, model_path: str, *args: Any, action_log: Optional[str] = None, **kwargs: Any) -> None:
        if not action_log:
            raise ValueError("residual_hooks server: action_log is required (activation evidence, review R1)")
        super().__init__(model_path, *args, action_log=action_log, **kwargs)
        self._hooks: dict = {}

    # ---- residual hook objects -------------------------------------------------------------------------------
    def _layers(self):
        return self._model.language_model.model.layers

    def _hook(self, sampler: Mapping) -> tuple[str, Any]:
        key = json.dumps({"factory": sampler.get("factory"), "params": sampler.get("params") or {}}, sort_keys=True)
        if key not in self._hooks:
            factory = str(sampler.get("factory") or "")
            module, sep, attr = factory.partition(":")
            if not sep or not module.startswith(tuple(self._sampler_prefixes)):
                raise ValueError(f"residual hook factory {factory!r} is not allowed (prefixes {self._sampler_prefixes})")
            hook = getattr(importlib.import_module(module), attr)(**dict(sampler.get("params") or {}))
            reads = tuple(getattr(hook, "READS", ()))
            if not set(reads) <= HOOK_READABLE:
                raise ValueError(f"residual hook may read only {sorted(HOOK_READABLE)}, declares {list(reads)}")
            hook.bind(total_layers=len(self._layers()))
            self._hooks[key] = hook
        return key, self._hooks[key]

    @staticmethod
    def _view(obs: Mapping[str, Any], reads: Sequence[str]) -> dict:
        return {k: np.array(obs[k], copy=True) for k in reads if k in obs}

    # ---- R4 -------------------------------------------------------------------------------------------------
    def _check_shape(self, rec: ForwardRecord, mixer) -> dict:
        s = rec.summary()
        n_layers = len(self._layers())
        action_dim = int(self._model.get_action_dim(self.unnorm_key))
        start = int(getattr(mixer, "correction_start_layer", 1) or 1)
        mixed = n_layers - start + 1
        problems = []
        if s["forwards"] != action_dim:
            problems.append(f"{s['forwards']} forwards, expected {action_dim}")
        if s["seq_lens"] and (s["seq_lens"][0] <= 1 or any(n != 1 for n in s["seq_lens"][1:])):
            problems.append(f"sequence lengths {s['seq_lens']}, expected [T>1, 1, ...]")
        if any(c != n_layers for c in s["layer_calls"]):
            problems.append(f"layer calls {s['layer_calls']}, expected {n_layers} per forward")
        if any(c != mixed for c in s["mixed_calls"]):
            problems.append(f"mixed calls {s['mixed_calls']}, expected {mixed} per forward")
        ctxs = [json.dumps(f["ctx"], sort_keys=True, default=str) for f in rec.forwards]
        if len(set(ctxs)) > 1:
            problems.append("the mixer's step context changed within one request")
        if problems:
            raise ResidualHookShapeError("; ".join(problems))
        s.update(n_layers=n_layers, correction_start_layer=start, mixed_per_forward=mixed)
        return s

    # ---- evidence (R1, R9) ----------------------------------------------------------------------------------
    def _write(self, rec: dict) -> None:
        with self._log_lock:
            self._log_seq += 1
            rec = {"seq": self._log_seq, **rec}
            with open(self._action_log, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")

    def _ids(self, ctx) -> dict:
        return {"session": getattr(ctx, "session_id", None), "episode": getattr(ctx, "episode_id", None)}

    def _decode_short(self, path: str, seeds, obs, ctx, exc) -> None:
        self._write({**self._ids(ctx), "path": path, "event": "decode_short", "seeds": [int(s) for s in seeds],
                     "input_sha256": fingerprint(dict(obs)), "error": str(exc)})

    # ---- native paths: unchanged behaviour, decode-short evidence only ------------------------------------
    def predict(self, obs, ctx):
        try:
            return super().predict(obs, ctx)
        except SimplerDecodeShort as exc:
            self._decode_short("native", [], obs, ctx, exc)
            raise

    def sample_chunks(self, obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping],
                      readouts: Sequence[str], ctx) -> list:
        if not sampler:
            try:
                return super().sample_chunks(obs, seeds, sampler, readouts, ctx)
            except SimplerDecodeShort as exc:
                self._decode_short("native_seeded" if not readouts else "logits", seeds, obs, ctx, exc)
                raise
        if readouts:
            raise ValueError("residual_hooks: readouts cannot be combined with a residual hook sampler")
        if len(seeds) != 1:
            raise ValueError(f"residual_hooks: exactly one seed per request, got {len(seeds)}")
        with self._infer_lock:
            key, hook = self._hook(sampler)
            state_key = (getattr(ctx, "session_id", None), getattr(ctx, "episode_id", None), key)
            state = self._states.get(state_key)
            if state is None:
                state = hook.init_state()
            mixer, info = hook.begin(state, self._view(obs, getattr(hook, "READS", ())))
            rec = ForwardRecord()
            try:
                with residual_hooks(self._layers(), mixer, rec):
                    tokens = self._generate(obs)
            except SimplerDecodeShort as exc:
                self._decode_short(PATH, seeds, obs, ctx, exc)
                raise
            shape = self._check_shape(rec, mixer)
            action = self._actions(tokens)
            a = np.ascontiguousarray(action, dtype=np.float64)
            self._write({**self._ids(ctx), "path": PATH, "tokens": [int(t) for t in tokens],
                         "action_f64": a.reshape(-1).tolist(), "action_f64_hex": a.tobytes().hex(),
                         "seeds": [int(s) for s in seeds], "input_sha256": fingerprint(dict(obs)),
                         "hook": json.loads(key), "info": dict(info), "shape": shape})
            self._states[state_key] = hook.next_state(state, info)
            return [(action[None, :], {"seed": int(seeds[0])})]


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(HarnessOpenVLASimplerStarServer)
