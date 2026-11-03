"""HarnessStudio server for OpenVLA with the per-token ``logits`` readout of the NATIVE greedy decode
(EXT openvla-contrast, design L; docs/extensions/openvla-contrast.md).

It is ``HarnessOpenVLAServer`` (policies/openvla_hs.py, unchanged) plus one readout:

- Every request that does not ask for ``logits`` runs the parent code unchanged: native vla-eval requests,
  seeded requests without readouts (baseline arm, shadow-null native replays, identity), SCALE samplers, and the
  parent's own refusal of other readouts.
- ``readouts == ["logits"]`` with ``sampler: None``: vla-eval's native ``predict`` (copied in
  ``_predict_recorded``; the only difference is two recording hooks passed to ``predict_action``'s generate
  kwargs) returns the native greedy action AND, per generated position, the raw float32 logit vector the greedy
  argmax saw (``transformers`` 4.40.1 ``_greedy_search``: ``next_token_logits`` -> logits processors -> argmax).
  A per-request self-check proves that the recorded rows reproduce the generated tokens exactly (token level, all
  positions) and the native action bitwise; a mismatch raises (infrastructure failure, never silent).
- ``logits`` with a sampler, or with any other readout, is refused.

Extras of a ``logits`` request (companion keys ``logits_*``; functions of public inputs and frozen weights only):

    logits              float32 [T, V_logits]   T = generated positions (7 unless EOS came early), V_logits = 32064
    logits_tokens       int32   [T]             the generated (= native greedy) tokens
    logits_bin_actions  float32 [n_bins, A]     executable action of every bin and dimension (row b = _actions of
                                                token vocab_size-1-b in every dimension)
    logits_token_end    int                     model.vocab_size (32000)
    logits_n_bins       int                     config.n_action_bins (256)

A method maps tokens t to the executed action with
``table[clip(token_end - t - 1, 0, n_bins - 1), d]``; this equals ``_actions(t)[d]`` bitwise because every row is
produced by ``_actions`` element-wise. The native de-tokenization clips to ``bin_centers.shape[0] - 1`` = 254 (255
centres), so table row 255 (token 31744) equals row 254 by construction, and every token <= 31745 maps to the top
bin 254 (token 0 included).

Launch exactly like openvla_hs.py (same YAML args, same pinned interpreter and cuBLASLt serving mode), with this file
as the script and the harnessstudio checkout on PYTHONPATH. Never ``uv run`` it (no inline script metadata on
purpose: the pinned openvla-hs environment is used directly).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vla_eval.model_servers.openvla import OpenVLAModelServer  # noqa: F401  (the copied predict's origin)
from harnessstudio.policies.openvla_hs import HarnessOpenVLAServer

READOUT = "logits"


def _recorders():
    """(logits recorder, token recorder) as transformers hook objects; imported lazily (torch/transformers)."""
    import torch
    from transformers import LogitsProcessor, StoppingCriteria

    class LogitsRecorder(LogitsProcessor):
        """Stores each step's scores as handed to the processor list and returns the SAME tensor object."""

        def __init__(self) -> None:
            self.rows: list = []

        def __call__(self, input_ids, scores):
            self.rows.append(scores[0].detach().clone())
            return scores

    class TokenRecorder(StoppingCriteria):
        """Called after each token is appended: records the generated ids; never stops generation."""

        def __init__(self) -> None:
            self.tokens: list = []

        def __call__(self, input_ids, scores, **kwargs):
            self.tokens.append(int(input_ids[0, -1]))
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    return LogitsRecorder(), TokenRecorder()


class RecordedDecodeMismatch(RuntimeError):
    """The recorded logits do not reproduce the native decode (tokens or action)."""


class HarnessOpenVLALogitsServer(HarnessOpenVLAServer):
    harness_capabilities = HarnessOpenVLAServer.harness_capabilities | {"logits_readout"}

    # ---- routing: only requests that ask for logits reach new code (review R2) ---------------------------------
    def sample_chunks(self, obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping],
                      readouts: Sequence[str], ctx) -> list:
        if READOUT not in readouts:
            return super().sample_chunks(obs, seeds, sampler, readouts, ctx)
        if tuple(readouts) != (READOUT,):
            raise ValueError(f"the 'logits' readout cannot be combined with other readouts (asked for {list(readouts)})")
        if sampler:
            raise ValueError("the 'logits' readout is defined for the native greedy decode only (sampler None)")
        with self._infer_lock:
            table = self._bin_table()
            end = int(self._model.vocab_size)
            n_bins = int(self._model.config.n_action_bins)
            out = []
            for seed in seeds:  # greedy: the same native decode for every seed (compute paid per seed, as the parent)
                action, rows, tokens = self._recorded(obs)
                out.append((action[None, :], {"seed": int(seed), "logits": rows, "logits_tokens": tokens,
                                              "logits_bin_actions": table, "logits_token_end": end,
                                              "logits_n_bins": n_bins}))
            return out

    def _recorded(self, obs: Mapping[str, Any]) -> tuple:
        """Native action + recorded rows + generated tokens, after the token- and action-level self-check (R3)."""
        action, logits_rec, token_rec = self._predict_recorded(obs)
        rows = self._rows_numpy(logits_rec.rows)
        tokens = np.asarray(token_rec.tokens, dtype=np.int64)
        self._self_check(action, rows, tokens)
        return action, rows, tokens.astype(np.int32)

    @staticmethod
    def _rows_numpy(rows: list) -> np.ndarray:
        import torch
        if not rows:
            raise RecordedDecodeMismatch("the logits recorder saw no decoding step")
        stacked = torch.stack(rows)
        if stacked.dtype != torch.float32:
            raise RecordedDecodeMismatch(f"recorded logits are {stacked.dtype}, expected float32")
        return stacked.cpu().numpy()  # one device-to-host copy per request

    def _self_check(self, action: np.ndarray, rows: np.ndarray, tokens: np.ndarray) -> None:
        if rows.ndim != 2 or len(tokens) != rows.shape[0]:
            raise RecordedDecodeMismatch(f"{rows.shape[0]} recorded rows for {len(tokens)} generated tokens")
        if not np.array_equal(rows.argmax(-1), tokens):
            raise RecordedDecodeMismatch("argmax of the recorded logits differs from the generated tokens")
        action_dim = int(self._model.get_action_dim(self.unnorm_key))
        if len(tokens) == action_dim and not np.array_equal(self._actions(tokens), action):
            raise RecordedDecodeMismatch("the generated tokens do not de-tokenize to the native action")

    def _bin_table(self) -> np.ndarray:
        """float32 [n_bins, action_dim]: _actions of token vocab_size-1-b in every dimension (cached per server)."""
        cached = self.__dict__.get("_logits_bin_table")
        if cached is None:
            n_bins = int(self._model.config.n_action_bins)
            action_dim = int(self._model.get_action_dim(self.unnorm_key))
            tok = int(self._model.vocab_size) - 1 - np.arange(n_bins)
            cached = self._actions(np.tile(tok[:, None], (1, action_dim)))
            cached.setflags(write=False)
            self.__dict__["_logits_bin_table"] = cached
        return cached

    def _predict_recorded(self, obs):
        """vla-eval OpenVLAModelServer.predict (model_servers/openvla.py:118-135) with the two recorders.

        Differences (pinned by an AST test): the recorder construction, the two extra keyword arguments of
        predict_action and the return value.
        """
        import torch
        from transformers import LogitsProcessorList, StoppingCriteriaList

        pil_image = self._preprocess_image(obs)
        task_description = obs.get("task_description", "")
        prompt = f"In: What action should the robot take to {task_description}?\nOut:"

        inputs = self._processor(prompt, pil_image).to(self._device, dtype=torch.bfloat16)

        kwargs: dict[str, Any] = {"do_sample": False}
        if self.unnorm_key:
            kwargs["unnorm_key"] = self.unnorm_key

        logits_rec, token_rec = _recorders()
        action = self._model.predict_action(**inputs, **kwargs, logits_processor=LogitsProcessorList([logits_rec]),
                                            stopping_criteria=StoppingCriteriaList([token_rec]))
        # Gripper: RLDS [0=close,1=open] → robosuite [-1=open,+1=close]
        action_arr = np.asarray(action, dtype=np.float32)
        action_arr[..., -1] = -np.sign(2 * action_arr[..., -1] - 1)
        return action_arr, logits_rec, token_rec


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(HarnessOpenVLALogitsServer)
