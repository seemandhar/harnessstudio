"""scale-gtok decoding hooks: SCALE on the motion tokens, argmax on the gripper token (runs inside the policy server).

``ScaleGripperArgmax`` is ``hs_scale.decoder.ScaleDecoder`` (the faithful SCALE port, arXiv:2602.04208v2, official
code github.com/snumprlab/scale b4ad2a6) with ONE token rule changed:

- action tokens 1..action_dim-1 (OpenVLA LIBERO: tokens 1-6 = the 6 motion dims): ``ScaleDecoder.select_token``
  unchanged (u_k, tau_k = T0 * sigmoid(u_k), multinomial on softmax(l_k / tau_k) with the decision's generator);
- action token action_dim (token 7 = the gripper dim): the masked argmax over the 256 action bins of the SAME
  step logits, i.e. ``top_indices[0]`` of ``self_uncertainty`` (SCALE's own tau < 1e-3 branch). Its info is
  ``{"u": u_k, "tau": tau_k}`` with u_k and tau_k computed exactly as SCALE computes them (review RC1), so the
  EMA (next_state), gamma (attention_temperature) and the reported tau stay SCALE's.

The logits of token 7 come from SCALE's own forward pass (gamma-tempered vision attention, the sampled motion-token
prefix in the KV cache). Token 7 is the last token of a decision and the server builds a fresh
``torch.Generator(device).manual_seed(seed)`` per decision (openvla_hs.py e7c6a4c3, ``_decode`` line 185), so
skipping token 7's multinomial changes no other random draw (review RC3): no draw-and-discard is needed.

Token position (review RC2): the server calls ``attention_temperature`` exactly once per ``_decode``, before the
token loop (openvla_hs.py e7c6a4c3 line 186), then ``select_token`` once per action token. The counter is reset in
``attention_temperature`` and ``select_token`` refuses a call before the first reset or beyond ``action_dim`` calls
since the last reset. This also refuses the unmerged openvla-ksample K > 1 paths (per_seed: K * action_dim calls
after one reset; batched: token-major interleaving) instead of silently mis-assigning positions. The card pins
K = 1 (proposer hs_scale:SingleSample). ``tests/test_server_pin.py`` pins the server file hash.

Nothing heavy is imported at module level (torch lazily), as in hs_scale.decoder.
"""
from __future__ import annotations

from typing import Any, Optional

from hs_scale.decoder import DEFAULTS, ScaleDecoder

__all__ = ["DEFAULTS", "ScaleGripperArgmax"]


class ScaleGripperArgmax(ScaleDecoder):
    """SCALE with a deterministic mode-of-bins rule for the gripper token (the last action token)."""

    def __init__(self, T0: float = 1.0, epsilon: float = 1.0e-12, num_logits: int = 256, kappa: float = 2.0,
                 alpha: float = 0.8, attn_sensitivity: float = 0.3) -> None:
        super().__init__(T0=T0, epsilon=epsilon, num_logits=num_logits, kappa=kappa, alpha=alpha,
                         attn_sensitivity=attn_sensitivity)
        self._position: Optional[int] = None   # token position within the current decision; None = no decision open

    # ---- decision boundary: the server calls this once per _decode, before the token loop -------------------
    def attention_temperature(self, state: Optional[list]) -> float:
        self._position = 0
        return super().attention_temperature(state)

    # ---- token-level decoding ------------------------------------------------------------------------------
    def select_token(self, step_logits: Any, generator: Any) -> tuple[int, dict]:
        position = self._position
        if position is None:
            raise RuntimeError("scale-gtok: select_token called before attention_temperature (no decision open); "
                               "the server's hook order is not the one pinned (openvla_hs.py e7c6a4c3 _decode)")
        if position >= self.action_dim:
            raise RuntimeError(f"scale-gtok: select_token called {position + 1} times since the last "
                               f"attention_temperature, more than action_dim={self.action_dim}; refusing (K > 1 or "
                               "batched/interleaved decoding is not supported by the token-position counter)")
        ndim = getattr(step_logits, "ndim", None)
        if ndim is not None and ndim != 1:
            raise ValueError(f"scale-gtok: step_logits must be one [vocab] vector, got ndim={ndim}")
        self._position = position + 1
        if position != self.action_dim - 1:
            return super().select_token(step_logits, generator)       # motion tokens: SCALE, bit for bit
        # Gripper token: SCALE's own u_k and tau_k (RC1), argmax token (SCALE's tau < 1e-3 branch).
        import torch
        u_k, top_indices, _ = self.self_uncertainty(step_logits)
        tau_k = self.T0 * torch.sigmoid(torch.tensor(u_k)).item()
        selected_token = top_indices[0].item()
        return int(selected_token), {"u": u_k, "tau": tau_k}
