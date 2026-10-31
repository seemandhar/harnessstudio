"""SCALE decoding hooks (runs inside a policy server with the ``decoding_hooks`` capability).

Port of the official SCALE implementation (github.com/snumprlab/scale, MIT, commit b4ad2a6),
``prismatic/extern/hf/modeling_prismatic.py``: ``OpenVLAForActionPrediction._compute_self_uncertainty``
and the ``decoding_mode == "scale"`` branch of ``predict_action`` (Algorithm 1). The arithmetic is the
official code's, op for op, so the same logits and the same random stream give the same tokens.

What lives where:
- this class: the algorithm. Self-uncertainty u_k from the action-token logits (Eq. 2-3), per-token
  temperature tau_k = T0 * sigmoid(u_k) and sampling (Eq. 4-5), the step-level EMA (Eq. 6-7) and the
  vision-attention temperature gamma = kappa ** tanh(s * mean(last_u - ema_u)) (Eq. 8);
- the policy server (``harnessstudio/policies/openvla_hs.py``): the model mechanics. It runs the KV-cached
  token loop, applies gamma to every ViT attention layer (official ``apply_visual_attention_temperature``),
  owns the torch generator of each seed and keeps the per-episode state between requests.

Hook protocol used by the server (all state is plain Python lists, kept per episode by the server):
    bind(action_token_begin, action_token_end, action_dim)
    init_state() -> state                         # official: uncertainty_history = None
    attention_temperature(state) -> gamma         # before the forward pass
    select_token(step_logits, generator) -> (token_id, {"u": u_k, "tau": tau_k})
    next_state(state, token_infos) -> state       # after the action's tokens are decoded

The constructor imports nothing heavy (the harness builds it for tier 0); torch is imported lazily.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

# Paper defaults (configs/scale.yaml of the official repo, "tuned on LIBERO-Long and applied unchanged").
DEFAULTS = {"T0": 1.0, "epsilon": 1.0e-12, "num_logits": 256, "kappa": 2.0, "alpha": 0.8, "attn_sensitivity": 0.3}


class ScaleDecoder:
    """SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution (arXiv:2602.04208)."""

    def __init__(self, T0: float = 1.0, epsilon: float = 1.0e-12, num_logits: int = 256, kappa: float = 2.0,
                 alpha: float = 0.8, attn_sensitivity: float = 0.3) -> None:
        if T0 < 0 or kappa <= 0 or not 0.0 <= alpha <= 1.0 or epsilon <= 0:
            raise ValueError("SCALE needs T0 >= 0, kappa > 0, 0 <= alpha <= 1, epsilon > 0")
        self.T0, self.epsilon, self.num_logits = float(T0), float(epsilon), int(num_logits)
        self.kappa, self.alpha, self.attn_sensitivity = float(kappa), float(alpha), float(attn_sensitivity)
        # OpenVLA: the last 256 entries of the 32000-token Llama vocabulary are the action bins.
        self.action_token_begin, self.action_token_end, self.action_dim = 31744, 32000, 7

    # ---- server-side binding -----------------------------------------------------------------
    def bind(self, action_token_begin: int, action_token_end: int, action_dim: int) -> None:
        self.action_token_begin, self.action_token_end = int(action_token_begin), int(action_token_end)
        self.action_dim = int(action_dim)

    # ---- per-episode state (official `uncertainty_history` = [ema_u per dim, last_u per dim]) ----
    def init_state(self) -> list:
        return [[0.0] * self.action_dim, [0.0] * self.action_dim]

    def attention_temperature(self, state: Optional[list]) -> float:
        """gamma_t = kappa ** tanh(s * mean_k(last_u_k - ema_u_k)); 1.0 before any history (Sec. 3.3.2)."""
        if state is None:
            state = self.init_state()
        ema_u, last_u = state[0], state[1]
        if all(u == 0.0 for u in ema_u):
            return 1.0
        mean_delta_u = np.mean([last_u[i] - ema_u[i] for i in range(self.action_dim)])
        return self.kappa ** np.tanh(self.attn_sensitivity * mean_delta_u)

    def next_state(self, state: Optional[list], token_infos: Sequence[dict]) -> list:
        """EMA update of the step-level uncertainty (Eq. 6-7), exactly as the official code orders it."""
        if state is None:
            state = self.init_state()
        new_u = [float(info["u"]) for info in token_infos]
        ema_u, last_u = state[0], state[1]
        if all(u == 0.0 for u in ema_u):
            return [list(new_u), list(new_u)]
        return [[self.alpha * ema_u[i] + (1.0 - self.alpha) * last_u[i] for i in range(self.action_dim)], new_u]

    # ---- token-level decoding (Sec. 3.2 + 3.3.1) -------------------------------------------------
    def self_uncertainty(self, step_logits: Any):
        """u_k = E_{x~p}[log(q_high(x) / q_low(x))] over the top-`num_logits` action tokens (Eq. 2-3).
        Returns (u_k as a Python float, top_indices, probs over the masked action vocabulary)."""
        import torch
        masked_logits = step_logits.clone()
        masked_logits[:self.action_token_begin] = float('-inf')
        masked_logits[self.action_token_end:] = float('-inf')
        probs = torch.softmax(masked_logits, dim=-1)
        K = min(self.num_logits, 256) if self.num_logits > 0 else 256
        top_probs, top_indices = torch.topk(probs, k=K, largest=True, sorted=True)
        p = top_probs / top_probs.sum()
        q_low = torch.full_like(p, self.epsilon)
        q_low[0] = 1.0 - (K - 1) * self.epsilon
        q_high = torch.ones_like(p) / K
        u_k = torch.sum(p * torch.log(q_high / q_low)).item()
        return u_k, top_indices, probs

    def select_token(self, step_logits: Any, generator: Any) -> tuple[int, dict]:
        """tau_k = T0 * sigmoid(u_k); near-greedy below 1e-3, else a_k ~ Cat(softmax(l_k / tau_k)) (Eq. 4-5).
        `step_logits`: [vocab] float logits of the next token; `generator`: the seed's torch generator."""
        import torch
        u_k, top_indices, _ = self.self_uncertainty(step_logits)
        tau_k = self.T0 * torch.sigmoid(torch.tensor(u_k)).item()
        if tau_k < 1e-3:
            selected_token = top_indices[0].item()
        else:
            top_logits = step_logits[top_indices]
            scaled_probs = torch.softmax(top_logits / tau_k, dim=-1)
            sampled_idx = torch.multinomial(scaled_probs, num_samples=1, generator=generator).item()
            selected_token = top_indices[sampled_idx].item()
        return int(selected_token), {"u": u_k, "tau": tau_k}
