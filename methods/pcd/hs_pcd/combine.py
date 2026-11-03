"""PCD's per-step token combination, re-derived from Koorye/PCD @ 31b027e (no licence: independent re-derivation,
nothing copied; every rule cites the released file:line).

contrast_policies/openvla_contrast.py:
- :9       _TOKEN_DIM = 32064 (the width of the zero fallback rows)
- :40-44   stream A (real frame) scores; if it has fewer than action_dim (7) rows they are REPLACED by
           zeros(7, 32064) float32 (argmax of a zero row = token 0 -> bin 254, the top bin)
- :46-51   stream B (counterfactual frame) scores; if it has fewer than action_dim - 1 (6) rows they are replaced by
           zeros(6, 32064); a 6-row B is kept as is (not padded)
- :53-55   final = A.clone(); ONLY if B has exactly action_dim rows:
           final[:-1] = (1 + alpha) * final[:-1] - alpha * B[:-1]     (positions 0..5; the gripper row is A's)
           NOTE: with the zero A fallback and a 7-row B, positions 0..5 become -alpha * B (= argmin of B's rows):
           the contrast is still applied (ext review R1b; the earlier audit text said "every argmax is token 0")
- :58      tokens = final.argmax(-1) over the FULL logit width (no plausibility cut, no temperature)
- :106-124 scores = torch.cat(generate(output_scores=True)['scores']): raw float32 logits of the greedy decode
           (transformers greedy with no processors); on HarnessStudio they come from the `logits` readout.

The expression is evaluated exactly in this form and order on float32 arrays with a Python-double alpha
(ext review R9d): numpy float32 * Python float stays float32 (value-based casting in numpy 1.x, NEP 50 in 2.x),
each multiply / subtract is one IEEE-rounded op, so it equals torch's CPU/CUDA eager kernels bitwise.
`s + alpha * (s - c)` is algebraically equal but NOT bitwise equal; never use it.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

TOKEN_DIM = 32064   # openvla_contrast.py:9


def pcd_code_combine(s, c, alpha: float, action_dim: int = 7, token_dim: Optional[int] = None):
    """Code-faithful PCD combine. s, c: [T, V] float32 per-position logits of streams A (real) and B (counterfactual).

    Returns (tokens int64 [action_dim], info) with info = {a_fallback, b_short, contrasted, b_rows, a_rows}.
    token_dim: width of the zero fallback rows (the authors' constant 32064); default = the observed row width.
    """
    tokens, info, _ = pcd_code_final(s, c, alpha, action_dim, token_dim)
    return tokens, info


def pcd_code_final(s, c, alpha: float, action_dim: int = 7, token_dim: Optional[int] = None):
    """As pcd_code_combine, plus the final score rows [action_dim, V] float32 (the authors' final_scores)."""
    s = np.asarray(s, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    if token_dim is None:
        widths = [x.shape[1] for x in (s, c) if x.ndim == 2 and x.shape[0] > 0]
        token_dim = widths[0] if widths else TOKEN_DIM
    a_rows = int(s.shape[0]) if s.ndim == 2 else 0
    b_rows = int(c.shape[0]) if c.ndim == 2 else 0
    a_fallback = a_rows < action_dim                                   # :42-44
    scores = np.zeros((action_dim, token_dim), dtype=np.float32) if a_fallback else s
    if b_rows < action_dim - 1:                                        # :49-51
        c = np.zeros((action_dim - 1, token_dim), dtype=np.float32)
    final = scores.copy()                                              # :53
    contrasted = c.shape[0] == action_dim                              # :54
    if contrasted:
        final[:-1] = (1 + alpha) * final[:-1] - alpha * c[:-1]         # :55
    tokens = final.argmax(axis=-1)                                     # :58
    return tokens.astype(np.int64), {"a_fallback": bool(a_fallback), "b_short": b_rows < action_dim,
                                     "contrasted": bool(contrasted), "a_rows": a_rows, "b_rows": b_rows}, final


def table_action(extras: Mapping, tokens) -> np.ndarray:
    """Executable action of `tokens` through the server's bin table (bitwise == the server's _actions(tokens);
    EXT openvla-contrast section 3.1 / R1a)."""
    table = np.asarray(extras["logits_bin_actions"], dtype=np.float32)
    end, n_bins = int(extras["logits_token_end"]), int(extras["logits_n_bins"])
    idx = np.clip(end - np.asarray(tokens, dtype=np.int64) - 1, 0, n_bins - 1)
    return table[idx, np.arange(table.shape[1])]


def token_stats(tokens, native_tokens, extras: Mapping) -> dict:
    """Per-dimension evidence (SETTING.md v2 edit 3): changed vs the native greedy token, and selections outside
    the action vocabulary [token_end - n_bins, token_end)."""
    tokens = np.asarray(tokens, dtype=np.int64)
    native = np.asarray(native_tokens, dtype=np.int64)
    end, n_bins = int(extras["logits_token_end"]), int(extras["logits_n_bins"])
    n = min(len(tokens), len(native))
    changed = [int(d) for d in range(n) if tokens[d] != native[d]]
    outside = [int(d) for d in range(len(tokens)) if not (end - n_bins <= tokens[d] < end)]
    return {"changed_dims": changed, "non_action_vocab_dims": outside}


def top2_gap(rows) -> list:
    """top1 - top2 logit gap per row (float64), for the tie-sensitivity evidence (R9b)."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return []
    top = np.sort(rows, axis=-1)[:, -2:]
    return [float(x) for x in top[:, 1] - top[:, 0]]
