"""Toy `null: shadow` controller for EXT openvla-contrast (tests only; never an installed method).

Per step it sends two seeded `logits` requests: the real public observation, and the same observation with its
first image replaced by a deterministic public-frame transform (a stand-in for PCD's inpainted counterfactual).
It combines the per-position logits with the code-faithful PCD expression and executes the combined tokens through
the server's bin->action table. Loaded as `hs_ext_logits_toy:toy` with tests/toys on sys.path.
"""
from __future__ import annotations

import numpy as np

from harnessstudio.core.method import Budgets, Fidelity, MethodConfig, ModuleRef

REQUIRED = frozenset({"act", "chunk", "seeded_sampling", "isolated_streams", "logits_readout"})


def pcd_code_combine(s, c, alpha: float, contrast_dims, action_dim: int):
    """Code-faithful combine (Koorye/PCD openvla_contrast.py:42-58), evaluated exactly as (1+alpha)*s - alpha*c
    with a Python-double alpha on float32 arrays (review R9d). Returns (tokens int64 [action_dim], info)."""
    s = np.asarray(s, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    width = s.shape[1] if s.ndim == 2 else c.shape[1]
    if s.shape[0] < action_dim:                     # A short: its scores become zeros (contrast still applies)
        s = np.zeros((action_dim, width), dtype=np.float32)
    f = s.copy()
    dims = list(contrast_dims)
    contrasted = c.shape[0] == action_dim           # B short: contrast skipped
    if contrasted:
        f[dims] = (1 + alpha) * s[dims] - alpha * c[dims]
    tokens = f.argmax(-1)
    return tokens, {"contrasted": bool(contrasted), "changed": int(np.sum(tokens != s.argmax(-1)))}


def table_action(extras, tokens) -> np.ndarray:
    table = np.asarray(extras["logits_bin_actions"], dtype=np.float32)
    end, n_bins = int(extras["logits_token_end"]), int(extras["logits_n_bins"])
    idx = np.clip(end - np.asarray(tokens, dtype=np.int64) - 1, 0, n_bins - 1)
    return table[idx, np.arange(table.shape[1])]


def counterfactual(image, shift: int) -> np.ndarray:
    return ((np.asarray(image, dtype=np.uint16) + shift) % 256).astype(np.uint8)


class LogitsContrastToy:
    def __init__(self, alpha: float = 0.8, contrast_dims=(0, 1, 2, 3, 4, 5), shift: int = 37) -> None:
        self.alpha, self.contrast_dims, self.shift = float(alpha), tuple(int(d) for d in contrast_dims), int(shift)

    async def run(self, ctx) -> None:
        stats = ctx.accounting.extra.setdefault("logits_toy", {"steps": 0, "changed_dims": 0, "skipped": 0})
        while not ctx.ended:
            obs = ctx.observe()
            images = dict(obs["images"])
            key = next(iter(images))                 # the image OpenVLA reads (vla-eval openvla.py:88-89)
            images[key] = counterfactual(images[key], self.shift)
            [real] = await ctx.policy.sample(1, readouts=["logits"])
            [cfp] = await ctx.policy.sample(1, obs={"images": images}, readouts=["logits"])
            action_dim = int(np.asarray(real.extras["logits_bin_actions"]).shape[1])
            tokens, info = pcd_code_combine(real.extras["logits"], cfp.extras["logits"], self.alpha,
                                            self.contrast_dims, action_dim)
            stats["steps"] += 1
            stats["changed_dims"] += info["changed"]
            stats["skipped"] += int(not info["contrasted"])
            await ctx.execute(table_action(real.extras, tokens)[None, :])


def toy(alpha: float = 0.8) -> MethodConfig:
    # policy_calls: 2 requests per step over the longest horizon used (libero_10 520 steps -> 1040), plus margin (A2).
    return MethodConfig(name=f"test-logits-contrast-toy-a{alpha:g}", version="1", paper="none",
                        fidelity=Fidelity.FAITHFUL, required=REQUIRED, null="shadow",
                        controller=ModuleRef(f"{__name__}:LogitsContrastToy", {"alpha": alpha}),
                        budgets=Budgets(policy_calls=1100))


def toy_alpha0() -> MethodConfig:
    """alpha = 0: (1+0)*s - 0*c == s exactly, so the candidate must execute the native actions bitwise (GPU G5)."""
    return toy(0.0)
