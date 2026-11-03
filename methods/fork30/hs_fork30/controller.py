"""fork30 (HarnessStudio v2 controller, OpenVLA libero_10 / openvla-libero-hs): native greedy decoding; at decision
`fork_step` (30) the frozen cue R = #{k in 1..29 : executed a[k] == a[k-1], exact 7-D equality} is evaluated; if
R >= `min_repeats` (6) every later decision decodes with the frozen SCALE decoder (hs_scale, paper defaults); otherwise
greedy for the whole episode. Review runs/scalepp-20261102/reviews/fork-r4-review.json (F1-F4); cue from
analysis/fork-headroom (scripts/features.py: rep = sum(all(A[1:] == A[:-1])) over the first 30 executed rows).

* exactly ONE ctx.policy.sample per decision (sampler None before the fork and on unfired episodes; the SCALE ref after
  the fork), so decision d uses crn_seed(ns, benchmark, task, episode, d, 0) - the seed SCALE s0/s1 used at d;
* executed actions = the action as the vla-eval LIBERO benchmark executes it (dims 0-5 as sent, gripper -1 if < 0 else
  +1), i.e. the recorded rows the cue was computed on;
* the SCALE decoder's per-episode state starts at the fork (gamma = 1 at the first SCALE decision; disclosed);
* public inputs: the executed action history only (no observation is read; send_state: false, F2);
* null (shadow): the cue and the post-fork SCALE calls run on the shadow session, native greedy actions execute.
Evidence: ctx.record("fork", fired, repeats, at) at the fork step, accounting.extra["fork30"].
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

# identity of the read-only escape operator (review F1: hs_scale decoder 619d9b96, paper defaults)
SCALE_DECODER_SHA256 = "619d9b960bd84878a34825c1e35e7f182cade2515cc51721a01fa5fe14256d60"
SCALE_FINGERPRINT = "73123974003d91a9d99156e7866925cae973e4dca97bd018dc1f6d47cec4255f"
SCALE_DEFAULTS = {"T0": 1.0, "epsilon": 1.0e-12, "num_logits": 256, "kappa": 2.0, "alpha": 0.8, "attn_sensitivity": 0.3}


def executed_action(action: Any) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.size != 7:
        raise ValueError(f"fork30 expects 7-D LIBERO actions, got {a.size}")
    out = a.copy()
    out[6] = np.float32(-1.0) if float(a[6]) < 0 else np.float32(1.0)
    return out


def scale_ref(sampler: Mapping[str, Any]) -> dict:
    """Validated sampler ref; refuses unless it is hs_scale's frozen decoder at the paper defaults."""
    ref = {"factory": str(sampler.get("factory")), "params": dict(sampler.get("params") or {})}
    if ref["factory"] != "hs_scale.decoder:ScaleDecoder" or ref["params"] != SCALE_DEFAULTS:
        raise ValueError("fork30's operator must be hs_scale.decoder:ScaleDecoder at the SCALE paper defaults")
    mod = importlib.import_module("hs_scale.decoder")
    if hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() != SCALE_DECODER_SHA256:
        raise ValueError(f"hs_scale decoder differs from the frozen 619d9b96 ({mod.__file__})")
    from harnessstudio.core import registry
    if registry.load("scale").fingerprint() != SCALE_FINGERPRINT:
        raise ValueError("the scale method fingerprint differs from the frozen 73123974")
    mod.ScaleDecoder(**ref["params"])
    return ref


class RepeatCue:
    """R = #{k in 1..fork_step-1 : A[k] == A[k-1]} on executed actions; decided after `fork_step` executed actions."""

    def __init__(self, fork_step: int = 30, min_repeats: int = 6) -> None:
        if fork_step < 2 or min_repeats < 1:
            raise ValueError("fork30 needs fork_step >= 2 and min_repeats >= 1")
        self.fork_step, self.min_repeats = int(fork_step), int(min_repeats)
        self.k = 0
        self.prev: Optional[np.ndarray] = None
        self.repeats = 0
        self.decided: Optional[bool] = None

    def step(self, action: Any) -> Optional[bool]:
        """After an executed action. Returns the decision (True = fork) exactly once, after action fork_step-1."""
        a = executed_action(action)
        if self.k < self.fork_step and self.prev is not None and np.array_equal(a, self.prev):
            self.repeats += 1
        self.prev = a
        self.k += 1
        if self.k == self.fork_step and self.decided is None:
            self.decided = self.repeats >= self.min_repeats
            return self.decided
        return None


class Fork30Controller:
    def __init__(self, fork_step: int = 30, min_repeats: int = 6, sampler: Optional[Mapping[str, Any]] = None) -> None:
        RepeatCue(fork_step, min_repeats)                  # validate
        self.fork_step, self.min_repeats = int(fork_step), int(min_repeats)
        self.sampler = scale_ref(sampler or {"factory": "hs_scale.decoder:ScaleDecoder", "params": SCALE_DEFAULTS})

    async def run(self, ctx) -> None:
        cue = RepeatCue(self.fork_step, self.min_repeats)
        forked = False
        extra = ctx.accounting.extra
        state = {"decisions": 0, "scale_decisions": 0}

        def summary() -> dict:
            return {"fork_step": self.fork_step, "min_repeats": self.min_repeats, "repeats": cue.repeats,
                    "decided": cue.decided, "fired": bool(cue.decided), "policy_decisions": state["decisions"],
                    "scale_decisions": state["scale_decisions"], "steps": cue.k}

        extra["fork30"] = summary()
        while not ctx.ended:
            proposals = await ctx.policy.sample(1, sampler=self.sampler if forked else None)
            state["decisions"] += 1
            state["scale_decisions"] += int(forked)
            extra["fork30"] = summary()
            chunk = np.asarray(proposals[0].actions)
            if chunk.ndim != 2 or chunk.shape[0] != 1:
                raise ValueError(f"fork30 is defined for one action per request (OpenVLA); got {chunk.shape}")
            if await ctx.execute(chunk) < 1:
                break
            decision = cue.step(chunk[0])
            if decision is not None:
                forked = decision
                ctx.record("fork", fired=bool(decision), repeats=cue.repeats, at=self.fork_step)
            extra["fork30"] = summary()
