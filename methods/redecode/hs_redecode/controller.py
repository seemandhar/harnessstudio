"""ReDecode (round-4 CONTROL P1, runs/rtcfpp-20261101/ideation/r4-control.json; review r4-proposal-review.json C-R1..C-R11;
amendment A4): a HarnessStudio v2 controller for openpi pi0_fast_libero on vla-eval LIBERO.

Per decision (cadence = exec_rows = 5 = profile execute_horizon):
  1. one native seeded call `await ctx.policy.sample(1)` (no override: the bitwise-native call of the baseline arm);
  2. if the NATIVE chunk is the openpi zero chunk (hs_tilt_guard_v12.action_op.is_openpi_zero_chunk vs the decision-time
     state s, tol 1e-4), up to two retries `await ctx.policy.sample(1, obs={"states": s_sent})` with
     s_sent = s + delta, then s - delta, delta_i = (std_i + 1e-6) * 2/256 on all 8 state dims (one z-score state-token
     bin of the openpi FAST tokenizer; std from the pinned norm stats, C-R2);
     - a retry is VALID iff its RAW output is not the zero chunk for the state actually sent (C-R1: a failed retry is
       served as mean + s_sent[:6], which differs from mean + s by about one bin > tol, so it must be tested against
       s_sent); the first valid retry is re-based: dims 0-5 -= (s_sent - s)[:6] (C-R3: openpi AbsoluteActions added back
       s_sent; the re-base is a convention that makes the executed command reference the true state, a change of
       exactly one state-token bin, <= 0.0071 action units, largest on ry), gripper as decoded;
     - if both retries are zero chunks, the native zero chunk is kept;
  3. the selected chunk goes through the FROZEN tilt-guard v1.2 action op (hs_tilt_guard_v12.composite.
     TiltGuardV12ActionOp, imported read-only, C-R9) with zero="substitute" (Z: tail, then hold, cap 2) and
     envelope "off" ('redecode') or "bte" ('redecode-b'), so Z and B run byte-for-byte as in v1.2. A successful retry
     is not a zero chunk for the op, so Z does not fire and B's boundary detector counts it like any policy decision
     (disclosed: B may arm differently than in v1.2 'bz');
  4. execute rows[:exec_rows] of the op output with ctx.execute.
Every other decision executes the native rows bitwise (the v1.2 op returns its input object when nothing is active).

Per call the controller records (ctx.record "redecode"): decision index, call role (native / retry+1 / retry-1), the
number of the 8 state tokens the shift changes (public replica of openpi tokenizer.py L70: digitize(z, linspace(-1, 1,
257)[:-1]) - 1 with z = (s - mean)/(std + 1e-6)), whether the raw output is the zero chunk for the sent state, the
selected source and the v1.2 op diagnostics. The framework logs the seeds of every call ("sample" events); the
method never reads seeds (INVARIANTS 2). Each retry advances the framework decision counter, so later native calls use
different CRN seeds than the baseline (C-R4); pi0-FAST decodes greedily, so the evaluator's tier-2 check must show seed
invariance. Null arm: shadow (the controller runs and pays its retries on the shadow session; executed actions native).

No RNG, no simulator state, no case id/seed/arm; public inputs only (obs['states'], the policy's own outputs).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from hs_tilt_guard_v12.action_op import is_openpi_zero_chunk
from hs_tilt_guard_v12.composite import TiltGuardV12ActionOp

PKG = Path(__file__).resolve().parent
BINS = np.linspace(-1.0, 1.0, 256 + 1)[:-1]


def load_norm_stats(path: str, sha256: Optional[str]) -> dict:
    p = Path(path)
    p = p if p.is_absolute() else PKG / p
    data = p.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if sha256 and digest != sha256:
        raise RuntimeError(f"norm stats {path}: sha256 {digest} != pinned {sha256}")
    ns = json.loads(data)
    return ns.get("norm_stats", ns)


def state_tokens(s: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Public replica of the openpi FAST state tokens (z-score normalization, 256 bins over [-1, 1])."""
    z = (np.asarray(s, dtype=np.float64) - mean) / (std + 1e-6)
    return np.digitize(z, bins=BINS) - 1


class ReDecodeController:
    def __init__(self, shift_bins: Sequence[int] = (1, -1), bin_width: float = 2.0 / 256, max_retries: int = 2,
                 zero_chunk_tol: float = 1e-4, exec_rows: int = 5, state_key: str = "states",
                 norm_stats: str = "assets/pi0_fast_libero_norm_stats.json", norm_stats_sha256: Optional[str] = None,
                 tilt_guard: Optional[Mapping[str, Any]] = None, tilt_guard_source_entry: str = "") -> None:
        ns = load_norm_stats(norm_stats, norm_stats_sha256)
        self.state_mean = np.asarray(ns["state"]["mean"], dtype=np.float64)
        self.state_std = np.asarray(ns["state"]["std"], dtype=np.float64)
        self.action_mean = np.asarray(ns["actions"]["mean"], dtype=np.float64)
        self.delta = (self.state_std + 1e-6) * float(bin_width)          # C-R2
        self.shift_bins = [int(b) for b in shift_bins]
        self.max_retries = int(max_retries)
        if self.max_retries < 0 or self.max_retries > len(self.shift_bins):
            raise ValueError("max_retries must be in [0, len(shift_bins)]")
        self.zero_tol = float(zero_chunk_tol)
        self.exec_rows = int(exec_rows)
        self.state_key = state_key
        tg = dict(tilt_guard or {})
        if tg.get("zero") != "substitute":
            raise ValueError("ReDecode falls back to tilt-guard v1.2 Z: tilt_guard.zero must be 'substitute'")
        self.tg_params = tg
        self.tg_source = str(tilt_guard_source_entry)     # provenance only (the frozen v1.2 card the params were copied from)
        # built here (constructors run per episode in every arm; reading pinned assets only)
        self.op = TiltGuardV12ActionOp(**tg)
        self.stats = {"decisions": 0, "native_zero": 0, "retry_calls": 0, "retry_valid": 0, "retry_zero": 0,
                      "retry_tokens_unchanged": 0, "fallback_native_zero": 0, "selected_retry": {}}

    # ---- pure helpers (unit-tested) ---------------------------------------------------------------------------
    def state_vector(self, obs) -> Optional[np.ndarray]:
        st = obs.get(self.state_key) if isinstance(obs, Mapping) else None
        if st is None:
            return None
        try:
            s = np.asarray(st)
        except (TypeError, ValueError):
            return None
        if s.ndim != 1 or s.size != self.state_mean.size or not np.all(np.isfinite(s.astype(np.float64))):
            return None
        return s

    def sent_state(self, s: np.ndarray, sign: int) -> np.ndarray:
        """State actually sent on a retry (same dtype as the observed state)."""
        return (s.astype(np.float64) + sign * self.delta).astype(s.dtype)

    def tokens_changed(self, s: np.ndarray, s_sent: np.ndarray) -> int:
        return int(np.sum(state_tokens(s, self.state_mean, self.state_std)
                          != state_tokens(s_sent, self.state_mean, self.state_std)))

    def rebase(self, raw: np.ndarray, s: np.ndarray, s_sent: np.ndarray) -> np.ndarray:
        """C-R3: executed dims 0-5 = served - (s_sent - s)[:6]; gripper (and any other dim) unchanged."""
        out = np.array(raw, dtype=np.float64, copy=True)
        out[:, :6] -= (s_sent.astype(np.float64) - s.astype(np.float64))[:6]
        return out.astype(np.asarray(raw).dtype)

    def is_zero(self, chunk: np.ndarray, state: np.ndarray) -> bool:
        c = np.asarray(chunk)
        if c.ndim != 2 or c.shape[1] != self.action_mean.size or not np.all(np.isfinite(c)):
            return False
        return is_openpi_zero_chunk(c, np.asarray(state, dtype=np.float64), self.action_mean, self.zero_tol)

    # ---- one decision -------------------------------------------------------------------------------------------
    async def decide(self, ctx) -> int:
        j = self.stats["decisions"]
        self.stats["decisions"] += 1
        obs = ctx.observe()
        native = np.array((await ctx.policy.sample(1))[0].actions)
        s = self.state_vector(obs)
        chosen = native
        rec: dict = {"decision": j, "native_zero": False, "retries": [], "selected": "native"}
        if s is not None and self.is_zero(native, s):
            self.stats["native_zero"] += 1
            rec["native_zero"] = True
            for k in range(self.max_retries):
                sign = self.shift_bins[k]
                s_sent = self.sent_state(s, sign)
                changed = self.tokens_changed(s, s_sent)
                raw = np.array((await ctx.policy.sample(1, obs={self.state_key: s_sent}))[0].actions)
                self.stats["retry_calls"] += 1
                zero_sent = self.is_zero(raw, s_sent)                  # C-R1: test against the state actually sent
                self.stats["retry_zero" if zero_sent else "retry_valid"] += 1
                self.stats["retry_tokens_unchanged"] += changed == 0
                rec["retries"].append({"k": k + 1, "shift_bins": sign, "tokens_changed": changed,
                                       "zero_for_sent_state": zero_sent,
                                       "zero_for_true_state": self.is_zero(raw, s)})
                if not zero_sent:
                    chosen = self.rebase(raw, s, s_sent)
                    rec["selected"] = f"retry{k + 1}"
                    key = f"retry{k + 1}"
                    self.stats["selected_retry"][key] = self.stats["selected_retry"].get(key, 0) + 1
                    break
            else:
                self.stats["fallback_native_zero"] += 1
                rec["selected"] = "native_zero"
        out = self.op.transform(chosen, obs, ctx.memory)
        d = self.op.diagnostics()
        rec["tilt_guard"] = {k: d.get(k) for k in ("zero", "zero_run", "z_mode", "boundary", "tilt_deg", "active",
                                                   "sat_only_would_act")}
        if "envelope" in d:
            rec["tilt_guard"]["envelope"] = {k: d["envelope"].get(k) for k in ("mode", "route", "push", "k_source")}
        rec["stats"] = {**self.stats, "selected_retry": dict(self.stats["selected_retry"])}
        ctx.record("redecode", **rec)
        return await ctx.execute(np.asarray(out)[: self.exec_rows])

    async def run(self, ctx) -> None:
        while not ctx.ended:
            await self.decide(ctx)
