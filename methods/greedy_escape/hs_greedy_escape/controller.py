"""greedy-escape controller (HarnessStudio v2 controller method, OpenVLA libero_10 / openvla-libero-hs).

Per decision (OpenVLA: one 7-D action per request, execute_horizon 1):
  1. exactly ONE `await ctx.policy.sample(1, sampler=S)` (GE3), where S = None outside an escape window (the seeded
     request the baseline arm sends: OpenVLA's native greedy predict_action, bitwise native) and S = the frozen escape
     operator inside a window (default hs_scale.decoder:ScaleDecoder at the SCALE paper defaults, GE2). One sample per
     decision keeps ctx's decision counter equal to the step index, so window decision d uses the CRN seed
     crn_seed(ns, benchmark, task, episode, d, 0): the seed the SCALE map record used at the same decision;
  2. `await ctx.execute(chunk)` (the single action);
  3. the public stall monitor (monitor.StallMonitor) is updated with the executed action and the finger width of
     the next public observation (obs[state_key][6] - obs[state_key][7]); a fire starts a 40-decision window.

The escape decoder's per-episode state lives in the policy server, keyed by (session, episode, sampler): it is
created at the first window decision (gamma = 1) and advances only on window decisions; it persists across windows
(disclosed substitution). The controller never reads seeds, rewards, success, simulator state, case ids or the arm.

Null arm (`null: shadow`): the controller runs unchanged on the shadow policy session (its greedy and window calls
are paid there) while the framework executes the native actions, so its windows are triggered on the native
trajectory. Outside windows the shadow greedy action equals the native one (greedy ignores the seed); inside
windows the monitor sees the window action it requested, not the executed native one (disclosed). The null pays
one native call per step on top of the controller's call, i.e. at least the candidate's compute.

Activation evidence: trace events "greedy_escape" (phase fire / window_end) and the running summary in
accounting.extra["greedy_escape"] (window starts, triggers, suppressed fires, escape decisions, empty-close tallies),
which the framework copies into the episode record of the candidate and null arms (the baseline arm is native and
runs no controller).
"""
from __future__ import annotations

import importlib
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .monitor import StallMonitor

DEFAULT_ESCAPE = {"factory": "hs_scale.decoder:ScaleDecoder",
                  "params": {"T0": 1.0, "epsilon": 1.0e-12, "num_logits": 256, "kappa": 2.0, "alpha": 0.8,
                             "attn_sensitivity": 0.3}}


def escape_ref(escape: Optional[Mapping[str, Any]]) -> dict:
    """Validated JSON sampler reference {factory, params} sent with window requests (server-side decoding hook)."""
    ref = dict(escape or DEFAULT_ESCAPE)
    factory = str(ref.get("factory") or "")
    module, sep, attr = factory.partition(":")
    if not sep or not module.startswith("hs_") or not attr:
        raise ValueError(f"escape factory {factory!r} must be 'hs_<pkg>.module:Class' (server sampler prefix hs_)")
    params = dict(ref.get("params") or {})
    # Build it once here (light: numpy only) so a wrong factory or parameter fails at registration (tier 0).
    getattr(importlib.import_module(module), attr)(**params)
    return {"factory": factory, "params": params}


class GreedyEscapeController:
    def __init__(self, fixed_point_steps: int = 15, fixed_point_eps: float = 0.01, empty_close_steps: int = 10,
                 empty_width_m: float = 0.004, empty_closes: int = 3, window_decisions: int = 40,
                 max_windows: int = 4, escape: Optional[Mapping[str, Any]] = None, state_key: str = "states",
                 finger_dims: Sequence[int] = (6, 7)) -> None:
        self.thresholds = dict(fixed_point_steps=int(fixed_point_steps), fixed_point_eps=float(fixed_point_eps),
                               empty_close_steps=int(empty_close_steps), empty_width_m=float(empty_width_m),
                               empty_closes=int(empty_closes), window_decisions=int(window_decisions),
                               max_windows=int(max_windows))
        StallMonitor(**self.thresholds)           # validate
        self.escape = escape_ref(escape)
        self.state_key = str(state_key)
        dims = [int(d) for d in finger_dims]
        if len(dims) != 2:
            raise ValueError("finger_dims must name the two finger joints")
        self.finger_dims = tuple(dims)

    # ---- public proprio --------------------------------------------------------------------------------------
    def finger_width(self, obs: Mapping[str, Any]) -> Optional[float]:
        st = obs.get(self.state_key) if isinstance(obs, Mapping) else None
        if st is None:
            return None
        try:
            s = np.asarray(st, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        hi = max(self.finger_dims)
        if s.size <= hi or not np.all(np.isfinite(s[list(self.finger_dims)])):
            return None
        return float(s[self.finger_dims[0]] - s[self.finger_dims[1]])

    # ---- episode loop ----------------------------------------------------------------------------------------
    async def run(self, ctx) -> None:
        mon = StallMonitor(**self.thresholds)
        extra = ctx.accounting.extra
        extra["greedy_escape"] = mon.summary()
        while not ctx.ended:
            window = mon.in_window
            proposals = await ctx.policy.sample(1, sampler=self.escape if window else None)
            chunk = np.asarray(proposals[0].actions)
            if chunk.ndim != 2 or chunk.shape[0] != 1:
                raise ValueError(f"greedy-escape is defined for one action per request (OpenVLA); got {chunk.shape}")
            executed = await ctx.execute(chunk)
            if executed < 1:
                break
            event = mon.step(chunk[0], self.finger_width(ctx.observe()))
            # record fields must not reuse the trace's own keys (t, step) or _log's parameters (kind, name)
            if event is not None:
                ctx.record("greedy_escape", phase="fire", fire_step=event["step"], reasons=event["reasons"],
                           run_len=event["run_len"], empty_count=event["empty_count"], window=event["window"])
            if window and not mon.in_window:
                ctx.record("greedy_escape", phase="window_end", last_action=mon.k, windows=len(mon.windows))
            extra["greedy_escape"] = mon.summary()
