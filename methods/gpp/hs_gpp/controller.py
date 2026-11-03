"""gpp controller (HarnessStudio v2, OpenVLA libero_10 / openvla-libero-hs): native greedy base plus public-signal
scripted primitives (rules.py) and an optional greedy-escape module (hs_greedy_escape, imported read-only).

Per decision:
  * a scripted primitive is active -> execute its next action (0 policy calls; counts against the 520 horizon);
  * an H2 (or H3) fire was detected after the previous action -> start that primitive;
  * otherwise exactly ONE `await ctx.policy.sample(1, sampler=S)` (S = None: OpenVLA's native greedy request, bitwise
    the baseline's; S = the frozen greedy-escape operator only inside a GE window when the GE module is on). If H1 is
    on and the proposed action is an open that ends a qualifying clamp run, the open is vetoed and the clamp-lift (or
    its veto-only sham) starts; else the proposed action is executed.
Precedence (review compositions GE.gpp-core): an active window or primitive runs to completion; otherwise H2, then H1
(then H3), then the GE trigger. Primitive triggers are not evaluated inside GE windows. All executed actions (policy and
scripted) feed every monitor, so a failed H2 correction's empty closes count toward GE's n_empty as GE defines them.
GE windows run concurrently with a primitive that starts at the same step (the primitive's steps consume window
decisions; disclosed). Scripted steps make no policy call, so after a primitive the CRN decision index lags the step.

Public inputs only: executed actions and obs[state_key] (send_state: true). No seeds, rewards, success, simulator state,
case ids or arm. Null (shadow): the controller runs and logs primitives on the native trajectory; native actions
execute; its monitors see the actions it requested (exact until the first fire; disclosed).

Activation evidence: ctx.record("gpp", phase=fire|end|scripted, ...) and accounting.extra["gpp"] (per-fire module,
variant, trigger fields, outcome, steps, frame steps fire/end/+10/+30, H2 next-close-held, H1 reopen/transport
witnesses, scripted step count, policy decisions, width_missing_steps; GE summary when on).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .rules import EMPTY, ClampLift, H1Trigger, H2Trigger, H3Trigger, Reclose, Tracker, proprio

MODES = {"on", "sham", "off"}
DECLARATIONS = {"analytic_control", "send_state"}


# G3 (critic gpp-src-review): greedy-escape is pinned by identity - the method fingerprint and the runtime file hashes of
# the bound greedy-escape screen - not only by its parameters.
GE_FINGERPRINT = "f03731f11b84f97306ba5080b1def6b72e64468503a25813f46a700e6f9f9fe1"
GE_FILES = {"monitor.py": "ca7b8270a038def5bb2ce6007c352a574c0eb60dc0a8d021b4aa7755be677e59",
            "controller.py": "f3041ef3291c1145359326729dda60dfa0ec79877721915bae9bc73079dc0e86",
            "paper.yaml": "3554893e32513d8d1598281186974ad27755d96b676a2cc08c452c4f9dcea1c1"}


def _ge_parts(ge: Mapping[str, Any]):
    """greedy-escape module exactly as frozen in methods/greedy_escape (refuses any other identity or threshold)."""
    import hashlib
    from pathlib import Path

    import hs_greedy_escape
    from harnessstudio.core import registry
    from harnessstudio.core.card import load_card
    from hs_greedy_escape.controller import escape_ref
    from hs_greedy_escape.monitor import StallMonitor

    root = Path(hs_greedy_escape.__file__).resolve().parent
    for name, digest in GE_FILES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"greedy-escape {name} differs from the bound greedy-escape screen ({root})")
    if registry.load("greedy-escape").fingerprint() != GE_FINGERPRINT:
        raise ValueError("greedy-escape method fingerprint differs from f03731f1 (the bound screen)")

    frozen = load_card(Path(hs_greedy_escape.__file__).with_name("paper.yaml"))["params"]
    keys = ("fixed_point_steps", "fixed_point_eps", "empty_close_steps", "empty_width_m", "empty_closes",
            "window_decisions", "max_windows")
    thresholds = {k: ge[k] for k in keys}
    if thresholds != {k: frozen[k] for k in keys} or dict(ge["escape"]) != dict(frozen["escape"]):
        raise ValueError("the GE module must use greedy-escape's frozen thresholds and escape operator exactly")
    return StallMonitor, thresholds, escape_ref(ge["escape"])


class GppController:
    def __init__(self, modules: Mapping[str, Any], h2: Mapping[str, Any], h1: Mapping[str, Any],
                 h3: Mapping[str, Any], ge: Optional[Mapping[str, Any]] = None, state_key: str = "states",
                 declarations: Optional[list] = None) -> None:
        m = dict(modules)
        for k in ("h2", "h1", "h3"):
            if m.get(k) not in MODES:
                raise ValueError(f"modules.{k} must be one of {sorted(MODES)}")
        if not isinstance(m.get("ge"), bool):
            raise ValueError("modules.ge must be true or false")
        unknown = set(declarations or ()) - DECLARATIONS
        if unknown or "analytic_control" not in set(declarations or ()):
            raise ValueError("gpp declares analytic_control (scripted EE-delta actions from public proprio)")
        self.modes = m
        self.h2p, self.h1p, self.h3p = dict(h2), dict(h1), dict(h3)
        self.state_key = str(state_key)
        self._h2_trigger_keys = ("min_close", "within", "max_fires", "refire_rise")
        self._h1_trigger_keys = ("min_run", "clamp_w", "settle", "max_rise", "max_descent", "max_rot", "chain_gap",
                                 "chain_lift", "max_fires")
        self._h3_trigger_keys = ("min_step", "identical", "eps", "max_move")
        self._new_triggers()                         # validates the parameter sets
        self.ge = None
        if m["ge"]:
            if not ge:
                raise ValueError("modules.ge is on but no ge parameters were given")
            self.ge = _ge_parts(ge)

    def _new_triggers(self):
        return (H2Trigger(**{k: self.h2p[k] for k in self._h2_trigger_keys}),
                H1Trigger(**{k: self.h1p[k] for k in self._h1_trigger_keys}),
                H3Trigger(**{k: self.h3p[k] for k in self._h3_trigger_keys}))

    # ---- primitives ------------------------------------------------------------------------------------------------
    def _reclose(self, module: str, z: Optional[float]) -> Reclose:
        p = self.h2p if module == "h2" else self.h3p
        mode = self.modes[module]
        return Reclose(direction=-1 if mode == "on" else 1, open_steps=p["open_steps"], speed=p["speed"],
                       distance=p["distance"], max_move_steps=p["max_move_steps"], stop_dz=p["stop_dz"],
                       module=module, z_start=z, stop_from_move=p["stop_from_move"])

    def _lift(self, vetoed: Any, z_fire: float) -> ClampLift:
        p = self.h1p
        return ClampLift(vetoed, z_fire, lift=self.modes["h1"] == "on", speed=p["speed"], max_steps=p["max_steps"],
                         height=p["height"], check_at=p["check_at"], min_rise=p["min_rise"], empty_w=EMPTY)

    # ---- descriptive witnesses (public; for the physical falsifiers, never used for control) ------------------------
    @staticmethod
    def _witness(tr: Tracker, fire: dict) -> dict:
        e = fire.get("end")
        if e is None:
            return {}
        out: dict = {}
        if fire["module"] == "h2":
            runs = [r for r in tr.runs if r[0] > e] + ([(tr.run_start, len(tr.A))] if tr.run_start is not None
                                                         and tr.run_start > e else [])
            nxt = next((r for r in runs if r[1] - r[0] >= 10), None)
            if nxt is not None:
                c, j = nxt
                kk = min(c + 10, j - 1)
                zs = [z for z in tr.z[c:j] if z is not None]
                out["next_close"] = {"close": c, "held": bool(tr.w[kk] is not None and tr.w[kk] > EMPTY),
                                     "lift": (max(zs) - zs[0]) if zs else None, "complete": j <= len(tr.A) - 1}
        if fire["module"] == "h1" and fire.get("outcome") in ("lift_done", "lift_cap", "veto_done"):
            opens = [k for k in range(e + 1, len(tr.A)) if tr.A[k][6] < 0]
            out["reopen_after"] = (opens[0] - e) if opens else None
            stop = opens[0] if opens else len(tr.A)
            if tr.xyz[e] is not None:
                xy0 = tr.xyz[e][:2]
                moved = [float(np.linalg.norm(tr.xyz[k][:2] - xy0)) for k in range(e + 1, stop)
                         if tr.xyz[k] is not None and tr.w[k] is not None and tr.w[k] > EMPTY]
                out["max_xy_while_held"] = max(moved) if moved else 0.0
                out["transport"] = bool(moved and max(moved) >= 0.10)
        return out

    # ---- episode loop ----------------------------------------------------------------------------------------------
    async def run(self, ctx) -> None:
        h2t, h1t, h3t = self._new_triggers()
        p0 = proprio(ctx.observe().get(self.state_key))
        tr = Tracker(None if p0 is None else p0[2])
        gmon = None
        if self.ge is not None:
            StallMonitor, thresholds, escape = self.ge
            gmon = StallMonitor(**thresholds)
        fires: list = []
        state = {"scripted": 0, "decisions": 0, "width_missing": 0}
        prim = None
        pending: Optional[tuple] = None

        def summary() -> dict:
            out = {"modes": dict(self.modes), "fires": [], "scripted_steps": state["scripted"],
                   "policy_decisions": state["decisions"], "width_missing_steps": state["width_missing"],
                   "steps": len(tr.A)}
            for f in fires:
                out["fires"].append({**f, **self._witness(tr, f)})
            for mod in ("h2", "h1", "h3"):
                out[f"{mod}_fires"] = sum(1 for f in fires if f["module"] == mod)
            out["first_fire"] = ({"module": fires[0]["module"], "at": fires[0]["trigger_at"]} if fires else None)
            if gmon is not None:
                out["ge"] = gmon.summary()
            return out

        def observe(action) -> Optional[tuple]:
            prop = proprio(ctx.observe().get(self.state_key))
            if prop is None:
                state["width_missing"] += 1
            tr.update(action, prop)
            h2t.observe_z(None if prop is None else prop[0])
            return prop

        def start(module: str, prim_obj, trigger: dict, trigger_at: int) -> None:
            fire = {"module": module, "variant": prim_obj.variant, "trigger_at": trigger_at, "start": len(tr.A),
                    "end": None, "outcome": None, "trigger": {k: v for k, v in trigger.items()}}
            fires.append(fire)
            ctx.record("gpp", phase="fire", module=module, variant=prim_obj.variant, trigger_at=trigger_at,
                       start_at=len(tr.A), trigger=fire["trigger"])

        def finish(prim_obj) -> None:
            f = fires[-1]
            f["end"], f["outcome"], f["primitive_steps"] = tr.k, prim_obj.outcome, prim_obj.steps
            f["result"] = dict(prim_obj.info)
            f["frames"] = {"fire": f["trigger_at"], "start": f["start"], "end": tr.k, "plus10": tr.k + 10,
                           "plus30": tr.k + 30}
            if f["module"] == "h2":
                h2t.note_primitive_end(tr.z[tr.k])
            ctx.record("gpp", phase="end", module=f["module"], variant=f["variant"], outcome=prim_obj.outcome,
                       end_at=tr.k, primitive_steps=prim_obj.steps, result=f["result"])

        def post_step_triggers(window_before: bool) -> Optional[tuple]:
            if window_before:
                return None
            if self.modes["h2"] != "off":
                ev = h2t.check(tr)
                if ev is not None:
                    h2t.fired(ev)
                    return ("h2", ev, ev["detect"])
            if self.modes["h3"] != "off":
                ev = h3t.check(tr)
                if ev is not None:
                    h3t.fired(ev)
                    return ("h3", ev, ev["detect"])
            return None

        extra = ctx.accounting.extra
        extra["gpp"] = summary()
        while not ctx.ended:
            window_before = bool(gmon is not None and gmon.in_window)
            if prim is None and pending is not None:
                module, ev, at = pending
                pending = None
                prim = self._reclose(module, tr.z[tr.k] if tr.k >= 0 else None)
                start(module, prim, ev, at)
            if prim is not None:
                a = prim.action()
                if a is None:                     # defensive: finished without a pending step
                    finish(prim)
                    prim = None
                    continue
                if await ctx.execute(a[None, :]) < 1:
                    break
                state["scripted"] += 1
                prop = observe(a)
                if gmon is not None:
                    gmon.step(a, None if prop is None else prop[1])
                prim.after(prop)
                ctx.record("gpp", phase="scripted", module=prim.module, variant=prim.variant, at=tr.k,
                           a_z=float(a[2]), grip=float(a[6]))
                if prim.outcome is not None:
                    finish(prim)
                    prim = None
                    pending = post_step_triggers(bool(gmon is not None and gmon.in_window))
                extra["gpp"] = summary()
                continue
            proposals = await ctx.policy.sample(1, sampler=self.ge[2] if window_before else None)
            state["decisions"] += 1
            extra["gpp"] = summary()           # refreshed before the action leaves (the last one never returns, F8)
            chunk = np.asarray(proposals[0].actions)
            if chunk.ndim != 2 or chunk.shape[0] != 1:
                raise ValueError(f"gpp is defined for one action per request (OpenVLA); got {chunk.shape}")
            a = chunk[0]
            if not window_before and self.modes["h1"] != "off":
                ev = h1t.check(tr, a)
                if ev is not None:
                    h1t.fired(ev)
                    prim = self._lift(a, ev["z"])
                    start("h1", prim, ev, ev["veto"])
                    extra["gpp"] = summary()
                    continue
            if await ctx.execute(chunk) < 1:
                break
            prop = observe(a)
            if gmon is not None:
                gmon.step(a, None if prop is None else prop[1])
            pending = post_step_triggers(window_before)
            extra["gpp"] = summary()
