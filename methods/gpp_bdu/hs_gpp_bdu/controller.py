"""gpp-bdu controller: the frozen gpp controller (hs_gpp v1.1, imported READ-ONLY behind identity guards) plus the bdu
module (bdu.py). Isolation: hs_gpp is imported by running gpp-core / ge-gpp-core screens, so it is never edited; this
package subclasses hs_gpp.controller.GppController, reuses its rules, primitives, witnesses and greedy-escape guard, and
overrides only run() - the parent loop line for line plus the bdu hooks. With bdu off, or before the first bdu fire,
every request and executed action equals the parent's (tests/test_stack.py proves it on the fake stack; the live CRN
audit of the screen checks it against the ge-gpp-core / greedy-escape / gpp-core records).

Precedence (BDU-RC3): an active GE window, primitive or bdu window runs to completion; otherwise H2, then H1, then bdu,
then the GE trigger. bdu is not evaluated inside GE windows. All executed (edited) actions feed every monitor. A close
commanded inside a bdu window is executed natively, ends the window and is then eligible for H2/H1 as usual. A bdu
window decodes greedily (sampler None); a GE window that starts at the same step as a bdu fire is consumed by the bdu
window (disclosed; the A5 replay shows it never happens on the greedy map).

Evidence: ctx.record("bdu", phase=fire|edit|end) and accounting.extra["gpp"]: the parent's summary fields, bdu fires in
the same `fires` list (module "bdu"; frames fire/start/end/+10/+30; result: steps, edited steps, blocked z, realized vs
commanded lateral, cos; witness P2: EE >= 1 cm below the blocked z or a settled held close within 40 steps after the
window), bdu_fires, bdu_edited_steps.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

import hs_gpp
from hs_gpp.controller import GppController
from hs_gpp.rules import EMPTY, Tracker, proprio

from .bdu import BduMonitor, BduWindow

# Identity of the frozen components (hs_gpp v1.1 = impl/gpp-v11/report.json; greedy-escape pinned by hs_gpp's G3 guard).
GPP_FILES = {"controller.py": "7efb500cbb2bd394b1b889d5dc50089de7964f52a0d6b5c200f471190cefcc8d",
             "rules.py": "7974e68a4c29c0d0e88e9a89b1d6d0f188f3d3aa94f51fadef131dab8fe46983",
             "__init__.py": "8003afaa7d4a4bac0cb823066c742e024e02f5c6284dc6d74e13f8c33f3360c7",
             "paper_gpp_core.yaml": "c473795d9dc66b160daff924ecabf02f92a8d1245aeb10176c45a7accb440592",
             "paper_ge_gpp_core.yaml": "be5c518876df422c1e50b40daf9c5b99b02582a6526b46a93f3099c2ab0f2cb7"}
PARENT_FINGERPRINTS = {"gpp-core": "6c80527fa03e49812fdc723fd56ab8df32fe9dba9a072dfad39d0bd179f4efe4",
                       "ge-gpp-core": "91c88e6ff30086cbe5400cb0bb170b1c6054823309e31d9f77cb6658d421f09a",
                       "greedy-escape": "f03731f11b84f97306ba5080b1def6b72e64468503a25813f46a700e6f9f9fe1"}
BDU_MODES = {"on", "sham", "off"}


def check_parent_identity() -> None:
    from harnessstudio.core import registry
    root = Path(hs_gpp.__file__).resolve().parent
    for name, digest in GPP_FILES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"hs_gpp {name} differs from the frozen gpp v1.1 ({root})")
    for name, fp in PARENT_FINGERPRINTS.items():
        if registry.load(name).fingerprint() != fp:
            raise ValueError(f"{name} fingerprint differs from the frozen {fp[:8]}")


class GppBduController(GppController):
    def __init__(self, modules: Mapping[str, Any], bdu: Mapping[str, Any], **parent: Any) -> None:
        check_parent_identity()
        m = dict(modules)
        mode = m.pop("bdu", None)
        if mode not in BDU_MODES:
            raise ValueError(f"modules.bdu must be one of {sorted(BDU_MODES)}")
        super().__init__(modules=m, **parent)
        self.modes["bdu"] = mode
        self.bdup = dict(bdu)
        self._new_bdu()                              # validates the parameters

    def _new_bdu(self) -> BduMonitor:
        p = self.bdup
        return BduMonitor(n=p["n"], az=p["az"], dz=p["dz"], max_fires=p["max_fires"])

    def _window(self, tr: Tracker) -> BduWindow:
        p = self.bdup
        return BduWindow(tr.xyz[tr.k], tr.z[tr.k], sham=self.modes["bdu"] == "sham", max_steps=p["window"],
                         lift_az=p["lift_az"], lateral=p["lateral"])

    @staticmethod
    def _bdu_witness(tr: Tracker, fire: dict, horizon: int = 40) -> dict:
        e, z0 = fire.get("end"), (fire.get("result") or {}).get("z_blocked")
        if e is None or z0 is None:
            return {}
        below = held = None
        for t in range(e + 1, min(len(tr.A), e + 1 + horizon)):
            if below is None and tr.z[t] is not None and tr.z[t] < z0 - 0.01:
                below = t - e
            if (held is None and t >= 2 and tr.A[t][6] > 0 and tr.A[t - 1][6] > 0 and tr.A[t - 2][6] > 0
                    and None not in (tr.w[t], tr.w[t - 1], tr.w[t - 2]) and tr.w[t] > EMPTY
                    and abs(tr.w[t] - tr.w[t - 1]) < 5e-4 and abs(tr.w[t - 1] - tr.w[t - 2]) < 5e-4):
                held = t - e
        return {"p2_below_1cm_after": below, "p2_settled_held_close_after": held,
                "p2": below is not None or held is not None, "p2_complete": len(tr.A) >= e + 1 + horizon}

    # ---- the parent's run() loop with the bdu hooks (marked BDU) ---------------------------------------------------
    async def run(self, ctx) -> None:
        h2t, h1t, h3t = self._new_triggers()
        p0 = proprio(ctx.observe().get(self.state_key))
        tr = Tracker(None if p0 is None else p0[2])
        gmon = None
        if self.ge is not None:
            StallMonitor, thresholds, escape = self.ge
            gmon = StallMonitor(**thresholds)
        fires: list = []
        state = {"scripted": 0, "decisions": 0, "width_missing": 0, "bdu_edited": 0}
        prim = None
        pending: Optional[tuple] = None
        bmon = self._new_bdu() if self.modes["bdu"] != "off" else None          # BDU
        bwin: Optional[BduWindow] = None                                        # BDU
        bpending: Optional[dict] = None                                          # BDU

        def summary() -> dict:
            out = {"modes": dict(self.modes), "fires": [], "scripted_steps": state["scripted"],
                   "policy_decisions": state["decisions"], "width_missing_steps": state["width_missing"],
                   "steps": len(tr.A)}
            for f in fires:
                wit = self._bdu_witness(tr, f) if f["module"] == "bdu" else self._witness(tr, f)
                out["fires"].append({**f, **wit})
            for mod in ("h2", "h1", "h3", "bdu"):
                out[f"{mod}_fires"] = sum(1 for f in fires if f["module"] == mod)
            out["bdu_edited_steps"] = state["bdu_edited"]
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

        def feed_bdu(action, prop, ge_window: bool) -> None:                    # BDU
            if bmon is not None:
                bmon.feed(action, None if prop is None else prop[0], reset=ge_window)

        def bdu_check(ge_window: bool) -> Optional[dict]:                       # BDU: after H2/H3, before GE
            if bmon is None or ge_window or not bmon.ready():
                return None
            bmon.fired()
            return {"detect": tr.k, "z": tr.z[tr.k], "count": self.bdup["n"]}

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

        def bdu_start(ev: dict) -> BduWindow:                                   # BDU
            w = self._window(tr)
            fire = {"module": "bdu", "variant": w.variant, "trigger_at": ev["detect"], "start": len(tr.A),
                    "end": None, "outcome": None, "trigger": dict(ev)}
            fires.append(fire)
            ctx.record("bdu", phase="fire", variant=w.variant, trigger_at=ev["detect"], start_at=len(tr.A),
                       z_blocked=ev["z"])
            return w

        def bdu_finish(w: BduWindow) -> None:                                   # BDU
            f = fires[-1]
            f["end"], f["outcome"], f["primitive_steps"] = tr.k, w.outcome, w.steps
            f["result"] = w.info()
            f["frames"] = {"fire": f["trigger_at"], "start": f["start"], "end": tr.k, "plus10": tr.k + 10,
                           "plus30": tr.k + 30}
            bmon.window_ended()
            ctx.record("bdu", phase="end", outcome=w.outcome, end_at=tr.k, steps=w.steps, edited=w.edited,
                       realized_lateral_m=w.realized_xy)

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
            if prim is None and bwin is None and pending is not None:
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
                feed_bdu(a, prop, window_before)                                 # BDU
                prim.after(prop)
                ctx.record("gpp", phase="scripted", module=prim.module, variant=prim.variant, at=tr.k,
                           a_z=float(a[2]), grip=float(a[6]))
                if prim.outcome is not None:
                    finish(prim)
                    prim = None
                    gw = bool(gmon is not None and gmon.in_window)
                    pending = post_step_triggers(gw)
                    if pending is None:                                          # BDU
                        bpending = bdu_check(gw)
                extra["gpp"] = summary()
                continue
            if bwin is None and bpending is not None:                            # BDU: window starts
                bwin = bdu_start(bpending)
                bpending = None
            sampler = self.ge[2] if (window_before and bwin is None) else None   # BDU: bdu window decodes greedily
            proposals = await ctx.policy.sample(1, sampler=sampler)
            state["decisions"] += 1
            extra["gpp"] = summary()           # refreshed before the action leaves (the last one never returns, F8)
            chunk = np.asarray(proposals[0].actions)
            if chunk.ndim != 2 or chunk.shape[0] != 1:
                raise ValueError(f"gpp is defined for one action per request (OpenVLA); got {chunk.shape}")
            a = chunk[0]
            if bwin is not None:                                                 # BDU: window decision
                a_exec, edited = bwin.edit(a)
                if await ctx.execute(a_exec[None, :]) < 1:
                    break
                state["bdu_edited"] += int(edited)
                prop = observe(a_exec)
                if gmon is not None:
                    gmon.step(a_exec, None if prop is None else prop[1])
                feed_bdu(a_exec, prop, window_before)
                bwin.after(edited, None if prop is None else prop[2])
                if edited:
                    ctx.record("bdu", phase="edit", at=tr.k, variant=bwin.variant, a_z=float(a_exec[2]),
                               a_xy=[float(a_exec[0]), float(a_exec[1])])
                if bwin.outcome is not None:
                    bdu_finish(bwin)
                    bwin = None
                    gw = bool(gmon is not None and gmon.in_window)
                    pending = post_step_triggers(gw)
                    if pending is None:
                        bpending = bdu_check(gw)
                extra["gpp"] = summary()
                continue
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
            feed_bdu(a, prop, window_before)                                     # BDU
            pending = post_step_triggers(window_before)
            if pending is None:                                                  # BDU
                bpending = bdu_check(window_before)
            extra["gpp"] = summary()
