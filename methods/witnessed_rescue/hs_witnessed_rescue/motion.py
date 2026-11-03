"""Analytic grasp / lift / place primitives with a measured custody test, on public proprio only.

`states` = [eef xyz (grip site), eef axis-angle(3), finger qL, qR] (vla-eval LIBERO). Actions are 7-D OSC deltas:
0.05 m per unit position action, 0.5 rad per unit rotation action, gripper -1 open / +1 close (vla-eval binarises
the gripper, so every action sets it explicitly). Constants and the yaw convention follow the reviewed Harness VLA
port (its primitives module in methods/harness_vla, sha256 2fadc1f6...: rotation_matrix, yaw_of,
wrap and the +yaw_err rotation sign are vendored from there). The EE position is the grip site; no TCP offset
is guessed.

Every step goes through the controller's `step(action)` so that the public trajectory used by the triggers, the
analytic-step accounting and the framework's shadow null (which substitutes native actions) all apply.
"""
from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Mapping, Optional

import numpy as np

ACTION_SCALE = 0.05
ROT_SCALE = 0.5

DEFAULTS = {
    "step_clip": 0.05,           # commanded m per step in free motion (= action 1.0; realized ~8 mm/step on LIBERO)
    "descend_clip": 0.025,       # commanded m per step while descending to a grasp / place (action 0.5)
    "tol": 0.012,                # move_to tolerance (review A4: realized error <= 1.2 cm)
    "max_move_steps": 80,
    "max_waypoint_xy": 0.25,     # OSC IK flips beyond ~0.30 m in one move (harness_vla)
    "stall_window": 10,
    "stall_progress": 0.004,
    "align_yaw": True,
    "closing_axis_offset": math.pi / 2,   # finger closing direction = eef yaw + offset (A4 checks it)
    "yaw_step": 0.1,             # rad per step
    "yaw_tol": 0.05,
    "approach_height": 0.10,     # above the object top
    "z_close_margin": 0.035,     # z_close = z_top - min(margin, 0.5 (z_top - z_table)); review range [0.02, 0.04]
    "contact_window": 5,         # descent contact stop: < contact_progress in contact_window steps
    "contact_progress": 0.002,
    "open_steps": 4,
    "close_steps": 10,
    "lift": 0.06,                # review: one of {0.06, 0.08}
    "lift_steps": 10,            # custody: width change measured over 10 lift steps
    "lift_extra_steps": 20,      # further lift steps to reach the lift height
    "lift_tol": 0.004,           # the lift servo converges tighter than move_to so a 6 cm lift can show >= 5 cm
    "custody_min_width": 0.004,
    "custody_max_width_drop": 0.0015,
    "custody_min_rise": 0.05,
    "carry_clearance": 0.08,
    "place_offset": 0.02,
    "release_steps": 8,
    "retreat": 0.08,
}


def rotation_matrix(axis_angle) -> np.ndarray:
    v = np.asarray(axis_angle, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * (kx @ kx)


def yaw_of(R: np.ndarray) -> float:
    return float(math.atan2(R[1, 0], R[0, 0]))


def wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def grasp_yaw(current_yaw: float, short_axis: float, offset: float) -> float:
    """Eef yaw that closes the fingers across the object's short axis, equivalent mod pi, nearest the current."""
    base = short_axis - offset
    cands = [base + k * math.pi for k in range(-3, 4)]
    best = min(cands, key=lambda y: abs(wrap(y - current_yaw)))
    return float(current_yaw + wrap(best - current_yaw))


def z_close(z_top: float, z_table: float, margin: float) -> float:
    return float(z_top - min(margin, 0.5 * (z_top - z_table)))


def custody(width_after_close: float, width_lift: float, rise: float, p: Mapping[str, Any]) -> dict:
    """Measured custody: width > 4 mm after close AND width change > -1.5 mm over the lift steps AND rise >= 5 cm.
    A controller predicate: logged, never counted as physical evidence (blind frames decide that)."""
    dw = float(width_lift - width_after_close)
    ok = bool(width_after_close > p["custody_min_width"] and dw > -p["custody_max_width_drop"]
              and rise >= p["custody_min_rise"])
    return {"custody": ok, "width_after_close": round(float(width_after_close), 5),
            "width_change_lift": round(dw, 5), "rise": round(float(rise), 4)}


class Motion:
    """Primitives bound to one episode. `step(action) -> executed (0/1)`, `state() -> np.ndarray | None`,
    `ended() -> bool`."""

    def __init__(self, step: Callable[[np.ndarray], Awaitable[int]], state: Callable[[], Optional[np.ndarray]],
                 ended: Callable[[], bool], params: Optional[Mapping[str, Any]] = None) -> None:
        p = dict(DEFAULTS)
        p.update(dict(params or {}))
        unknown = set(p) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown motion params {sorted(unknown)}")
        self.p, self._step, self._state, self._ended = p, step, state, ended
        self.gripper = -1.0
        self.steps = 0

    # -- state ---------------------------------------------------------------------------------------------
    def pos(self) -> np.ndarray:
        s = self._state()
        if s is None or len(s) < 3:
            raise RuntimeError("no end-effector position in the observation")
        return np.asarray(s[:3], dtype=np.float64)

    def yaw(self) -> Optional[float]:
        s = self._state()
        return None if s is None or len(s) < 6 else yaw_of(rotation_matrix(s[3:6]))

    def width(self) -> Optional[float]:
        s = self._state()
        return None if s is None or len(s) < 8 else float(s[6] - s[7])

    async def _do(self, a: np.ndarray) -> int:
        n = await self._step(a)
        self.steps += n
        return n

    def _action(self, dpos=None, dyaw: float = 0.0, gripper: Optional[float] = None) -> np.ndarray:
        a = np.zeros(7, dtype=np.float32)
        if dpos is not None:
            a[0:3] = np.clip(np.asarray(dpos, dtype=np.float64) / ACTION_SCALE, -1.0, 1.0)
        a[5] = float(np.clip(dyaw / ROT_SCALE, -1.0, 1.0))
        a[6] = self.gripper if gripper is None else gripper
        return a

    # -- primitives -------------------------------------------------------------------------------------------
    async def hold(self, gripper: float, steps: int) -> int:
        self.gripper = float(gripper)
        done = 0
        for _ in range(int(steps)):
            if self._ended():
                break
            done += await self._do(self._action(np.zeros(3), 0.0, self.gripper))
        return done

    async def servo(self, target, yaw: Optional[float] = None, clip: Optional[float] = None,
                    max_steps: Optional[int] = None, contact_stop: bool = False, tol: Optional[float] = None) -> dict:
        """Proportional servo to an absolute eef xyz (and optional yaw), stall and contact early returns."""
        p = self.p
        target = np.asarray(target, dtype=np.float64)
        clip = float(p["step_clip"] if clip is None else clip)
        max_steps = int(p["max_move_steps"] if max_steps is None else max_steps)
        tol = float(p["tol"] if tol is None else tol)
        steps, stalled, contact, yaw_off = 0, False, False, False
        window = [self.pos()]
        yaw_window: list = []
        last_yaw_err = None
        while steps < max_steps and not self._ended():
            cur = self.pos()
            err = target - cur
            yerr = 0.0
            if yaw is not None and p["align_yaw"] and not yaw_off:
                y = self.yaw()
                if y is not None:
                    yerr = wrap(yaw - y)
                    # Guard: a yaw error that grows over the window means the sign convention does not hold here.
                    if last_yaw_err is not None and steps % 5 == 0 and abs(yerr) > last_yaw_err + 0.05:
                        yaw_off, yerr = True, 0.0
                    if steps % 5 == 0:
                        last_yaw_err = abs(yerr)
            if float(np.linalg.norm(err)) < tol and abs(yerr) < p["yaw_tol"]:
                break
            yaw_window.append(abs(yerr))
            dyaw = float(np.clip(yerr, -p["yaw_step"], p["yaw_step"]))
            steps += await self._do(self._action(np.clip(err, -clip, clip), dyaw))
            window.append(self.pos())
            if contact_stop and len(window) > 2 * p["contact_window"] and err[2] < -tol:
                if window[-1 - p["contact_window"]][2] - window[-1][2] < p["contact_progress"]:
                    contact = True
                    break
            if len(window) > p["stall_window"] and steps % p["stall_window"] == 0:
                moved = float(np.linalg.norm(window[-1] - window[-1 - p["stall_window"]]))
                turned = (yaw_window[-p["stall_window"]] - yaw_window[-1]) if len(yaw_window) >= p["stall_window"] else 0.0
                if moved < p["stall_progress"] and turned < 0.02:
                    stalled = True
                    break
        final = self.pos()
        return {"target": [round(float(v), 4) for v in target], "final": [round(float(v), 4) for v in final],
                "error": round(float(np.linalg.norm(target - final)), 4), "steps": steps, "stalled": stalled,
                "contact": contact, "yaw_disabled": yaw_off}

    async def move_to(self, target, yaw: Optional[float] = None, clip: Optional[float] = None) -> dict:
        """Waypointed move: segments of at most max_waypoint_xy in xy."""
        target = np.asarray(target, dtype=np.float64)
        cur = self.pos()
        dxy = target[:2] - cur[:2]
        n = max(1, int(math.ceil(float(np.linalg.norm(dxy)) / self.p["max_waypoint_xy"])))
        res: dict = {}
        total = 0
        for i in range(1, n + 1):
            wp = np.array([cur[0] + dxy[0] * i / n, cur[1] + dxy[1] * i / n, target[2]])
            res = await self.servo(wp, yaw=yaw, clip=clip)
            total += res["steps"]
            if self._ended() or res["stalled"]:
                break
        res["steps"] = total
        return res

    async def grasp(self, obj: Mapping[str, Any]) -> dict:
        """Open, rise to a safe height, move above the object (yaw across its short axis), descend with a
        contact stop to z_close, close, lift; then the measured custody test."""
        p = self.p
        log: dict = {"primitive": "grasp"}
        cx, cy, ztop = (float(v) for v in obj["center_xyz"])
        ztable = float(obj["z_table"])
        zc = z_close(ztop, ztable, p["z_close_margin"])
        log["z_close"] = round(zc, 4)
        yaw = None
        cur_yaw = self.yaw()
        if cur_yaw is not None and obj.get("yaw_short") is not None:
            yaw = grasp_yaw(cur_yaw, float(obj["yaw_short"]), p["closing_axis_offset"])
            log["target_yaw"] = round(yaw, 4)
        await self.hold(-1.0, p["open_steps"])
        z_safe = max(float(self.pos()[2]), ztop + p["approach_height"])
        log["rise"] = await self.servo([*self.pos()[:2], z_safe])
        log["above"] = await self.move_to([cx, cy, z_safe], yaw=yaw)
        log["descend"] = await self.servo([cx, cy, zc], yaw=yaw, clip=p["descend_clip"], contact_stop=True)
        await self.hold(1.0, p["close_steps"])
        w0 = self.width()
        log.update(await self.lift_and_test(w0))
        return log

    async def lift_and_test(self, w0: Optional[float]) -> dict:
        """Lift `lift` with the gripper closed; custody from width over the first lift_steps and the eef rise."""
        p = self.p
        self.gripper = 1.0
        z0 = float(self.pos()[2])
        target = self.pos() + np.array([0.0, 0.0, p["lift"]])
        res = await self.servo(target, max_steps=p["lift_steps"], tol=p["lift_tol"])
        w10 = self.width()
        more = await self.servo(target, max_steps=p["lift_extra_steps"], tol=p["lift_tol"])
        rise = float(self.pos()[2]) - z0
        c = custody(w0 if w0 is not None else 0.0, w10 if w10 is not None else 0.0, rise, p)
        return {"lift": {"steps": res["steps"] + more["steps"], "final": more["final"]}, **c}

    async def place(self, dest: Mapping[str, Any], h_obj: float) -> dict:
        """Carry at z_surface + h_obj + clearance, descend (contact stop) to z_surface + h_obj/2 + offset, open,
        retreat up."""
        p = self.p
        dx, dy = (float(v) for v in dest["center_xyz"][:2])
        zs = float(dest["z_surface"])
        z_carry = max(float(self.pos()[2]), zs + h_obj + p["carry_clearance"])
        z_place = zs + h_obj / 2 + p["place_offset"]
        self.gripper = 1.0
        log: dict = {"primitive": "place", "z_carry": round(z_carry, 4), "z_place": round(z_place, 4)}
        log["rise"] = await self.servo([*self.pos()[:2], z_carry])
        log["carry"] = await self.move_to([dx, dy, z_carry])
        log["descend"] = await self.servo([dx, dy, z_place], clip=p["descend_clip"], contact_stop=True)
        log["width_before_release"] = self.width()
        await self.hold(-1.0, p["release_steps"])
        log["retreat"] = await self.servo(self.pos() + np.array([0.0, 0.0, p["retreat"]]))
        return log
