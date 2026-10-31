"""Harness VLA primitive library (paper Table 1) on HarnessStudio's ctx, ported from RPent robots/libero/tools.py.

Analytic primitives read only public proprioception: `states` = [eef pos3, eef axis-angle3, gripper qpos2]
(vla-eval LIBERO / LIBERO-Pro). RPent reads `robot0_eef_quat` from the raw simulator observation; here the
same rotation comes from the published axis-angle. Every primitive acts through ctx.execute / ctx.decide,
so budgets, traces and the shadow null apply. Every action sets the gripper explicitly (vla-eval
binarises the gripper: a 0 command would close it).

Constants follow RPent's LIBERO OSC_POSE scaling: 0.05 m per unit position action, 0.5 rad per unit
rotation action (move_to/move_pose yaw use RPent's 0.10 normaliser as in the original).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from harnessstudio.skills import eef_delta as sk

ACTION_SCALE = 0.05     # metres per unit position action
ROT_SCALE = 0.5         # radians per unit rotation action (rotate_wrist / rotate_pitch)
STALL_WINDOW = 10       # move_to / move_pose: check progress every 10 steps ...
STALL_PROGRESS = 0.004  # ... and stop early if the eef moved < 4 mm (horizon adaptation; RPent runs to max_steps)


def _state(ctx) -> Optional[np.ndarray]:
    obs = ctx.observe()
    value = obs.get("states", obs.get("state"))   # LIBERO / LIBERO-Pro: "states"; fake tier-1 stack: "state"
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1)


def rotation_matrix(axis_angle) -> np.ndarray:
    """Rodrigues: axis-angle (any magnitude) -> 3x3 rotation."""
    v = np.asarray(axis_angle, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * (kx @ kx)


def yaw_of(R: np.ndarray) -> float:
    return float(math.atan2(R[1, 0], R[0, 0]))


def pitch_of(R: np.ndarray) -> float:
    return float(math.atan2(R[1, 2], -R[2, 2]))


def wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def proprio(ctx) -> dict:
    """Coordinate-free robot state for the planner (RPent state view: eef pose + gripper opening)."""
    s = _state(ctx)
    if s is None:
        pos = ctx.embodiment.eef_position(ctx.observe())
        return {"eef_pos": None if pos is None else [round(float(x), 4) for x in pos]}
    out = {"eef_pos": [round(float(x), 4) for x in s[:3]]}
    if len(s) >= 6:
        R = rotation_matrix(s[3:6])
        out["eef_yaw_rad"] = round(yaw_of(R), 3)
        out["eef_pitch_rad"] = round(pitch_of(R), 3)
    if len(s) >= 8:
        out["gripper_opening_m"] = round(float(abs(s[6]) + abs(s[7])), 4)  # ~0.08 open, ~0 closed on air
    return out


def _pos(ctx) -> np.ndarray:
    pos = ctx.embodiment.eef_position(ctx.observe())
    if pos is None:
        raise ValueError("this benchmark does not expose the end-effector position")
    return np.asarray(pos, dtype=np.float64)


def _rot(ctx) -> Optional[np.ndarray]:
    s = _state(ctx)
    if s is None or len(s) < 6:
        return None
    return rotation_matrix(s[3:6])


def _grip_open(ctx) -> Optional[float]:
    s = _state(ctx)
    if s is None or len(s) < 8:
        return None
    return float(abs(s[6]) + abs(s[7]))


def _gripper(value) -> float:
    if value in ("open", "close"):
        return -1.0 if value == "open" else 1.0
    return 1.0 if float(value) >= 0 else -1.0


def _blank(ctx) -> np.ndarray:
    return np.zeros(int(ctx.embodiment.action_dim or 7), dtype=np.float32)


def _dims(ctx, name: str) -> list:
    return ctx.embodiment.dims(name)


def _xyz(xyz) -> np.ndarray:
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError('xyz must be a list of three numbers, e.g. "xyz": [-0.05, 0.0, 1.05]')
    return np.asarray([float(v) for v in xyz], dtype=np.float64)


def _ended(ctx) -> dict:
    return {"episode_ended": bool(ctx.ended), "steps_left": int(ctx.remaining)}


async def move_to(ctx, xyz, gripper=-1.0, step_clip: float = 0.025, tol: float = 0.012, max_steps: int = 80,
                  target_yaw: Optional[float] = None, yaw_step_clip: float = 0.1, yaw_tol: float = 0.02,
                  **_: Any) -> dict:
    """RPent move_to: servo the EEF to a world xyz, holding orientation (or also converging yaw)."""
    target = _xyz(xyz)
    g = _gripper(gripper)
    start = _pos(ctx)
    if abs(target[0] - start[0]) > 0.30 or abs(target[1] - start[1]) > 0.30:
        return {"error": "refused: |dx| or |dy| > 0.30 in one move_to flips the OSC IK; add a waypoint",
                "eef_pos": [round(float(v), 4) for v in start]}
    if target_yaw is None:
        # harnessstudio.skills.eef_delta.move_to is RPent's servo without yaw: action = clip(err / 0.05, +-step_clip/0.05).
        # It runs in segments of STALL_WINDOW steps so a blocked arm returns early (official 300-step horizon).
        steps, ok, stalled = 0, False, False
        while steps < int(max_steps) and not ctx.ended:
            before = _pos(ctx)
            n = min(STALL_WINDOW, int(max_steps) - steps)
            res = await sk.move_to(ctx, target, tolerance=float(tol), gain=1.0, max_steps=n, gripper=g,
                                   max_action=min(1.0, float(step_clip) / ACTION_SCALE))
            steps += int(res.steps)
            ok = bool(res.ok)
            if ok or res.steps == 0:
                break
            if float(np.linalg.norm(_pos(ctx) - before)) < STALL_PROGRESS:
                stalled = True
                break
        final = _pos(ctx)
        out = {"name": "move_to", "target_xyz": [round(float(v), 4) for v in target],
               "final_eef_pos": [round(float(v), 4) for v in final],
               "final_dist_m": round(float(np.linalg.norm(target - final[:len(target)])), 4),
               "reached": ok, "steps_used": steps, **_ended(ctx)}
        if stalled:
            out["stalled"] = ("the arm stopped moving (blocked by an object/fixture or out of reach); "
                              "approach from above or from another side")
        return out
    pdims, gdims = _dims(ctx, "position"), _dims(ctx, "gripper")
    rdims = _dims(ctx, "rotation")
    steps = 0
    yaw_err = 0.0
    while steps < int(max_steps) and not ctx.ended:
        cur = _pos(ctx)
        R = _rot(ctx)
        if R is None:
            return {"error": "yaw control needs the eef orientation in the state"}
        if math.hypot(R[0, 0], R[1, 0]) < 1e-6:
            return {"error": "cannot control yaw: eef local x axis is nearly vertical"}
        yaw_err = wrap(float(target_yaw) - yaw_of(R))
        diff = target - cur
        if float(np.linalg.norm(diff)) < tol and abs(yaw_err) < yaw_tol:
            break
        a = _blank(ctx)
        a[pdims] = np.clip(np.clip(diff, -step_clip, step_clip) / ACTION_SCALE, -1.0, 1.0)
        a[rdims[2]] = float(np.clip(np.clip(yaw_err, -yaw_step_clip, yaw_step_clip) / 0.10, -1.0, 1.0))
        a[gdims] = g
        steps += await ctx.execute(a[None, :])
    final = _pos(ctx)
    dist = float(np.linalg.norm(target - final))
    return {"name": "move_to", "target_xyz": [round(float(v), 4) for v in target],
            "final_eef_pos": [round(float(v), 4) for v in final], "final_dist_m": round(dist, 4),
            "final_yaw_error_rad": round(yaw_err, 4), "steps_used": steps, **_ended(ctx)}


async def move_pose(ctx, xyz, target_pitch: Optional[float] = None, target_yaw: Optional[float] = None,
                    gripper=-1.0, step_clip: float = 0.02, pitch_step: float = 0.08, yaw_step: float = 0.08,
                    tol: float = 0.012, ori_tol: float = 0.05, max_steps: int = 150, **_: Any) -> dict:
    """RPent move_pose: servo position and pitch/yaw simultaneously."""
    target = _xyz(xyz)
    g = _gripper(gripper)
    pdims, rdims, gdims = _dims(ctx, "position"), _dims(ctx, "rotation"), _dims(ctx, "gripper")
    steps = 0
    p_err = y_err = 0.0
    window_start, stalled = _pos(ctx), False
    while steps < int(max_steps) and not ctx.ended:
        cur = _pos(ctx)
        if steps and steps % STALL_WINDOW == 0:
            if float(np.linalg.norm(cur - window_start)) < STALL_PROGRESS and abs(p_err) < ori_tol * 4 and abs(y_err) < ori_tol * 4:
                stalled = True
                break
            window_start = cur
        R = _rot(ctx)
        if R is None:
            return {"error": "move_pose needs the eef orientation in the state"}
        p_err = 0.0 if target_pitch is None else wrap(float(target_pitch) - pitch_of(R))
        y_err = 0.0 if target_yaw is None else wrap(float(target_yaw) - yaw_of(R))
        diff = target - cur
        if float(np.linalg.norm(diff)) < tol and abs(p_err) < ori_tol and abs(y_err) < ori_tol:
            break
        a = _blank(ctx)
        a[pdims] = np.clip(np.clip(diff, -step_clip, step_clip) / ACTION_SCALE, -1.0, 1.0)
        a[rdims[0]] = float(np.clip(np.clip(p_err, -pitch_step, pitch_step) / 0.10, -1.0, 1.0))
        a[rdims[2]] = float(np.clip(np.clip(y_err, -yaw_step, yaw_step) / 0.10, -1.0, 1.0))
        a[gdims] = g
        steps += await ctx.execute(a[None, :])
    final = _pos(ctx)
    R = _rot(ctx)
    return {"name": "move_pose", "final_eef_pos": [round(float(v), 4) for v in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "final_pitch": None if R is None else round(pitch_of(R), 4), "steps_used": steps,
            **({"stalled": "the arm stopped moving"} if stalled else {}), **_ended(ctx)}


async def _rotate(ctx, which: str, target: Optional[float], delta: Optional[float], gripper, max_steps: int,
                  tol: float, step_clip: float, pos_tol: float) -> dict:
    R = _rot(ctx)
    if R is None:
        return {"error": f"rotate_{which} needs the eef orientation in the state"}
    if target is None and delta is None:
        return {"error": f"need target_{which} or delta_{which}"}
    angle = yaw_of if which == "yaw" else pitch_of
    start = angle(R)
    start_yaw = yaw_of(R)
    goal = float(target) if target is not None else start + float(delta)
    hold = _pos(ctx).copy()
    pdims, rdims, gdims = _dims(ctx, "position"), _dims(ctx, "rotation"), _dims(ctx, "gripper")
    g = _gripper(gripper)
    max_drift, steps = 0.0, 0
    while steps < int(max_steps) and not ctx.ended:
        cur = _pos(ctx)
        max_drift = max(max_drift, float(np.linalg.norm(cur - hold)))
        R = _rot(ctx)
        err = wrap(goal - angle(R))
        if abs(err) < tol:
            break
        a = _blank(ctx)
        a[pdims] = np.clip((hold - cur) / ACTION_SCALE, -1.0, 1.0)
        if which == "yaw":
            a[rdims[2]] = float(np.clip(np.clip(err, -step_clip, step_clip) / ROT_SCALE, -1.0, 1.0))
        else:
            yaw_err = wrap(start_yaw - yaw_of(R))
            a[rdims[0]] = float(np.clip(np.clip(err, -step_clip, step_clip) / ROT_SCALE, -1.0, 1.0))
            a[rdims[2]] = float(np.clip(np.clip(yaw_err, -step_clip, step_clip) / ROT_SCALE, -1.0, 1.0))
        a[gdims] = g
        steps += await ctx.execute(a[None, :])
    final = angle(_rot(ctx))
    max_drift = max(max_drift, float(np.linalg.norm(_pos(ctx) - hold)))
    final_err = wrap(goal - final)
    return {"name": f"rotate_{'wrist' if which == 'yaw' else 'pitch'}", f"start_{which}": round(start, 4),
            f"target_{which}": round(goal, 4), f"final_{which}": round(final, 4), "final_err": round(final_err, 4),
            "converged": abs(final_err) < tol, "max_pos_drift_m": round(max_drift, 4),
            "position_hold_ok": max_drift <= pos_tol, "steps_used": steps, **_ended(ctx)}


async def rotate_wrist(ctx, target_yaw: Optional[float] = None, delta_yaw: Optional[float] = None, gripper=1.0,
                       max_steps: int = 40, tol: float = 0.02, step_clip: float = 0.1, pos_tol: float = 0.03,
                       **_: Any) -> dict:
    """RPent rotate_wrist: wrist yaw set-point about world z, holding xyz."""
    return await _rotate(ctx, "yaw", target_yaw, delta_yaw, gripper, max_steps, tol, step_clip, pos_tol)


async def rotate_pitch(ctx, target_pitch: Optional[float] = None, delta_pitch: Optional[float] = None, gripper=1.0,
                       max_steps: int = 40, tol: float = 0.02, step_clip: float = 0.1, pos_tol: float = 0.03,
                       **_: Any) -> dict:
    """RPent rotate_pitch: tilt about world x, holding xyz and yaw."""
    return await _rotate(ctx, "pitch", target_pitch, delta_pitch, gripper, max_steps, tol, step_clip, pos_tol)


async def set_gripper(ctx, gripper=-1.0, steps: int = 5, **_: Any) -> dict:
    """RPent set_gripper: hold the pose and drive the gripper for `steps` steps."""
    g = _gripper(gripper)
    res = await sk.set_gripper(ctx, g, steps=int(steps))
    return {"name": "set_gripper", "gripper": g, "steps_used": int(res.steps),
            "gripper_opening_m": _grip_open(ctx), **_ended(ctx)}


async def release(ctx, max_steps: int = 20, **_: Any) -> dict:
    """RPent release: open the gripper for max_steps steps holding the pose."""
    start = _grip_open(ctx)
    res = await sk.release(ctx, steps=int(max_steps))
    return {"name": "release", "steps_used": int(res.steps), "start_gripper_opening": start,
            "final_gripper_opening": _grip_open(ctx), **_ended(ctx)}


async def vla_act(ctx, prompt: str, max_chunks: int = 12, stop: str = "object_lifted", lift_thresh: float = 0.05,
                  gripper_closed_thresh: float = 0.06, gripper_open_thresh: float = 0.0, descent_thresh: float = 0.10,
                  **_: Any) -> dict:
    """VLA ACT: the frozen policy with the planner's prompt, one decision (chunk) at a time, until the
    early-return predicate holds, the chunk budget is spent or the episode ends.
    stop="object_lifted" is RPent pi0_pick's predicate (descent >= descent_thresh, then re-ascent >=
    lift_thresh with the gripper closed on something); stop="none" is pi0_doubled (contact skills:
    drawers, knobs, pushes) and runs the whole budget unless the episode ends."""
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "vla_act needs a non-empty prompt"}
    if stop not in ("object_lifted", "none"):
        return {"error": "stop must be 'object_lifted' or 'none'"}
    start_z = float(_pos(ctx)[-1])
    min_z = peak_z = post_min_peak = start_z
    grip0 = _grip_open(ctx)
    min_grip = last_grip = grip0
    chunks = steps = 0
    descent_done = success = False
    for _i in range(int(max_chunks)):
        if ctx.ended:
            break
        steps += await ctx.decide(instruction=prompt.strip())
        chunks += 1
        z = float(_pos(ctx)[-1])
        grip = _grip_open(ctx)
        peak_z = max(peak_z, z)
        if z < min_z:
            min_z = post_min_peak = z
        else:
            post_min_peak = max(post_min_peak, z)
        if start_z - min_z >= descent_thresh:
            descent_done = True
        if grip is not None:
            min_grip = grip if min_grip is None else min(min_grip, grip)
            last_grip = grip
        if stop == "object_lifted" and grip is not None:
            closed = gripper_open_thresh <= grip < gripper_closed_thresh
            if descent_done and (post_min_peak - min_z) >= lift_thresh and closed:
                success = True
                break
    out = {"name": "vla_act", "prompt": prompt.strip(), "stop": stop, "chunks_used": chunks, "steps_used": steps,
           "max_chunks": int(max_chunks), "final_eef_pos": [round(float(v), 4) for v in _pos(ctx)],
           "final_gripper_opening": last_grip, **_ended(ctx)}
    if stop == "object_lifted":
        out.update({"lift_detected": success, "descent_m": round(start_z - min_z, 4),
                    "post_min_ascent_m": round(post_min_peak - min_z, 4),
                    "min_gripper_opening": min_grip})
    return out


PRIMITIVES = {"move_to": move_to, "move_pose": move_pose, "rotate_wrist": rotate_wrist,
              "rotate_pitch": rotate_pitch, "set_gripper": set_gripper, "release": release, "vla_act": vla_act}
