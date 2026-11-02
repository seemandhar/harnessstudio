"""Public-proprio geometry, boundary detector and the deterministic re-home law for vlm-rehome (pure functions).

Conventions (vla-eval LIBERO with quat_no_antipodal, openpi pi0_fast_libero): obs['states'] = [eef pos x,y,z (m),
eef axis-angle rx,ry,rz (rad, robosuite convention, may exceed pi), finger qpos left, right]; finger width =
states[6] - states[7]. Actions: [dx,dy,dz, drx,dry,drz, gripper] in [-1, 1]; the benchmark's robosuite OSC_POSE controller
(control_delta, osc_pose.json in the LIBERO image: output_max 0.05 m and 0.5 rad, uncouple_pos_ori) maps them to a goal
offset of 0.05*a_pos m and a WORLD-frame rotation of 0.5*a_rot (goal R = exp(0.5*a_rot) R), gripper +1 close / -1 open.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def rotmat(aa: Sequence[float]) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    th = float(np.linalg.norm(aa))
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def logmap(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector (rad) of a rotation matrix, angle in [0, pi]; robust near pi."""
    R = np.asarray(R, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        # near pi: axis from the symmetric part
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        k = B[:, i] / np.sqrt(max(B[i, i], 1e-12))
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return w * th


def geodesic_deg(aa_a, aa_b) -> float:
    return float(np.degrees(np.linalg.norm(logmap(rotmat(aa_a) @ rotmat(aa_b).T))))


def tilt_deg(aa) -> float:
    """Angle of the gripper approach axis (EE z) from straight down (r1-domain / tilt_guard convention)."""
    z = rotmat(aa)[:, 2]
    return float(np.degrees(np.arccos(np.clip(-z[2], -1.0, 1.0))))


def is_zero_chunk(chunk: np.ndarray, state: np.ndarray, action_mean: np.ndarray, tol: float) -> bool:
    """openpi FAST decode failure: every row identical and dims 0-5 == Unnormalize(0) (= mean) + state (dims 0-5)."""
    c = np.asarray(chunk, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] < 6 or not np.all(np.isfinite(c)):
        return False
    if not np.all(np.abs(c - c[0:1]) <= tol):
        return False
    return bool(np.all(np.abs(c[0, :6] - (np.asarray(action_mean, float)[:6] + np.asarray(state, float)[:6])) <= tol))


class BoundaryDetector:
    """r1-domain boundary (scripts-domain/boundary_and_custody.py), streamed per decision: the first policy decision
    whose gripper command is open (<= 0) and which is not a zero chunk, after >= min_close consecutive close decisions
    (command > 0) with finger width > hold_width. A zero-chunk open decision neither counts nor ends the hold; a close
    decision with width <= hold_width resets it. After a boundary the counter restarts (at most max_boundaries)."""

    def __init__(self, min_close: int = 4, hold_width: float = 0.004, max_boundaries: int = 2) -> None:
        self.min_close, self.hold_width, self.max_boundaries = int(min_close), float(hold_width), int(max_boundaries)
        self.run = 0
        self.count = 0

    def update(self, gripper_cmd: float, width: float, zero_chunk: bool) -> bool:
        if self.count >= self.max_boundaries:
            return False
        if gripper_cmd > 0 and width > self.hold_width:
            self.run += 1
        elif gripper_cmd <= 0 and not zero_chunk:
            if self.run >= self.min_close:
                self.run = 0
                self.count += 1
                return True
            self.run = 0
        elif gripper_cmd <= 0 and zero_chunk:
            pass
        else:
            self.run = 0
        return False


def rehome_action(p, aa, p0, aa0, *, z_first: bool, kp: float, kr: float, pos_scale: float, rot_scale: float,
                  gripper_open: float, lift_tol: float) -> tuple:
    """One re-home step. Position target: lift first to max(z, z0) holding x,y (until within lift_tol of it), then p0.
    a_pos = clip(kp * (target - p) / pos_scale), a_rot = clip(kr * log(R0 R^T) / rot_scale) (world frame), gripper open.
    Returns (action[7], lifting) where lifting says whether the lift phase is still active."""
    p, p0 = np.asarray(p, float), np.asarray(p0, float)
    lifting = bool(z_first and p[2] < p0[2] - lift_tol)
    target = np.array([p[0], p[1], p0[2]]) if lifting else p0
    a = np.zeros(7)
    a[:3] = np.clip(kp * (target - p) / pos_scale, -1.0, 1.0)
    a[3:6] = np.clip(kr * logmap(rotmat(aa0) @ rotmat(aa).T) / rot_scale, -1.0, 1.0)
    a[6] = gripper_open
    return a, lifting


def home_reached(p, aa, p0, aa0, tol_m: float, tol_deg: float) -> bool:
    return bool(np.linalg.norm(np.asarray(p0, float) - np.asarray(p, float)) <= tol_m
                and geodesic_deg(aa0, aa) <= tol_deg)


def state_vec(obs, key: str = "states") -> Optional[np.ndarray]:
    st = obs.get(key) if isinstance(obs, dict) else None
    if st is None:
        return None
    try:
        s = np.asarray(st, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if s.size < 8 or not np.all(np.isfinite(s[:8])):
        return None
    return s
