"""Pure geometry for the boundary tilt envelope (B) and its yaw-matched control (development-only).

Conventions (verified on recorded development traces by the domain critic and by geometry test T1):
- LIBERO public state ``states`` = [eef xyz (3), eef axis-angle (3), gripper finger qpos (2)]; with
  ``quat_no_antipodal: true`` the axis-angle may have |aa| in (pi, 2*pi); Rodrigues is valid there (T2).
- The gripper approach axis is u = R(aa)[:, 2]; at the LIBERO start pose it points straight down (tilt ~3 deg).
- tilt = angle between u and d = -e_z (straight down), in [0, pi].
- robosuite 1.4 OSC delta rotation is a WORLD-frame axis-angle increment, left-multiplied (R_next ~ Exp(w) R), with
  w = 0.5 rad * clip(a_rot, -1, 1) per component (output_max 0.5).
- k = normalize(u x d). A rotation about +k moves u toward d, so the first-order tilt change of a world rotation w is
  dtilt = -(w . k): the component of a command along -k increases tilt, along +k decreases it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

DOWN = np.array([0.0, 0.0, -1.0])
EZ = np.array([0.0, 0.0, 1.0])


def rotmat(aa) -> np.ndarray:
    """Rodrigues: rotation matrix of an axis-angle vector (valid for any |aa|)."""
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(aa))
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def log_so3(M: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (principal branch, angle in [0, pi])."""
    M = np.asarray(M, dtype=np.float64)
    c = float(np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-4:
        w, v = np.linalg.eigh((M + M.T) / 2.0)
        return v[:, int(np.argmax(w))] * th
    return th / (2.0 * np.sin(th)) * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def tilt_of_matrix(R: np.ndarray) -> float:
    """Tilt (radians) of the approach axis R[:, 2] from straight down."""
    return float(np.arccos(np.clip(-float(R[2, 2]), -1.0, 1.0)))


def tilt_rad(aa) -> float:
    return tilt_of_matrix(rotmat(aa))


def envelope_axis(R: np.ndarray, prev_k: Optional[np.ndarray], eps: float) -> Tuple[np.ndarray, str]:
    """k = normalize(u x d) with a deterministic fallback near the singularity (critic C6).

    |u x d| < eps (tilt within ~eps rad of 0 or pi): k = previous decision's k if one exists, else the horizontal
    projection of the EE x-axis (orthogonal to u, so horizontal when u is vertical), else +x. Returns (k, source)."""
    u = R[:, 2]
    c = np.cross(u, DOWN)
    n = float(np.linalg.norm(c))
    if n >= eps:
        return c / n, "cross"
    if prev_k is not None:
        pk = np.asarray(prev_k, dtype=np.float64)
        pn = float(np.linalg.norm(pk))
        if np.isfinite(pn) and pn > 0.5:
            return pk / pn, "previous"
    x = np.array([R[0, 0], R[1, 0], 0.0])
    xn = float(np.linalg.norm(x))
    if xn >= eps:
        return x / xn, "x_axis"
    return np.array([1.0, 0.0, 0.0]), "unit_x"


def push_magnitude(tilt: float, theta: float, kappa: float, omega_max: float) -> float:
    """Bounded restoring command (action units) about +k: min(kappa * (tilt - theta), omega_max), tilt/theta in rad."""
    return float(min(kappa * (tilt - theta), omega_max))


def bte_rotation_rows(rot: np.ndarray, k: np.ndarray, push: float):
    """Apply the envelope to rotation rows [n, 3] (critic C7 order).

    1. clip every component to [-1, 1] (robosuite's own per-component input clip);
    2. remove the tilt-increasing part: if (c . k) < 0, c <- c - (c . k) k;
    3. add the push about +k;
    4. re-clip per component to [-1, 1]. If that re-clip made (c . k) negative (possible only when several
       components saturate), fall back to scaling the step-3 vector uniformly into the box, which keeps its
       direction and so a k-component >= push / max|c| > 0.
    Returns (new_rows [n, 3], removed [n] (the negative k-part removed, <= 0), rescaled [n] bool)."""
    rot = np.asarray(rot, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    out = np.empty_like(rot)
    removed = np.zeros(len(rot))
    rescaled = np.zeros(len(rot), dtype=bool)
    for i, row in enumerate(rot):
        c = np.clip(row, -1.0, 1.0)
        kc = float(c @ k)
        if kc < 0.0:
            c = c - kc * k
            removed[i] = kc
        c2 = c + push * k
        c3 = np.clip(c2, -1.0, 1.0)
        if float(c3 @ k) < 0.0:
            c3 = c2 / max(1.0, float(np.max(np.abs(c2))))
            rescaled[i] = True
        out[i] = c3
    return out, removed, rescaled


def yaw_rotation_rows(rot: np.ndarray, norms: np.ndarray, sign: int) -> np.ndarray:
    """Yaw-matched control: clip(rot) + sign * norm_i * e_z per row (tilt-neutral: e_z . k = 0 for horizontal k).
    Not re-clipped, so the per-row delta norm equals norm_i exactly (robosuite clips per component afterwards; the
    realized norm after that clip is logged by the op)."""
    base = np.clip(np.asarray(rot, dtype=np.float64), -1.0, 1.0)
    return base + float(sign) * np.asarray(norms, dtype=np.float64)[:, None] * EZ[None, :]


def commanded_tilts(R: np.ndarray, rot_rows: np.ndarray, gain: float = 1.0, scale: float = 0.5) -> list:
    """Tilt (rad) after composing each row's world-frame commanded rotation Exp(gain*scale*clip(a)) R, cumulatively
    (a commanded-pose computation for tests T3/T4; no physics)."""
    ts = []
    M = np.asarray(R, dtype=np.float64)
    for r in np.asarray(rot_rows, dtype=np.float64):
        M = rotmat(gain * scale * np.clip(r, -1.0, 1.0)) @ M
        ts.append(tilt_of_matrix(M))
    return ts
