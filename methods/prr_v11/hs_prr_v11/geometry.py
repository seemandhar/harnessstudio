"""Rotation helpers for PRR (numpy only; no scipy in the HarnessStudio venv).

Conventions (verified on dev80 identity rows, implementation/prr/calibration/calibration.json): LIBERO public
``states`` = [eef pos 3 (world), eef axis-angle 3 (quat_no_antipodal, angle may exceed pi), finger qpos 2]; the
robosuite 1.4 OSC_POSE delta rotation is WORLD-frame (median cosine realized vs commanded 0.944 world, -0.41 tool frame).
"""
from __future__ import annotations

import numpy as np


def exp(aa) -> np.ndarray:
    """Rodrigues: axis-angle (any angle) -> rotation matrix."""
    aa = np.asarray(aa, dtype=np.float64)
    th = float(np.linalg.norm(aa))
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * K @ K


def log(R) -> np.ndarray:
    """Rotation matrix -> axis-angle vector with angle in [0, pi]."""
    R = np.asarray(R, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-4:
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        k = B[:, i] / np.sqrt(max(B[i, i], 1e-12))
        k = k / np.linalg.norm(k)
        if w @ k < 0:
            k = -k
        return k * th
    return w / (2.0 * np.sin(th)) * th


def tilt_deg(aa) -> float:
    """Angle (deg) between the tool z-axis and world -z (analysis/scripts-r6/collect.py tilt_deg, per state)."""
    R = exp(aa)
    return float(np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))))


def rot_error(aa_target, aa_current) -> np.ndarray:
    """World-frame log map log(R_target R_current^T): the rotation that takes the current pose to the target."""
    return log(exp(aa_target) @ exp(aa_current).T)


def gsign(command) -> float:
    """vla-eval LIBERO discretization of the gripper command: < 0 -> -1 (open), otherwise +1 (closed)."""
    return -1.0 if float(command) < 0.0 else 1.0


def rot_z(rows_xyz: np.ndarray, sign: int) -> np.ndarray:
    """Rotate position rows [N, 3] by sign * 90 deg about world z ((x, y) -> (-y, x) for +90)."""
    r = np.asarray(rows_xyz, dtype=np.float64)
    out = r.copy()
    if sign > 0:
        out[:, 0], out[:, 1] = -r[:, 1], r[:, 0]
    else:
        out[:, 0], out[:, 1] = r[:, 1], -r[:, 0]
    return out
