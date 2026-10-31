"""Forward/inverse kinematics of RoboDojo's dual ARX-X5 arms (public robot description only).

FastHarness's RoboDojo runtime gives the reviewer the measured end-effector poses (`current_eef`), the
forward kinematics of every proposed joint row (`student_eef_trajectory`) and executes the reviewer's
`edit` (EEF delta) and `eef` (EEF target) takeovers. That runtime is not released (the paper's README
leaves it to the caller), so this module re-derives the same quantities from the robot's published
URDF (`Assets/Robots/x5/X5A.urdf`, revolute joints 1-6) and mount poses (`env_cfg/robot/dual_x5.yml`).
Nothing here reads simulator state: inputs are the public 14-D joint state and the policy's proposals.

Frame: world (z up, metres). Each arm root sits at `ROOTS[arm]` with a +90 deg yaw. The tool point is
`TOOL_OFFSET` along link6's x axis (between the finger joints, URDF joint7/joint8 at x = 0.0866).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

ARMS = ("left", "right")
# 14-D packing used by RoboDojo / XPolicyLab: left arm 6, left gripper 1, right arm 6, right gripper 1.
ARM_SLICES = {"left": slice(0, 6), "right": slice(7, 13)}
GRIPPER_INDEX = {"left": 6, "right": 13}

# (origin xyz, origin rpy, axis) for joint1..joint6 of X5A.urdf.
_CHAIN = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
TOOL_OFFSET = 0.10
# dual_x5.yml default_root_pos / default_root_rot (wxyz); robot 0 is the left arm (state packs it first).
ROOTS = {"left": ((-0.3, -0.45, 0.765), (0.707, 0.0, 0.0, 0.707)),
         "right": ((0.3, -0.45, 0.765), (0.707, 0.0, 0.0, 0.707))}


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(m: np.ndarray) -> list:
    """Unit wxyz quaternion with w >= 0."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def rotvec_to_matrix(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    return _axis_angle(v / angle, angle)


def matrix_to_rotvec(m: np.ndarray) -> np.ndarray:
    cos = max(-1.0, min(1.0, (np.trace(m) - 1) / 2))
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:  # near pi: take the axis from the symmetric part
        w, x, y, z = matrix_to_quat(m)
        axis = np.array([x, y, z]) / max(1e-12, math.sqrt(x * x + y * y + z * z))
        return axis * angle
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2 * math.sin(angle))
    return axis * angle


_JOINT_FIXED = [(np.array(xyz), _rpy(*rpy), np.array(axis)) for xyz, rpy, axis in _CHAIN]
_ROOT = {arm: (np.array(p), quat_to_matrix(q)) for arm, (p, q) in ROOTS.items()}


def arm_fk(arm: str, joints: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """World-frame tool position (3,) and rotation (3, 3) for one arm's six joint angles."""
    pos, rot = _ROOT[arm]
    pos, rot = pos.copy(), rot.copy()
    for (xyz, fixed, axis), q in zip(_JOINT_FIXED, joints):
        pos = pos + rot @ xyz
        rot = rot @ fixed @ _axis_angle(axis, float(q))
    return pos + rot @ np.array([TOOL_OFFSET, 0.0, 0.0]), rot


def eef_pose(arm: str, joints: Sequence[float], gripper: float) -> dict:
    pos, rot = arm_fk(arm, joints)
    return {"position": [round(float(v), 5) for v in pos],
            "quaternion_wxyz": [round(v, 6) for v in matrix_to_quat(rot)],
            "gripper_opening_command": round(float(gripper), 4)}


def current_eef(state: Sequence[float], gripper_open_value: float = 1.0) -> dict:
    """FastHarness `current_eef` from the 14-D joint state; gripper command normalized to 0 closed..1 open."""
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    return {arm: eef_pose(arm, s[ARM_SLICES[arm]], s[GRIPPER_INDEX[arm]] / gripper_open_value) for arm in ARMS}


def solve_ik(arm: str, seed: Sequence[float], position: Sequence[float], rotation: np.ndarray,
             iterations: int = 80, damping: float = 0.02, tol: float = 1e-4) -> tuple[np.ndarray, float]:
    """Damped least squares from the measured joints; returns (joints, final pose error norm)."""
    q = np.asarray(seed, dtype=np.float64).copy()
    target_p, target_r = np.asarray(position, dtype=np.float64), np.asarray(rotation, dtype=np.float64)

    def error(qv):
        p, r = arm_fk(arm, qv)
        return np.concatenate([target_p - p, 0.3 * matrix_to_rotvec(target_r @ r.T)])

    err = error(q)
    for _ in range(iterations):
        if np.linalg.norm(err) < tol:
            break
        jac = np.zeros((6, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = 1e-5
            jac[:, j] = (error(q - dq) - err) / 1e-5   # d(-error)/dq = d(fk)/dq
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(6), err)
        q = q + np.clip(step, -0.2, 0.2)
        err = error(q)
    return q, float(np.linalg.norm(err))


def takeover_actions(state: Sequence[float], response: dict, steps: int,
                     gripper_open_value: float = 1.0, gripper_closed_value: float = 0.0) -> tuple[np.ndarray, dict]:
    """14-D joint actions for a reviewer `edit` / `eef` response, interpolated over `steps` steps.

    edit: per-arm delta on the measured tool pose (position metres, rotation vector radians, world frame),
          gripper keep/open/closed. eef: absolute per-arm tool target, gripper_closed bool.
    """
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    goal = s.copy()
    info = {}
    for arm in ARMS:
        joints = s[ARM_SLICES[arm]]
        pos, rot = arm_fk(arm, joints)
        if response["mode"] == "edit":
            edit = response["edit"][arm]
            t_pos = pos + np.asarray(edit["delta_position"], dtype=np.float64)
            t_rot = rotvec_to_matrix(edit["delta_rotation_vector"]) @ rot
            grip = {"keep": s[GRIPPER_INDEX[arm]], "open": gripper_open_value,
                    "closed": gripper_closed_value}[edit["gripper"]]
        else:
            target = response["target"][arm]
            t_pos = np.asarray(target["position"], dtype=np.float64)
            t_rot = quat_to_matrix(target["quaternion_wxyz"])
            grip = gripper_closed_value if target["gripper_closed"] else gripper_open_value
        solved, err = solve_ik(arm, joints, t_pos, t_rot)
        goal[ARM_SLICES[arm]] = solved
        goal[GRIPPER_INDEX[arm]] = grip
        info[arm] = {"ik_error": round(err, 6), "target_position": [round(float(v), 5) for v in t_pos]}
    fractions = np.arange(1, steps + 1, dtype=np.float64) / steps
    actions = s[None, :] + fractions[:, None] * (goal - s)[None, :]
    # grippers switch at once (a command, not a trajectory), as an operator's open/close would
    for arm in ARMS:
        actions[:, GRIPPER_INDEX[arm]] = goal[GRIPPER_INDEX[arm]]
    return actions.astype(np.float32), info


def roboDojo_like(action_dim: Optional[int], control_mode: str) -> bool:
    return action_dim == 14 and control_mode == "joint_abs"
