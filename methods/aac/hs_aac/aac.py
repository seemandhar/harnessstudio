"""AAC: Adaptive Action Chunking at inference time, action-entropy horizon selection (numpy only).

Source: Liang et al., "Adaptive Action Chunking at Inference-time for Vision-Language-Action Models",
arXiv:2604.04161v2 (CVPR 2026). Author code: Adaptive-Action-Chunking/robocasa@fed3e6b5eb348160dd0570f326f726758fee9056
robocasa/demos/action_optimization/action_entropy_v2.py (sha256 80abe7e1...) select_chunk_size(method=
"gaussian_bernoulli", move_th=3.0) as called by inference_client_action_entropy.py with --chunk_id_selector 0
(AAC_evaluation.sh); quaternion helpers from robocasa/demos/help_function.py (sha256 056cf383...); server
Adaptive-Action-Chunking/gr00t-multi-sample@11e926b0f34cf6acfcb92c0fe6127a1bdc7b856a (n_samples=20).

Rule (per decision, N=20 samples of one observation, chunk T=16):
  entropy:  for each step t, H_t = H_gauss(normalized eef position of the N samples at t)
                                 + H_gauss(normalized eef rotation at t) + H_bernoulli(gripper_close at t)
            (weights 1/1/1; H_gauss = 0.5*(d*log(2*pi*e) + logdet(cov + 1e-6 I)), np.cov ddof=1);
            chunk_mean[k] = mean(H_1..H_k); k_entropy = max(argmax(diff(chunk_mean)) + 1, 2)
  motion:   on sample 0 (unnormalized): m_k = ||sum_{t<k} eef_pos_t|| + ||compose(eef_rot_1..k)|| + 0.2 * toggle_k
            for k = 2..16; k_motion = first k with m_k > move_th (3.0), else 16
  execute:  sample 0 (no selector), k = max(k_motion, k_entropy), in [2, 16]

Ported line-for-line in numpy from HarnessStudio v1 (branch ext/aac, harnessstudio/methods/aac.py), which
was checked bitwise against the pinned upstream functions. The paper-level differences D1-D7 are listed in
paper.yaml (`disclosures`) and README.md.

v2 port notes (not paper deviations):
  P1 horizon: the author code hardcodes the motion range 2..16 and the fallback 16 (GR00T chunks, T=16).
     Here both use the chunk length T of the decision (range 2..T, fallback T), which is the author code at
     T=16. v1 refused T != 16; `strict_horizon=True` restores that refusal. T < 2 has no elbow and no motion
     range, so the whole chunk (k = T) is executed.
  P2 action groups: resolved by NAME from Embodiment.action_layout (bind) or an explicit `dims` override,
     instead of v1's extras['policy']['action_layout'] with GR00T keys; GR00T's v1 key names are accepted as
     aliases of the vla-eval names (position, rotation, gripper).
This module imports only numpy so the numerics can be compared with the upstream code in any environment.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

AUTHOR_HORIZON = 16
NUM_SAMPLES = 20

# Role -> accepted action-layout names, in order of preference: vla-eval DimSpec names first, then the
# GR00T panda_omron key names v1 read from the policy adapter.
GR00T_V1_KEYS = {"position": "action.end_effector_position", "rotation": "action.end_effector_rotation",
                 "gripper": "action.gripper_close"}
KEYS = {role: [role, GR00T_V1_KEYS[role]] for role in ("position", "rotation", "gripper")}
WIDTHS = {"position": 3, "rotation": 3, "gripper": 1}

# GR00T N1.5 RoboCasa365 multitask checkpoint-120000 experiment_cfg/metadata.json (sha256 below),
# new_embodiment.statistics.action.<key>.{min,max}; panda_omron min_max normalization for both keys.
METADATA_SHA256 = "8be0fc606c9220356bad497feefc4ae4daaa05670acccc31caecaa7ae5590b69"
NORMALIZATION = {"position": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                 "rotation": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}}
NORMALIZATION_SOURCE = {"file": "experiment_cfg/metadata.json", "sha256": METADATA_SHA256,
                        "keys": {"position": "end_effector_position", "rotation": "end_effector_rotation"}}

PARAMS = {"min_k": 2, "per_step": False, "sees_proposals": True, "move_th": 3.0,
          "entropy_weights": {"position": 1.0, "rotation": 1.0, "gripper": 1.0},
          "magnitude_weights": {"position": 1, "rotation": 1, "gripper_toggle": 0.2},
          "cov_eps": 1e-6, "keys": KEYS, "dims": None, "normalization": NORMALIZATION,
          "normalization_source": NORMALIZATION_SOURCE, "strict_horizon": False}


# ---------------- numeric helpers (same numpy operations as the author code) ----------------

def _safe_logdet_cov(X, eps=1e-6):
    X = np.asarray(X)
    N, d = X.shape
    if N <= 1:
        return -np.inf
    Sigma = np.cov(X, rowvar=False)
    Sigma = Sigma + eps * np.eye(d)
    try:
        L = np.linalg.cholesky(Sigma)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return -np.inf
    return logdet


def gaussian_entropy(X, eps=1e-6):
    X = np.asarray(X)
    N, d = X.shape
    if N <= 1:
        return 0.0
    logdet = _safe_logdet_cov(X, eps)
    return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)


def bernoulli_entropy(g):
    """= scipy.stats.entropy([p0, p1], base=e) with p1 clipped to [1e-9, 1 - 1e-9] (author code)."""
    g = np.asarray(g).ravel()
    p1 = float(np.clip(np.mean(g), 1e-9, 1 - 1e-9))
    p0 = 1.0 - p1
    pk = np.asarray([p0, p1], dtype=float)
    pk = pk / np.sum(pk)
    vec = np.where(pk > 0, -pk * np.log(np.where(pk > 0, pk, 1.0)), 0.0)  # scipy.special.entr
    return float(np.sum(vec))


def axisangle2quat(vec):
    angle = np.linalg.norm(vec)
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / angle
    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def quat_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0), dtype=np.float32)


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def compose_rotations(delta_rs):
    q_total = np.array([0.0, 0.0, 0.0, 1.0])
    for r in delta_rs:
        q = axisangle2quat(r)
        q_total = quat_multiply(q, q_total)
    return quat2axisangle(q_total)


def entropy_breakdown(pos_n, rot_n, grip, weights, eps):
    """pos_n, rot_n: (N, T, 3) normalized; grip: (N, T, 1) unnormalized. Author: action_entropy_gausssian_bernoulli."""
    N, T, D = pos_n.shape
    totals = []
    for t in range(T):
        h = (weights["position"] * gaussian_entropy(pos_n[:, t, :], eps)
             + weights["rotation"] * gaussian_entropy(rot_n[:, t, :], eps)
             + weights["gripper"] * bernoulli_entropy(grip[:, t]))
        totals.append(h)
    totals = np.array(totals)
    chunk_mean = [float(np.mean(totals[:k])) for k in range(1, T + 1)]
    return totals, chunk_mean


def action_magnitude(pos0, rot0, grip0, weights, horizon=AUTHOR_HORIZON):
    """Sample 0, unnormalized. pos0/rot0 (T, 3), grip0 (T, 1). Author: action_magnitude(end_to_end=True),
    k = 2..16 (P1: 2..horizon)."""
    pos0, rot0 = np.asarray(pos0), np.asarray(rot0)
    grip0 = np.asarray(grip0).astype(int)
    out = []
    for k in range(2, horizon + 1):
        dpos = np.linalg.norm(np.sum(pos0[:k, ], axis=0))
        drot = np.linalg.norm(compose_rotations(rot0[:k, ]))
        toggle = int(np.any(np.diff(grip0[:k, ])))   # D5: diff over the last axis of a (k, 1) array
        out.append(weights["position"] * dpos + weights["rotation"] * drot + weights["gripper_toggle"] * toggle)
    return out


def find_first_idx(chunk_m, move_th, horizon=AUTHOR_HORIZON):
    for i, m in enumerate(chunk_m):
        if m > move_th:
            return i + 2
    return horizon


def select_chunk_size(pos_n, rot_n, grip, pos0, rot0, grip0, *, move_th, entropy_weights, magnitude_weights, eps):
    """Author select_chunk_size(method='gaussian_bernoulli'); horizon = T of the samples (16 for GR00T)."""
    totals, vals = entropy_breakdown(pos_n, rot_n, grip, entropy_weights, eps)
    horizon = int(np.asarray(pos_n).shape[1])
    if horizon < 2:  # P1: no elbow (diff of one value is empty) and no motion range: run the whole chunk
        return horizon, {"k_entropy": horizon, "k_motion": horizon, "H_total": totals, "chunk_mean": vals,
                         "motion": []}
    diffs = np.diff(vals)
    best_k = int(np.argmax(diffs)) + 1
    k_entropy = max(best_k, 2)
    chunk_m = action_magnitude(pos0, rot0, grip0, magnitude_weights, horizon)
    k_motion = find_first_idx(chunk_m, move_th, horizon)
    return max(k_motion, k_entropy), {"k_entropy": k_entropy, "k_motion": k_motion, "H_total": totals,
                                      "chunk_mean": vals, "motion": chunk_m}


# ---------------- HarnessStudio components ----------------

def _finite(values):
    return [float(v) if math.isfinite(float(v)) else None for v in values]


class AACPool:
    """Proposer slot: N seeded samples per decision (sample 0 = the native stream). The DecisionLoop draws
    them with ctx.policy.sample(N); author server gr00t-multi-sample n_samples=20."""

    def __init__(self, num_samples: int = NUM_SAMPLES) -> None:
        if type(num_samples) is not int or num_samples < 2:
            raise ValueError("aac: num_samples must be an integer >= 2 (sample entropy needs N >= 2)")
        self.num_samples = num_samples


class AACEntropyExecutor:
    """Executor slot. Reads extras['proposals'] (all N samples of the decision, index 0 = native stream),
    groups the action columns BY NAME from the bound action layout (or `dims`), and plans
    k = select_chunk_size(...). No selector: the executed chunk is sample 0."""

    sees_proposals = True

    def __init__(self, min_k: int = 2, per_step: bool = False, sees_proposals: bool = True, move_th: float = 3.0,
                 entropy_weights: Optional[Mapping[str, float]] = None,
                 magnitude_weights: Optional[Mapping[str, float]] = None, cov_eps: float = 1e-6,
                 keys: Optional[Mapping[str, Any]] = None, dims: Optional[Mapping[str, Any]] = None,
                 normalization: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 normalization_source: Optional[Mapping[str, Any]] = None, strict_horizon: bool = False) -> None:
        if min_k != 2 or per_step is not False or sees_proposals is not True:
            raise ValueError("aac: min_k=2, per_step=False and sees_proposals=True (author chunk sizes 2..16)")
        self.min_k, self.per_step = min_k, per_step
        self.move_th, self.eps = float(move_th), float(cov_eps)
        self.entropy_weights = dict(entropy_weights or PARAMS["entropy_weights"])
        self.magnitude_weights = dict(magnitude_weights or PARAMS["magnitude_weights"])
        keys = keys or KEYS
        self.keys = {role: [keys[role]] if isinstance(keys[role], str) else list(keys[role]) for role in WIDTHS}
        self.normalization = {k: dict(v) for k, v in (normalization if normalization is not None
                                                       else NORMALIZATION).items()}
        self.normalization_source = dict(normalization_source or {})
        self.strict_horizon = bool(strict_horizon)
        self._layout: Optional[dict] = None
        self._columns: Optional[dict] = None
        if dims is not None:
            self._columns = {role: self._slice(role, dims.get(role), "dims override") for role in WIDTHS}
        self._last: Optional[dict] = None

    # -- action layout --------------------------------------------------------------
    def bind(self, embodiment) -> None:
        """Public action-space facts (vla-eval DimSpec names -> (start, end), in vector order)."""
        self._layout = {str(k): tuple(v) for k, v in dict(embodiment.action_layout).items()}

    def _slice(self, role, span, where):
        if span is None or len(span) != 2:
            raise ValueError(f"aac: {where} lacks a (start, end) for {role!r}")
        start, end = int(span[0]), int(span[1])
        if end - start != WIDTHS[role]:
            raise ValueError(f"aac: {role} has width {end - start} in the {where}, expected {WIDTHS[role]}")
        return slice(start, end)

    def columns(self) -> dict:
        if self._columns is None:
            if self._layout is None:
                raise ValueError("aac: no action layout (bind() was not called) and no `dims` override")
            cols = {}
            for role, names in self.keys.items():
                name = next((n for n in names if n in self._layout), None)
                if name is None:
                    raise ValueError(f"aac: action layout lacks {names} (has {sorted(self._layout)})")
                cols[role] = self._slice(role, self._layout[name], f"action layout entry {name!r}")
            self._columns = cols
        return self._columns

    def _normalize(self, values, role):
        """GR00T Normalizer min_max forward: 2 * (x - min) / (max - min) - 1, identity where min == max."""
        out = np.array(values, copy=True)              # keeps the native dtype (GR00T flats: float32)
        stats = self.normalization.get(role)
        if not stats:
            return out
        lo, hi = np.asarray(stats["min"], dtype=out.dtype), np.asarray(stats["max"], dtype=out.dtype)
        mask = lo != hi
        out[..., mask] = (out[..., mask] - lo[mask]) / (hi[mask] - lo[mask])
        out[..., mask] = 2 * out[..., mask] - 1
        return out

    # -- executor interface ------------------------------------------------------------
    def plan(self, chunk, extras, obs) -> int:
        proposals = extras.get("proposals")
        if proposals is None or len(proposals) < 2:
            raise ValueError("aac needs extras['proposals'] with N >= 2 samples (executor sees_proposals)")
        cols = self.columns()
        samples = np.stack([np.asarray(getattr(p, "actions", p)) for p in proposals])  # (N, T, A), native dtype
        if samples.ndim != 3:
            raise ValueError(f"aac: proposals must be [H, A] chunks, got shape {samples.shape[1:]}")
        if self.strict_horizon and samples.shape[1] != AUTHOR_HORIZON:
            raise ValueError(f"aac: author chunk sizes 2..16 assume T=16, got T={samples.shape[1]}")
        pos, rot, grip = samples[:, :, cols["position"]], samples[:, :, cols["rotation"]], samples[:, :, cols["gripper"]]
        k, info = select_chunk_size(self._normalize(pos, "position"), self._normalize(rot, "rotation"), grip,
                                    pos[0], rot[0], grip[0], move_th=self.move_th,
                                    entropy_weights=self.entropy_weights, magnitude_weights=self.magnitude_weights,
                                    eps=self.eps)
        native = np.asarray(getattr(chunk, "actions", chunk))
        self._last = {"k": int(k), "k_entropy": int(info["k_entropy"]), "k_motion": int(info["k_motion"]),
                      "n": int(samples.shape[0]), "T": int(samples.shape[1]),
                      # proposals[0] is the executed native chunk (no selector, no action_op), same dtype/shape
                      "proposal0_equals_chunk": bool(samples[0].shape == native.shape and np.array_equal(samples[0], native)),
                      "proposal_dtype": str(samples.dtype), "proposal_shape": list(samples.shape[1:]),
                      "H_total": _finite(info["H_total"]), "chunk_mean": _finite(info["chunk_mean"]),
                      "motion": _finite(info["motion"])}
        return int(k)

    def diagnostics(self) -> dict:
        """Last decision's evidence (finite JSON). v1 traced this as `executor_info`; the v2 core does not."""
        return dict(self._last or {})

    def continue_after(self, executed, obs) -> bool:
        return True

    def record_executed(self, executed_actions) -> None:
        return None
