"""Fixed, pure rules of H1-BCP (Boundary Clause Prompt), frozen before any GPU run.

Source of every rule: runs/rtcfpp-20261101/ideation/r8-task.json (scientist afba071c, hypothesis H1-BCP) and its
offline script ideation/scripts-r8-task/boundary_exposure.py (sha256 af2507b6...), with the conditions of
runs/rtcfpp-20261101/reviews/r8-review.json (critic a182f0a2, hypothesis_rulings H1-BCP, K1-c1..K1-c6).

1. ``split_clause``: copied verbatim (logic and regular expressions) from boundary_exposure.py ``split_clause``.
   Split at ' and VERB' (VERB in VERBS); if clause 2 contains 'it', replace it with the last 'in/on/into the ...'
   object of clause 1; refuse the object pronoun case (t5 'pick up ... and place it ...'). None = no split =
   the episode runs fully native.
2. ``control_prompt``: the critic's non-semantic surface control (K1-c3): the ORIGINAL instruction + ' .'.
3. ``CustodyTracker``: the public custody boundary of boundary_exposure.py ``boundary()`` over
   analysis/r8/features.py segments, evaluated online: a maximal run of executed gripper-close commands
   (action[6] > 0) of length >= HOLD_STEPS whose finger width (state[6] - state[7]) median over the post-step
   states of rows [min(a + 10, b - 1), b) is >= WIDTH_MIN, ended at row b by an executed open command
   (action[6] <= 0); as in the scientist's code, the median is rounded to 4 decimals before the comparison
   (features.py stores round(median, 4)). It fires once, at the first such segment; the boundary step is b (the open command's row).
4. ``chunk_compare``: the K1 per-decision measures (a)-(c) of the proposal: chunk (token) identity, cosine and
   L2 of the summed translation (rows 0-4, dims 0-2), gripper-row sign disagreement on rows 0-4.

No RNG, no simulator state, no case/seed/arm. Public inputs only.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional, Sequence

import numpy as np

VERBS = ("put", "close", "open", "turn", "pick", "place", "push")
CONTROL_SUFFIX = " ."
HOLD_STEPS = 15
WIDTH_MIN = 0.003
SETTLE_ROWS = 10
GRIPPER_DIM = 6
WIDTH_DIMS = (6, 7)
COMPARE_ROWS = 5           # rows 0-4 (the executed prefix at cadence 5)
TRANSLATION_DIMS = (0, 3)  # dims 0-2
COSINE_DISAGREE = 0.5      # a decision disagrees if cosine < 0.5 or a gripper flip
NORM_EPS = 1e-8


def split_clause(s: str) -> Optional[str]:
    """Verbatim logic of boundary_exposure.py split_clause. Returns clause 2 (pronoun resolved) or None."""
    m = re.search(r" and (%s)\b" % "|".join(VERBS), s)
    if not m:
        return None
    c1, c2 = s[:m.start()], s[m.start() + 5:]
    if re.search(r"\bit\b", c2):
        if re.search(r"\b(pick up|place)\b", c1 + " " + c2) and c2.startswith("place"):
            return None  # object pronoun (t5)
        objs = (re.findall(r"\b(?:in|on|into)\s+(the [a-z ]+?)$", c1)
                or re.findall(r"\b(?:in|on|into)\s+(the [a-z ]+)", c1))
        if not objs:
            return None
        c2 = re.sub(r"\bit\b", objs[-1].strip(), c2)
    return c2


def control_prompt(instruction: str) -> str:
    """K1-c3 surface-perturbation control: the original instruction with ' .' appended."""
    return instruction + CONTROL_SUFFIX


class CustodyTracker:
    """Online public custody boundary. Feed one executed step at a time: the executed action row and the
    observation state AFTER that step (vla-eval rows hold the post-step state, as analysis/r8/features.py)."""

    def __init__(self, hold_steps: int = HOLD_STEPS, width_min: float = WIDTH_MIN, settle_rows: int = SETTLE_ROWS,
                 gripper_dim: int = GRIPPER_DIM, width_dims: Sequence[int] = WIDTH_DIMS) -> None:
        self.hold_steps = int(hold_steps)
        self.width_min = float(width_min)
        self.settle_rows = int(settle_rows)
        self.gripper_dim = int(gripper_dim)
        self.width_dims = (int(width_dims[0]), int(width_dims[1]))
        self.step = 0                    # rows seen so far (= index of the next row)
        self.seg_start: Optional[int] = None
        self.widths: list = []           # post-step widths of the current closed segment (rows seg_start..)
        self.boundary: Optional[int] = None
        self.segments: list = []         # closed segments seen before the boundary: (start, end, len, w_settled)
        self.missing_state = 0

    def width(self, state: Any) -> Optional[float]:
        if state is None:
            return None
        try:
            s = np.asarray(state, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if s.size <= max(self.width_dims) or not np.all(np.isfinite(s[list(self.width_dims)])):
            return None
        return float(s[self.width_dims[0]] - s[self.width_dims[1]])

    def update(self, action: Any, post_state: Any) -> bool:
        """Consume one executed row. Returns True exactly once: on the row whose open command ends the first
        qualifying closed segment (that row's index is self.boundary)."""
        t = self.step
        self.step += 1
        if self.boundary is not None:
            return False
        g = float(np.asarray(action, dtype=np.float64).reshape(-1)[self.gripper_dim])
        fired = False
        if g > 0:
            if self.seg_start is None:
                self.seg_start, self.widths = t, []
            w = self.width(post_state)
            if w is None:
                self.missing_state += 1
            self.widths.append(w)
        elif self.seg_start is not None:
            a, b = self.seg_start, t
            n = b - a
            window = self.widths[min(a + self.settle_rows, b - 1) - a: b - a]
            vals = [w for w in window if w is not None]
            w_settled = float(np.median(vals)) if vals and len(vals) == len(window) else None
            self.segments.append({"start": a, "end": b, "len": n,
                                  "w_settled": None if w_settled is None else round(w_settled, 6)})
            # boundary_exposure.py compares features.py's w_settled, which is rounded to 4 decimals
            if n >= self.hold_steps and w_settled is not None and round(w_settled, 4) >= self.width_min:
                self.boundary = b
                fired = True
            self.seg_start, self.widths = None, []
        return fired


def chunk_bytes_identical(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def chunk_compare(native: Any, other: Any, rows: int = COMPARE_ROWS) -> dict:
    """K1 per-decision measures of ``other`` (clause or control decode) against the native decode."""
    n = np.asarray(native, dtype=np.float64)
    o = np.asarray(other, dtype=np.float64)
    r = min(rows, len(n), len(o))
    lo, hi = TRANSLATION_DIMS
    tn, to = n[:r, lo:hi].sum(axis=0), o[:r, lo:hi].sum(axis=0)
    nn, no = float(np.linalg.norm(tn)), float(np.linalg.norm(to))
    if nn < NORM_EPS and no < NORM_EPS:
        cos = 1.0
    elif nn < NORM_EPS or no < NORM_EPS:
        cos = 0.0
    else:
        cos = float(np.dot(tn, to) / (nn * no))
    flips = [int(i) for i in range(r) if (n[i, GRIPPER_DIM] > 0) != (o[i, GRIPPER_DIM] > 0)]
    token_identical = chunk_bytes_identical(np.asarray(native), np.asarray(other))
    return {"token_identical": bool(token_identical), "cosine": cos, "l2": float(np.linalg.norm(tn - to)),
            "gripper_flip_rows": flips, "gripper_flip": bool(flips),
            "disagree": bool(cos < COSINE_DISAGREE or flips), "rows": r}


def array_fingerprint(a: np.ndarray) -> str:
    """harnessstudio.context.trace.fingerprint of one numpy array (dtype, shape, sha256 of bytes)."""
    a = np.asarray(a)
    canon = {"dtype": str(a.dtype), "shape": list(a.shape),
             "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, allow_nan=False).encode()).hexdigest()
