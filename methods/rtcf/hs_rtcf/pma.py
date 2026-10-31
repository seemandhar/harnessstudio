"""Retrieve in Time: Progressive Memory Alignment (RTCF v2 §3.1, Eq. 5-9), incremental and vectorized.

The bank is stored flat: descriptors Z [sum L_i, D] (unit vectors) with memory boundaries offsets [N + 1]
(memory i occupies rows offsets[i]:offsets[i+1], L_i >= 1).

    c_{i,t}(q) = 1 - clip(z_t . z_{i,q}, -1, 1)                                   (Eq. 5)
    J_{i,1}(q) = c_{i,1}(q)                                                       (Eq. 7, open begin)
    J_{i,t}(q) = c_{i,t}(q) + min_{0<=delta<=v_max, q-delta>=1} J_{i,t-1}(q-delta) + gamma |delta - 1|   (Eq. 6)
    b*_{i,t} = argmin_q J_{i,t}(q);  d_{i,t} = J_{i,t}(b*_{i,t}) / t              (Eq. 8)
    i*_t = argmin_i d_{i,t};  b*_t = b*_{i*_t, t}                                 (Eq. 9)

Positions q are 0-based here (q = 0 is the paper's q = 1). Ties go to the lowest index (numpy argmin).

Pure functions (the parity script imports them):
    local_cost(z, Z) -> [sum L]
    pma_init(cost) -> J
    pma_step(J_prev, cost, local_index, v_max, gamma) -> J
    pma_select(J, offsets, t) -> (i_star, b_star (0-based within memory i_star), d [N])
    frame_nn_select(cost, offsets) -> (i_star, b_star, d)          (Frame-NN ablation, inferred rule)
and a stateful wrapper `ProgressiveMemoryAlignment` (one per episode).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def local_cost(z: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
    """Eq. 5: cosine dissimilarity between the query descriptor and every stored descriptor."""
    sim = np.asarray(descriptors, dtype=np.float64) @ np.asarray(z, dtype=np.float64).reshape(-1)
    return 1.0 - np.clip(sim, -1.0, 1.0)


def local_index(offsets: np.ndarray) -> np.ndarray:
    """Position of every flat row within its own memory (0..L_i-1)."""
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    if np.any(lengths < 1):
        raise ValueError("every memory needs at least one position")
    return np.arange(offsets[-1], dtype=np.int64) - np.repeat(offsets[:-1], lengths)


def pma_init(cost: np.ndarray) -> np.ndarray:
    """Eq. 7."""
    return np.array(cost, dtype=np.float64, copy=True)


def pma_step(prev: np.ndarray, cost: np.ndarray, loc: np.ndarray, v_max: int, gamma: float) -> np.ndarray:
    """Eq. 6 for every memory at once. A predecessor q - delta outside the memory (loc < delta) is +inf."""
    prev = np.asarray(prev, dtype=np.float64)
    n = prev.shape[0]
    best = np.full(n, np.inf)
    for delta in range(0, int(v_max) + 1):
        cand = np.full(n, np.inf)
        if delta == 0:
            cand[:] = prev
        elif delta < n:
            cand[delta:] = prev[:-delta]
        cand[loc < delta] = np.inf
        cand = cand + float(gamma) * abs(delta - 1)
        best = np.minimum(best, cand)
    return np.asarray(cost, dtype=np.float64) + best


def pma_select(frontier: np.ndarray, offsets: np.ndarray, t: int) -> tuple[int, int, np.ndarray]:
    """Eq. 8-9: per-memory best endpoint and score J/t; the best memory (lowest index on ties)."""
    offsets = np.asarray(offsets, dtype=np.int64)
    mins = np.minimum.reduceat(frontier, offsets[:-1])
    d = mins / float(t)
    i_star = int(np.argmin(d))
    seg = frontier[offsets[i_star]:offsets[i_star + 1]]
    return i_star, int(np.argmin(seg)), d


def frame_nn_select(cost: np.ndarray, offsets: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Frame-NN ablation ('w/o history alignment'; exact rule not given): the single most similar stored frame."""
    offsets = np.asarray(offsets, dtype=np.int64)
    flat = int(np.argmin(cost))
    i_star = int(np.searchsorted(offsets, flat, side="right") - 1)
    d = np.minimum.reduceat(cost, offsets[:-1])
    return i_star, flat - int(offsets[i_star]), d


class ProgressiveMemoryAlignment:
    """Per-episode PMA state: only the current frontier is kept (Alg. 1; reset by building a new instance)."""

    def __init__(self, descriptors: np.ndarray, offsets: np.ndarray, v_max: int = 2, gamma: float = 0.05,
                 retrieval: str = "pma") -> None:
        if retrieval not in ("pma", "frame_nn"):
            raise ValueError(f"unknown retrieval {retrieval!r}")
        if int(v_max) < 0 or float(gamma) < 0:
            raise ValueError("v_max and gamma must be non-negative")
        self.descriptors = np.asarray(descriptors)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.loc = local_index(self.offsets)
        self.v_max, self.gamma, self.retrieval = int(v_max), float(gamma), retrieval
        self.frontier: Optional[np.ndarray] = None
        self.t = 0

    @property
    def num_memories(self) -> int:
        return len(self.offsets) - 1

    def update(self, z: np.ndarray) -> tuple[int, int, float]:
        """Add one query descriptor; return (i*, b* 0-based within memory i*, d*)."""
        cost = local_cost(z, self.descriptors)
        self.t += 1
        if self.retrieval == "frame_nn":
            i, b, d = frame_nn_select(cost, self.offsets)
            return i, b, float(d[i])
        self.frontier = pma_init(cost) if self.frontier is None else \
            pma_step(self.frontier, cost, self.loc, self.v_max, self.gamma)
        i, b, d = pma_select(self.frontier, self.offsets, self.t)
        return i, b, float(d[i])
