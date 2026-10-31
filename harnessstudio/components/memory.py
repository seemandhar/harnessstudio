"""Experience stores (memory_retrieval, spec 5.2): read-only views over prepared artifacts.

TrajectoryStore holds development trajectories as JSONL, one line per episode:
    {"task": str, "success": bool, "steps": [{"state": [...], "action": [...]}, ...]}
`write_trajectories(records, path)` builds that file from development records (Preparer helper).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def write_trajectories(records: Iterable[dict], path: Path, *, only_success: bool = True,
                       state_key: str = "state") -> int:
    n = 0
    with open(path, "w") as f:
        for r in records:
            if only_success and not r.get("success"):
                continue
            steps = [{"state": s.get(state_key), "action": s["action"]} for s in r.get("experience", [])]
            if not steps:
                continue
            f.write(json.dumps({"task": r["task"], "success": bool(r.get("success")), "steps": steps}) + "\n")
            n += 1
    return n


class TrajectoryStore:
    def __init__(self, trajectories: list[dict]) -> None:
        self.trajectories = trajectories
        rows = [(i, t, np.asarray(s["state"], dtype=np.float64).reshape(-1), np.asarray(s["action"], dtype=np.float64))
                for i, tr in enumerate(trajectories) for t, s in enumerate(tr["steps"]) if s.get("state") is not None]
        self._index = [(i, t) for i, t, _, _ in rows]
        self._states = np.stack([s for _, _, s, _ in rows]) if rows else np.zeros((0, 0))
        self._actions = [a for _, _, _, a in rows]

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryStore":
        with open(path) as f:
            return cls([json.loads(line) for line in f if line.strip()])

    def __len__(self) -> int:
        return len(self.trajectories)

    def nearest(self, state, k: int = 1, task: Optional[str] = None) -> list[dict]:
        """k nearest stored steps by Euclidean distance on the state vector (optionally same task)."""
        if not len(self._states):
            return []
        q = np.asarray(state, dtype=np.float64).reshape(-1)
        d = np.linalg.norm(self._states[:, : len(q)] - q[None, : self._states.shape[1]], axis=1)
        order = np.argsort(d, kind="stable")
        out = []
        for j in order:
            i, t = self._index[j]
            if task is not None and self.trajectories[i]["task"] != task:
                continue
            out.append({"trajectory": i, "step": t, "distance": float(d[j]), "action": self._actions[j],
                        "progress": t / max(1, len(self.trajectories[i]["steps"]) - 1)})
            if len(out) == k:
                break
        return out
