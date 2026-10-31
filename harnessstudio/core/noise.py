"""Common random numbers: one noise seed per (case, decision, sample), shared by all arms."""
from __future__ import annotations

import hashlib
import json


def crn_seed(namespace: str, benchmark: str, task: str, episode_idx: int, decision: int, sample: int) -> int:
    key = json.dumps([namespace, benchmark, task, int(episode_idx), int(decision), int(sample)],
                     separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
