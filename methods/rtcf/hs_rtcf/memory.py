"""The frozen success-memory bank B+ (RTCF v2 §3): one row per policy invocation of every successful
development episode, all tasks of the suite together (retrieval uses no task id).

memory.npz:
    descriptors   float32 [sum L, D]   unit SigLIP descriptors z_{i,q}
    offsets       int64   [N + 1]      memory i = rows offsets[i]:offsets[i+1]
    coef_q        int64   [sum L, H, A] round(token_scale * C) of the invocation's full proposed chunk
                                       (f > 0 rows equal the FAST tokens' integer coefficients; see spectral.py)
    decode_failed bool    [sum L]      that proposal was openpi's zero chunk (failed FAST decode)
memory.json: provenance (encoder, norm-stats sha, token_scale, chunk shape, per-memory case keys and tasks).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

_CACHE: dict = {}


class MemoryBank:
    def __init__(self, descriptors: np.ndarray, offsets: np.ndarray, coef_q: np.ndarray, decode_failed: np.ndarray,
                 meta: Optional[dict] = None) -> None:
        self.descriptors = np.asarray(descriptors, dtype=np.float32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.coef_q = np.asarray(coef_q, dtype=np.int64)
        self.decode_failed = np.asarray(decode_failed, dtype=bool)
        self.meta = dict(meta or {})
        n = int(self.offsets[-1]) if len(self.offsets) else 0
        if len(self.offsets) < 1 or self.offsets[0] != 0 or np.any(np.diff(self.offsets) < 1):
            raise ValueError("bad memory offsets")
        if not (len(self.descriptors) == len(self.coef_q) == len(self.decode_failed) == n):
            raise ValueError("memory arrays disagree in length")

    @property
    def num_memories(self) -> int:
        return len(self.offsets) - 1

    @property
    def num_positions(self) -> int:
        return int(self.offsets[-1])

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def row(self, i: int, q: int) -> int:
        return int(self.offsets[i]) + int(q)

    @classmethod
    def empty(cls, dim: int, h: int, a: int) -> "MemoryBank":
        return cls(np.zeros((0, dim), np.float32), np.zeros(1, np.int64), np.zeros((0, h, a), np.int64),
                   np.zeros(0, bool))

    def save(self, npz_path: Path, json_path: Path) -> None:
        with open(npz_path, "wb") as f:
            np.savez(f, descriptors=self.descriptors, offsets=self.offsets, coef_q=self.coef_q,
                     decode_failed=self.decode_failed)
        Path(json_path).write_text(json.dumps(self.meta, indent=1, sort_keys=True))

    @classmethod
    def load(cls, npz_path: str | Path, json_path: Optional[str | Path] = None) -> "MemoryBank":
        """Read-only load, cached per file content (the bank is frozen and sha-pinned by prepared.json)."""
        raw = Path(npz_path).read_bytes()
        meta_raw = Path(json_path).read_bytes() if json_path else b""
        # the cache key covers the arrays AND the metadata (a bank object carries its meta; R10 reads it)
        key = hashlib.sha256(raw).hexdigest() + ":" + hashlib.sha256(meta_raw).hexdigest()
        if key not in _CACHE:
            import io
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                arrays = {k: np.array(z[k]) for k in ("descriptors", "offsets", "coef_q", "decode_failed")}
            for arr in arrays.values():
                arr.setflags(write=False)
            meta = json.loads(meta_raw.decode()) if json_path else {}
            _CACHE[key] = cls(meta=meta, **arrays)
        return _CACHE[key]
