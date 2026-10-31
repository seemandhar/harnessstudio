"""Dual-Memory bank, ported from src/soma_memory.py (RoboHarness repo, MIT, 716defc).

Two partitions (success / failure), each a matrix of L2-normalized 1168-d embeddings plus metadata records
{id, task, diagnosis, info, ...}. Retrieval is the original: cosine (dot product of normalized vectors), top-k,
entries below `threshold` (0.4) dropped. In HarnessStudio the bank is a frozen development artifact
(`memory.json`, written by prepare.Preparer); evaluation only reads it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

DIMENSION = 1168


def empty_bank() -> dict:
    return {"schema": "roboharness-memory-v1", "dimension": DIMENSION,
            "success": {"vecs": [], "meta": []}, "failure": {"vecs": [], "meta": []}}


class MemoryBank:
    def __init__(self, data: Optional[dict] = None) -> None:
        data = data or empty_bank()
        self.dimension = int(data.get("dimension", DIMENSION))
        self.index = {}
        for part in ("success", "failure"):
            p = data.get(part) or {}
            vecs = np.asarray(p.get("vecs") or np.empty((0, self.dimension)), dtype=np.float32)
            if vecs.ndim != 2 or (len(vecs) and vecs.shape[1] != self.dimension):
                raise ValueError(f"memory partition {part} has vectors of shape {vecs.shape}")
            self.index[part] = {"vecs": vecs.reshape(-1, self.dimension), "meta": list(p.get("meta") or [])}
            if len(self.index[part]["vecs"]) != len(self.index[part]["meta"]):
                raise ValueError(f"memory partition {part}: vector and metadata counts differ")

    @classmethod
    def load(cls, path: Union[str, Path, None]) -> "MemoryBank":
        if not path:
            return cls()
        return cls(json.loads(Path(path).read_text()))

    def add_experience(self, embedding, task_desc: str, success: bool, diagnosis: str = "",
                       info: Optional[dict] = None, record_id: str = "", source: Optional[dict] = None) -> None:
        partition = "success" if success else "failure"
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.shape[0] != self.dimension:
            raise ValueError(f"vector dimension {emb.shape[0]} != {self.dimension}")
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        record = {"id": record_id or f"{partition}_{len(self.index[partition]['meta'])}", "task": task_desc,
                  "diagnosis": diagnosis, "info": info if isinstance(info, dict) else {}, "source": source or {}}
        self.index[partition]["vecs"] = np.vstack([self.index[partition]["vecs"], emb.reshape(1, -1)])
        self.index[partition]["meta"].append(record)

    def retrieve(self, query_vector, top_k: int = 3, partition: str = "failure", threshold: float = 0.4) -> list:
        q = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm
        parts = [partition] if partition != "all" else ["success", "failure"]
        results = []
        for part in parts:
            vecs, meta = self.index[part]["vecs"], self.index[part]["meta"]
            if len(vecs) == 0:
                continue
            scores = np.dot(vecs, q)
            k_idx = min(top_k, len(scores))
            for idx in np.argsort(scores)[-k_idx:][::-1]:
                score = float(scores[idx])
                if score < threshold:
                    continue
                item = dict(meta[idx])
                item["score"] = score
                item["type"] = part
                results.append(item)
        if partition == "all":
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
        return results

    def stats(self) -> dict:
        return {"success_count": len(self.index["success"]["meta"]),
                "failure_count": len(self.index["failure"]["meta"]), "vector_dim": self.dimension}

    def to_json(self) -> dict:
        out = empty_bank()
        out["dimension"] = self.dimension
        for part in ("success", "failure"):
            out[part] = {"vecs": [[round(float(x), 7) for x in v] for v in self.index[part]["vecs"]],
                         "meta": self.index[part]["meta"]}
        return out
