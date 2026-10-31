"""FastHarness experience store (fast_harness/memory.py @ dfe070f), in memory instead of SQLite.

Same tables and update rules: stage examples (<= examples_per_stage per stage, most recent kept), decayed
success/failure statistics with a clean streak, error cases merged within `error_merge_radius`,
de-duplicated outcome events, and per-stage chunk-span EWMAs. The only change is storage: HarnessStudio
freezes experience as a hashed JSON artifact (`to_json` / `from_json`) built by the Preparer from
development episodes; evaluation episodes start from a fresh copy and never write it back.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

SCHEMA = "fast-harness-memory-v1"


@dataclass(frozen=True)
class Statistics:
    successes: float = 0
    failures: float = 0
    total: int = 0
    clean_streak: int = 0
    ever_failed: bool = False


@dataclass(frozen=True)
class Match:
    stage: Optional[str]
    distance: float
    reason: str


def distance(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size == 0 or a.shape != b.shape:
        return math.inf
    return float(np.sqrt(np.mean((a - b) ** 2)))


def feature_json(features) -> str:
    if not len(features) or not all(math.isfinite(x) for x in features):
        raise ValueError("Features must be nonempty and finite")
    return json.dumps(list(features), separators=(",", ":"), allow_nan=False)


class Memory:
    def __init__(self, namespace: str, decay: float = 0.98, examples_per_stage: int = 128,
                 error_merge_radius: float = 0.04) -> None:
        if (not namespace.strip() or not 0 < decay <= 1 or examples_per_stage < 1
                or not math.isfinite(error_merge_radius) or error_merge_radius < 0):
            raise ValueError("Invalid memory configuration")
        self.namespace, self.decay = namespace, decay
        self.examples_per_stage, self.error_merge_radius = examples_per_stage, error_merge_radius
        self._next_id = 1
        self.examples: dict[str, dict[str, tuple[int, tuple]]] = {}   # stage -> fingerprint -> (id, features)
        self.stats: dict[str, Statistics] = {}
        self.events: set = set()                                       # (episode, step, kind)
        self.error_rows: dict[str, dict] = {}                          # case_id -> row (insertion ordered)
        self.spans: dict[str, tuple[float, int]] = {}

    # -- persistence (the frozen artifact) ----------------------------------------
    def to_json(self) -> dict:
        return {"schema": SCHEMA, "namespace": self.namespace, "decay": self.decay,
                "examples_per_stage": self.examples_per_stage, "error_merge_radius": self.error_merge_radius,
                "examples": {s: sorted(([i, list(f)] for i, f in rows.values()), key=lambda r: r[0])
                             for s, rows in sorted(self.examples.items())},
                "stats": {s: {"successes": v.successes, "failures": v.failures, "total": v.total,
                              "clean_streak": v.clean_streak, "ever_failed": v.ever_failed}
                          for s, v in sorted(self.stats.items())},
                "events": sorted([list(e) for e in self.events]),
                "errors": list(self.error_rows.values()),
                "spans": {s: list(v) for s, v in sorted(self.spans.items())}}

    @classmethod
    def from_json(cls, data: dict) -> "Memory":
        if data.get("schema") != SCHEMA:
            raise ValueError("Unsupported FastHarness memory artifact")
        mem = cls(data["namespace"], data["decay"], data["examples_per_stage"], data["error_merge_radius"])
        for stage, rows in data["examples"].items():
            for i, feats in rows:
                f = tuple(float(x) for x in feats)
                fp = hashlib.sha256(feature_json(f).encode()).hexdigest()
                mem.examples.setdefault(stage, {})[fp] = (int(i), f)
                mem._next_id = max(mem._next_id, int(i) + 1)
        mem.stats = {s: Statistics(**v) for s, v in data["stats"].items()}
        mem.events = {tuple(e) for e in data["events"]}
        mem.error_rows = {row["case_id"]: dict(row) for row in data["errors"]}
        mem.spans = {s: (float(v[0]), int(v[1])) for s, v in data["spans"].items()}
        return mem

    def empty(self) -> bool:
        return not self.examples and not self.stats and not self.error_rows

    # -- FastHarness Memory API -------------------------------------------------------
    def stages(self) -> tuple:
        return tuple(sorted(s for s, rows in self.examples.items() if rows))

    def learn_stage(self, stage: str, features) -> None:
        features = tuple(features)
        fp = hashlib.sha256(feature_json(features).encode()).hexdigest()
        rows = self.examples.setdefault(stage, {})
        rows[fp] = (self._next_id, features)      # INSERT OR REPLACE: a repeat gets a new id
        self._next_id += 1
        if len(rows) > self.examples_per_stage:   # keep the most recent examples_per_stage
            keep = sorted(rows.items(), key=lambda kv: kv[1][0])[-self.examples_per_stage:]
            self.examples[stage] = dict(keep)

    def match(self, features, threshold: float, margin: float) -> Match:
        feature_json(features)
        x = np.asarray(features, dtype=np.float64)
        nearest: dict[str, float] = {}
        for stage, rows in self.examples.items():
            if not rows:
                continue
            mat = np.asarray([f for _, f in rows.values()], dtype=np.float64)
            if mat.shape[1] != x.shape[0]:
                continue
            nearest[stage] = float(np.min(np.sqrt(np.mean((mat - x[None, :]) ** 2, axis=1))))
        ranked = sorted(nearest, key=nearest.get)
        if not ranked or nearest[ranked[0]] > threshold:
            return Match(None, nearest[ranked[0]] if ranked else math.inf, "unknown_context")
        if len(ranked) > 1 and nearest[ranked[1]] - nearest[ranked[0]] < margin:
            return Match(None, nearest[ranked[0]], "ambiguous_stage")
        return Match(ranked[0], nearest[ranked[0]], "matched")

    def statistics(self, stage: str) -> Statistics:
        return self.stats.get(stage, Statistics())

    def record(self, *, episode: str, step: int, kind: str, stage: str, features, outcome: str,
               evidence: str, student: bool, error_type: str = "") -> Optional[str]:
        if outcome not in ("ok", "error", "unknown"):
            raise ValueError("Invalid observed outcome")
        encoded = feature_json(features)
        case_id = hashlib.sha256(f"{stage}|{encoded}|{error_type}".encode()).hexdigest()[:24]
        key = (episode, int(step), kind)
        if key in self.events:
            return None
        self.events.add(key)
        if student and outcome != "unknown":
            old, good = self.statistics(stage), outcome == "ok"
            self.stats[stage] = Statistics(old.successes * self.decay + int(good),
                                           old.failures * self.decay + int(not good), old.total + 1,
                                           old.clean_streak + 1 if good else 0, old.ever_failed or not good)
        if outcome == "error":
            nearest = sorted((distance(features, row["features"]), cid) for cid, row in self.error_rows.items()
                             if row["stage"] == stage and row["error_type"] == error_type)
            if nearest and nearest[0][0] <= self.error_merge_radius:
                case_id = nearest[0][1]
            if not student:
                old = self.statistics(stage)
                self.stats[stage] = Statistics(old.successes, old.failures, old.total, 0, True)
            if case_id in self.error_rows:
                row = self.error_rows[case_id]
                row["count"] += 1
                row["evidence"] = evidence
            else:
                self.error_rows[case_id] = {"case_id": case_id, "stage": stage, "features": list(features),
                                            "error_type": error_type, "count": 1, "evidence": evidence,
                                            "recovered": 0, "recovery_evidence": ""}
            return case_id
        return None

    def errors(self, stage: Optional[str] = None, limit: int = 12) -> tuple:
        rows = [r for r in self.error_rows.values() if stage is None or r["stage"] == stage]
        rows.sort(key=lambda r: (-r["count"], r["case_id"]))
        keys = ("case_id", "stage", "error_type", "count", "evidence", "recovered", "recovery_evidence")
        return tuple({k: r[k] for k in keys} for r in rows[:limit])

    def near_error(self, stage: str, features, threshold: float) -> bool:
        return any(distance(features, r["features"]) <= threshold
                   for r in self.error_rows.values() if r["stage"] == stage)

    def recovered(self, case_ids, evidence: str) -> None:
        for cid in case_ids:
            if cid in self.error_rows:
                self.error_rows[cid]["recovered"] += 1
                self.error_rows[cid]["recovery_evidence"] = evidence

    def record_span(self, stage: str, chunks: int) -> None:
        if not isinstance(chunks, int) or chunks < 1:
            return
        if stage not in self.spans:
            self.spans[stage] = (float(chunks), 1)
        else:
            ewma, samples = self.spans[stage]
            self.spans[stage] = (ewma * 0.7 + chunks * 0.3, samples + 1)

    def span(self, stage: str) -> Optional[float]:
        return self.spans[stage][0] if stage in self.spans else None

    def summary(self) -> dict:
        return {"stages": len(self.stages()), "examples": sum(len(r) for r in self.examples.values()),
                "confirmed": {s: v.total for s, v in sorted(self.stats.items())},
                "errors": len(self.error_rows), "events": len(self.events)}
