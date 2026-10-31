"""Development stage: FastHarness's cross-episode experience as a frozen artifact.

`hs prepare` runs the candidate arm (cold: empty experience, every chunk reviewed, as the paper's first
episode) on development episodes. Each episode's trace holds every experience-store mutation (`fh_mem`
events: stage examples, reviewed outcomes, error cases, recoveries, chunk spans). The Preparer replays
them in a fixed order (task, episode_idx) through the same Memory code, which is what FastHarness's SQLite
store would contain after those episodes, and writes it as `experience.json` (sha256-pinned by
`prepared.json`). Evaluation loads it read-only; evaluation outcomes never reach it.
"""
from __future__ import annotations

import json
from pathlib import Path

from .memory import Memory


def replay(records: list, decay: float = 1.0, previous: dict | None = None) -> dict:
    """previous: an earlier experience.json (chained development); new episodes are replayed on top of it."""
    memories: dict = {}
    episodes = []
    if previous:
        memories = {ns: Memory.from_json(m) for ns, m in previous.get("memories", {}).items()}
        episodes = list(previous.get("episodes", []))
    order = sorted(records, key=lambda r: (str(r.get("task")), int(r.get("episode_idx", 0))))
    for rec in order:
        episode = f"{rec.get('benchmark')}/{rec.get('task')}/{rec.get('episode_idx')}"
        episodes.append(episode)
        for ev in rec.get("events") or ():
            if ev.get("t") != "event" or ev.get("name") != "fh_mem":
                continue
            ns = ev["namespace"]
            mem = memories.setdefault(ns, Memory(ns, decay=decay))
            op = ev["op"]
            if op == "learn_stage":
                mem.learn_stage(ev["stage"], tuple(ev["features"]))
            elif op == "record":
                mem.record(episode=episode, step=int(ev["obs_step"]), kind=ev["record_kind"], stage=ev["stage"],
                           features=tuple(ev["features"]), outcome=ev["outcome"], evidence=ev["evidence"],
                           student=bool(ev["student"]), error_type=ev.get("error_type", ""))
            elif op == "recovered":
                mem.recovered(set(ev["case_ids"]), ev["evidence"])
            elif op == "record_span":
                mem.record_span(ev["stage"], int(ev["chunks"]))
    return {"schema": "fast-harness-experience-v1", "episodes": episodes,
            "memories": {ns: m.to_json() for ns, m in sorted(memories.items())},
            "summary": {ns: m.summary() for ns, m in sorted(memories.items())}}


class Preparer:
    collect_arm = "candidate"
    record = ()

    def __init__(self, memory_decay: float = 1.0) -> None:
        self.memory_decay = memory_decay

    def build(self, records: list, out_dir, previous: dict | None = None) -> dict:
        prev = None
        if previous and previous.get("experience"):
            prev = json.loads(Path(previous["experience"]).read_text())
        data = replay(records, self.memory_decay, prev)
        (Path(out_dir) / "experience.json").write_text(json.dumps(data, sort_keys=True, allow_nan=False))
        return {"experience": "experience.json"}
