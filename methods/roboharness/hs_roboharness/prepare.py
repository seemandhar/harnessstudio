"""Development stage: the paper's "In-Context Initialization" (Dual-Memory RAG populated by a preliminary pass on
standard LIBERO-Spatial and LIBERO-Object), frozen as `memory.json`.

`hs prepare` runs the candidate arm in memory-pass mode (controller.py: perception off, as the released code;
agent-view keyframes recorded every 5 steps). For each development episode this replays the original
ExperienceLogger.log_episode (src/soma_logger.py):
  - 5 frames uniformly spaced over the episode (original: from the episode video; here: the recorded keyframe
    nearest each position, at most 2 steps away)
  - index vector = encoder(first frame, task)  (tool server /embed: original CLIP + MiniLM + hash encoder)
  - failure -> VLM failure report "direct_cause: observation"; success -> VLM one-sentence execution summary
  - store in the success or failure partition (info = {}: the released logger passes no extra fields)
Outcomes of development episodes are allowed here; evaluation never writes to the bank.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import numpy as np

from . import vlm as V
from .memory import MemoryBank
from .perception import ToolClient, decode_b64_image

N_FRAMES = 5


def keyframes(record: dict) -> tuple[str, dict]:
    task, frames = "", {}
    for ev in record.get("events") or ():
        if ev.get("t") != "event":
            continue
        if ev.get("name") == "rh_memory_pass":
            task = str(ev.get("task", ""))
        elif ev.get("name") == "rh_keyframe":
            frames[int(ev["step"])] = ev["jpeg_b64"]
    return task, frames


def sample_frames(frames: dict, steps: int, n: int = N_FRAMES) -> list:
    """Original np.linspace(0, total_frames - 1, n) over the episode video (one frame per executed step),
    mapped to the nearest recorded keyframe."""
    if not frames:
        return []
    have = np.array(sorted(frames))
    total = max(1, int(steps))
    out = []
    for i in np.linspace(0, total - 1, n, dtype=int):
        out.append(frames[int(have[np.argmin(np.abs(have - i))])])
    return out


class Preparer:
    collect_arm = "candidate"
    record = ()

    def __init__(self, tool_url: str = "http://127.0.0.1:8131", workers: int = 8) -> None:
        self.tool_url = os.environ.get("HS_ROBOHARNESS_TOOL_URL", tool_url)
        self.workers = int(workers)

    def build(self, records: list, out_dir) -> dict:
        from harnessstudio.tools.llm import ScriptedOperator, build_operators
        from . import config as method_config
        # Tier 1 (HarnessStudio's synthetic "fake" benchmark only): no network; scripted diagnoses and a
        # deterministic stand-in embedding. Real benchmarks always use the pinned VLM and the tool server.
        offline = bool(records) and all(r.get("benchmark") == "fake" for r in records)
        if offline:
            brain = ScriptedOperator(['{"intrinsic_analysis": {"direct_cause": "Control Error", "observation": "missed"}}',
                                      '{"execution_summary": "grasped and placed"}'])
        else:
            brain = build_operators(method_config().operators)[V.BRAIN]
        tools = ToolClient(self.tool_url, timeout=120.0)
        bank = MemoryBank()
        log = {"episodes": [], "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        order = sorted(records, key=lambda r: (str(r.get("benchmark")), str(r.get("task")), int(r.get("episode_idx", 0))))
        jobs = []   # (entry, embedding, messages, success, rec) in replay order
        for rec in order:
            task, frames = keyframes(rec)
            steps = int((rec.get("accounting") or {}).get("steps", 0))
            diag = sample_frames(frames, steps)
            entry = {"benchmark": rec.get("benchmark"), "task": rec.get("task"), "episode_idx": rec.get("episode_idx"),
                     "success": bool(rec.get("success")), "steps": steps, "keyframes": len(frames), "instruction": task}
            if not diag or not task:
                entry["skipped"] = "no keyframes recorded (not a memory-pass episode)"
                log["episodes"].append(entry)
                continue
            index_img = decode_b64_image(diag[0])
            if offline:
                rng = np.random.default_rng(int.from_bytes(task.encode()[:8].ljust(8, b"_"), "little"))
                embedding = rng.standard_normal(1168).astype(np.float32)
            else:
                embedding = tools._post_sync("/embed", {"image": base64.b64encode(_png(index_img)).decode(),
                                                        "text": task})["embedding"]
            jpegs = [base64.b64decode(b) for b in diag]
            messages = (V.failure_report_messages(jpegs, task) if not rec.get("success")
                        else V.success_description_messages(jpegs, task))
            jobs.append((entry, embedding, messages, bool(rec.get("success")), rec))

        def call(messages):
            t0 = time.monotonic()
            try:
                text, usage = brain.complete(messages, V.MAX_TOKENS[V.BRAIN])
                err = None
            except Exception as exc:  # noqa: BLE001 - original _generate returns "{}" on API errors
                text, usage, err = "{}", {}, f"{type(exc).__name__}: {str(exc)[:200]}"
            return text, usage, err, round(time.monotonic() - t0, 2)

        # The diagnosis calls are independent; run them concurrently (scripted tier-1 replies stay sequential).
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1 if offline else self.workers) as pool:
            replies = list(pool.map(call, [j[2] for j in jobs]))
        for (entry, embedding, _, success, rec), (text, usage, err, secs) in zip(jobs, replies):
            log["llm_calls"] += 1
            log["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            log["completion_tokens"] += int(usage.get("completion_tokens", 0))
            report = V.extract_json(text) or {}
            diagnosis = V.diagnosis_from_failure_report(report) if not success else V.summary_from_success_report(report)
            rid = f"{'success' if success else 'failure'}_{len(log['episodes']):04d}"
            bank.add_experience(embedding, entry["instruction"], success, diagnosis=diagnosis, info={}, record_id=rid,
                                source={k: rec.get(k) for k in ("benchmark", "task", "episode_idx")})
            entry.update({"id": rid, "diagnosis": diagnosis, "reply": text[:2000], "llm_seconds": secs,
                          **({"llm_error": err} if err else {})})
            log["episodes"].append(entry)
        out_dir = Path(out_dir)
        (out_dir / "memory.json").write_text(json.dumps(bank.to_json()))
        log["memory"] = bank.stats()
        (out_dir / "build_log.json").write_text(json.dumps(log, indent=1))
        return {"memory": "memory.json", "build_log": "build_log.json"}


def _png(img: np.ndarray) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()
