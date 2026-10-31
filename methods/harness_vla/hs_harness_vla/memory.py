"""Few-shot (memory) mode of Harness VLA: task memory distilled from exploration on a reference scene.

Paper (arXiv:2607.08448, RPent): before evaluation an exploration agent works on ONE reference seed of each task
(with RESET), and writes task-specific memory: an audit JSON (localization table, what worked, failure modes) plus
the replayable recipe JSONL (the successful command sequence with the parameters actually used). Evaluation reads
that memory for the observed task language, re-localizes every entity in the new scene and reuses the technique,
never the absolute coordinates.

Here (HarnessStudio development stage):
  - development = `hs prepare harness-vla-sol-fewshot` on the reference scene (init state 0) of every task; RPent's
    RESET is emulated by chained prepare rounds that start from the same init state again (attempt k = round k),
    each round seeing the memory written by the earlier rounds;
  - the controller records every turn (`turn` events: the planner's thought, the commands with their parameters and
    their results); `Preparer.build` distills them deterministically (no LLM) into `memory.jsonl`, one line per
    attempt in RPent's task-memory shape: audit (localization table, outcome, failure lessons) + recipe (the executed
    primitive sequence; every target point also expressed as an offset from the landmark it was aimed at);
  - evaluation loads `memory.jsonl` read-only through ctx.artifacts and gives the planner the attempts whose task
    language equals the current instruction (`render`).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

MEMORY_FORMAT = "harness-vla-task-memory/1"
PERCEPTION = ("segment", "back_project")
POINT_ACTIONS = ("move_to", "move_pose")
RESULT_KEYS = ("reached", "final_eef_pos", "final_dist_m", "steps_used", "chunks_used", "lift_detected",
               "final_gripper_opening", "min_gripper_opening", "episode_ended", "error")


def norm_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _r(v: Any, nd: int = 3) -> Any:
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, (list, tuple)):
        return [_r(x, nd) for x in v]
    return v


def _compact_result(res: dict) -> dict:
    out = {k: _r(res[k]) for k in RESULT_KEYS if k in res and res[k] is not None}
    if "stalled" in res:
        out["stalled"] = True
    if isinstance(out.get("error"), str):
        out["error"] = out["error"][:160]
    return out


def _nearest_landmark(xyz: list, landmarks: dict) -> Optional[tuple]:
    best = None
    for label, lm in landmarks.items():
        c = lm.get("world_xyz")
        if not c or len(c) < 3:
            continue
        d = math.hypot(xyz[0] - c[0], xyz[1] - c[1])
        if best is None or d < best[0]:
            best = (d, label, lm)
    if best is None or best[0] > 0.25:
        return None
    _, label, lm = best
    c = lm["world_xyz"]
    rel = {"landmark": label, "offset_from_center": [round(xyz[i] - c[i], 3) for i in range(3)]}
    if lm.get("world_z_top") is not None:
        rel["height_over_top"] = round(xyz[2] - lm["world_z_top"], 3)
    return rel


def attempt_from_record(rec: dict) -> dict:
    """One development episode -> one attempt entry (audit + recipe), from the controller's trace events only."""
    task = str(rec.get("task", ""))
    cell, _, instruction = task.partition("/")
    landmarks: dict = {}          # latest sighting per segment prompt (used for the recipe's relative targets)
    first_seen: dict = {}         # first sighting per prompt = the scene's initial layout (shown in the audit)
    not_found: list = []
    recipe: list = []
    lessons: list = []
    thoughts: list = []
    stop_reason = None
    for ev in rec.get("events") or []:
        if ev.get("t") != "event":
            continue
        if ev.get("name") == "planner_stop":
            stop_reason = str(ev.get("reason", ""))[:200]
            continue
        if ev.get("name") != "turn":
            continue
        turn, step = ev.get("turn"), ev.get("step")
        if ev.get("thought"):
            thoughts.append({"turn": turn, "step": step, "thought": str(ev["thought"])[:300]})
        for item in ev.get("results") or []:
            cmd, res = dict(item.get("command") or {}), dict(item.get("result") or {})
            action = str(cmd.get("action", ""))
            if action == "segment":
                label = str(cmd.get("prompt") or f"point {cmd.get('point')}")
                if res.get("world_xyz") and res.get("found", True):
                    landmarks[label] = {"world_xyz": _r(res["world_xyz"]), "world_z_top": _r(res.get("world_z_top")),
                                        "world_x_range": _r(res.get("world_x_range")),
                                        "world_y_range": _r(res.get("world_y_range")),
                                        "score": _r(res.get("score")), "camera": cmd.get("camera", "agentview"),
                                        "seen_at_step": step}
                    first_seen.setdefault(label, landmarks[label])
                else:
                    not_found.append(label)
                continue
            if action in PERCEPTION:
                continue
            entry = {"action": action, **{k: _r(v) for k, v in cmd.items() if k != "action"},
                     "result": _compact_result(res), "at_step": step}
            xyz = cmd.get("xyz")
            if action in POINT_ACTIONS and isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
                rel = _nearest_landmark([float(v) for v in xyz[:3]], landmarks)
                if rel:
                    entry["target_relative"] = rel
            recipe.append(entry)
            # failure lessons (what did not work, with its parameters)
            if "error" in res:
                lessons.append(f"{action} {json.dumps({k: _r(v) for k, v in cmd.items() if k != 'action'})} -> "
                               f"error: {str(res['error'])[:120]}")
            elif "stalled" in res:
                lessons.append(f"{action} to {_r(cmd.get('xyz'))} stalled at {res.get('final_eef_pos')} "
                               f"(blocked/out of reach)")
            elif action == "vla_act" and cmd.get("stop", "object_lifted") == "object_lifted" \
                    and not res.get("lift_detected") and not res.get("episode_ended"):
                lessons.append(f"vla_act '{cmd.get('prompt')}' (stop object_lifted, {res.get('chunks_used')} chunks) "
                               f"did not lift the object; gripper opening {res.get('final_gripper_opening')}")
    success = bool(rec.get("success"))
    steps = int((rec.get("accounting") or {}).get("steps", 0))
    if success:
        last = recipe[-1] if recipe else None
        outcome = (f"SUCCESS at step {steps}" + (f" (last primitive before the goal fired: {last['action']})"
                                                 if last else ""))
    else:
        outcome = f"FAILED (no success in {steps} steps; planner stop: {stop_reason or 'horizon reached while acting'})"
    return {"format": MEMORY_FORMAT, "kind": "attempt", "cell": cell, "instruction": instruction,
            "instruction_key": norm_instruction(instruction), "attempt": int(rec.get("episode_idx", 0)),
            "success": success, "steps": steps, "outcome": outcome, "landmarks": first_seen,
            "segment_not_found": sorted(set(not_found) - set(first_seen)), "recipe": recipe, "lessons": lessons[:12],
            "planner_notes": thoughts[:14], "planner_stop": stop_reason,
            "llm_tokens": int((rec.get("accounting") or {}).get("llm_tokens", 0))}


def read_memory(path: Optional[str]) -> list:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


class Preparer:
    """Development stage: exploration episodes run the few-shot controller itself (candidate arm)."""
    collect_arm = "candidate"
    record: tuple = ()   # the turn events carry everything the memory needs; no per-step observations

    def build(self, records: list, out_dir, previous: Optional[dict] = None) -> dict:
        entries = read_memory((previous or {}).get("memory"))
        seen = {(e["cell"], e["instruction_key"], e["attempt"], e.get("dup", 0)) for e in entries}
        for rec in records:
            if not rec.get("valid"):
                continue
            att = attempt_from_record(rec)
            key = (att["cell"], att["instruction_key"], att["attempt"])
            dup = 0
            while key + (dup,) in seen:  # two tasks with one instruction (LIBERO-Pro goal_task): keep both
                dup += 1
            att["dup"] = dup
            seen.add(key + (dup,))
            entries.append(att)
        entries.sort(key=lambda e: (e["cell"], e["instruction_key"], e["attempt"], e.get("dup", 0)))
        out = Path(out_dir) / "memory.jsonl"
        out.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
        return {"memory": "memory.jsonl"}


# -- evaluation-time retrieval and rendering ------------------------------------------------------
def _fmt_cmd(e: dict) -> str:
    args = {k: v for k, v in e.items() if k not in ("action", "result", "at_step", "target_relative")}
    s = f"{e['action']} {json.dumps(args, separators=(',', ':'))}"
    rel = e.get("target_relative")
    if rel:
        s += (f"  [= '{rel['landmark']}' center + {rel['offset_from_center']}"
              + (f", {rel['height_over_top']:+.3f} m over its top" if rel.get("height_over_top") is not None else "")
              + "]")
    res = e.get("result") or {}
    s += f"  -> {json.dumps(res, separators=(',', ':'))}"
    return s


def _render_attempt(a: dict, full: bool) -> str:
    lines = [f"- Attempt {a['attempt'] + 1}{' (second task with this instruction)' if a.get('dup') else ''}: "
             f"{a['outcome']}."]
    if a["landmarks"]:
        lm = "; ".join(f"'{k}' center {v['world_xyz']} top z {v.get('world_z_top')}"
                       for k, v in list(a["landmarks"].items())[:8])
        lines.append(f"  Landmarks it localized, first sighting (reference-scene coordinates, STALE here): {lm}")
    if a.get("segment_not_found"):
        lines.append(f"  segment prompts that found nothing: {a['segment_not_found'][:6]}")
    if full:
        lines.append("  Executed primitive sequence (recipe):")
        lines += [f"    {i + 1}. {_fmt_cmd(e)}" for i, e in enumerate(a["recipe"][:18])]
        if a.get("planner_notes"):
            lines.append("  Its reasoning, turn by turn: " + " | ".join(n["thought"] for n in a["planner_notes"][:8]))
    else:
        steps = ", ".join(f"{e['action']}" + (f" '{e.get('prompt')}'" if e.get("prompt") else "")
                          for e in a["recipe"][:14])
        lines.append(f"  What it did: {steps}")
    if a["lessons"]:
        lines.append("  What did not work: " + " | ".join(a["lessons"][:6]))
    return "\n".join(lines)


def render(entries: list, instruction: str, max_chars: int = 9000) -> str:
    """Task memory text for the planner: every attempt on the reference scene of this task language. The latest
    successful attempt is shown in full (recipe + reasoning); the others as outcome + lessons."""
    key = norm_instruction(instruction)
    mine = [e for e in entries if e.get("kind") == "attempt" and e.get("instruction_key") == key]
    if not mine:
        return ""
    wins = [e for e in mine if e["success"]]
    best = max(wins, key=lambda e: (e["attempt"], -e["steps"])) if wins else None
    ordered = sorted(mine, key=lambda e: (e["attempt"], e.get("dup", 0)))
    parts = [_render_attempt(a, full=(a is best) or (best is None and a is ordered[-1])) for a in ordered]
    head = (f"TASK MEMORY (few-shot mode) for \"{instruction}\": {len(mine)} exploration attempt(s) on a REFERENCE "
            f"scene of this same task ({len(wins)} succeeded), written before this episode. THIS scene is a different "
            "initialization: objects and fixtures may have moved. Reuse the TECHNIQUE (which entity to localize and "
            "how to prompt segment, staging offsets relative to the landmark, VLA prompts and chunk counts, carry "
            "height, release height, order of primitives) and avoid what failed; re-derive every coordinate from "
            "THIS scene's segment results (never copy absolute coordinates).")
    text = head + "\n" + "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + " ...[memory truncated]"
    return text
