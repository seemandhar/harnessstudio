"""The rescue planner: frozen prompts + strict JSON schemas, an isolated `codex exec` operator (vendored
CodexPlanner, gpt-6.1-sol) or a scripted operator (tier 1 / unit tests), per-call accounting that keeps semantic
outcomes (accept / abstain / wrong_object / invalid) apart from transport outcomes (timeout / transport_error /
budget_refused / isolation_error), review condition C5.

The prompts carry the ORIGINAL instruction and a neutral description of the public trigger event only. They never
name the benchmark, a task list, a success/status line or any per-task string (review C3, F; pitfall
P-20261031-06: no success cue).
"""
from __future__ import annotations

import functools
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import anyio
import numpy as np

DEFAULTS = {
    "backend": "codex",            # "codex" | "scripted"
    "model": "gpt-6.1-sol",
    "effort": "medium",            # frozen by qualification A1: one of low / medium / high
    "timeout_s": 180.0,
    "max_calls_episode": 12,
    "max_calls_escalation": 4,
    "max_concurrent": 16,          # process-wide cap on concurrent codex processes (operator report: <= 32)
    "codex_binary": "codex",
    "scratch_root": "/tmp/codexop-scratch",
    "scripted_responses": [],      # backend "scripted" only
}

SEMANTIC = ("accept", "abstain", "wrong_object", "invalid")
TRANSPORT = ("timeout", "transport_error", "budget_refused", "isolation_error", "parse_error")

_SEMAPHORES: dict = {}
_SEM_LOCK = threading.Lock()


def _semaphore(n: int) -> threading.BoundedSemaphore:
    with _SEM_LOCK:
        if n not in _SEMAPHORES:
            _SEMAPHORES[n] = threading.BoundedSemaphore(n)
        return _SEMAPHORES[n]


SYSTEM = ("You are the visual grounding module of a robot-arm controller. A frozen manipulation policy is carrying "
          "out a natural-language instruction in a tabletop scene. You never move the robot yourself: you answer one "
          "question about the attached camera images with exactly one JSON object that matches the given schema. "
          "Image coordinates: row 0 is the top edge and col 0 the left edge of the image they refer to. When you "
          "are not sure, abstain (null target or verdict 'abstain') instead of guessing.")

EVENTS = {
    "H2_empty_close": "The robot's gripper has just closed but is holding nothing: it missed the object it was "
                      "reaching for.",
    "drop": "The robot was carrying a lifted object and the gripper has just lost it.",
    "H1_settled_clamp": "The robot's gripper has been closed on something for a while without lifting it, and the "
                        "policy is about to open the gripper again.",
    "analytic_continue": "The robot has just finished moving an object (descriptive ablation run).",
}

GRASP_ACTIONS = ("grasp_and_place", "grasp_and_handback", "handback")
LIFT_ACTIONS = ("lift_and_place", "lift_and_handback", "handback")

_POINT = {"type": "object", "additionalProperties": False, "required": ["label", "row", "col"],
          "properties": {"label": {"type": "string"}, "row": {"type": "integer"}, "col": {"type": "integer"}}}
_DEST = {"type": "object", "additionalProperties": False, "required": ["label", "row", "col", "open_top"],
         "properties": {"label": {"type": "string"}, "row": {"type": "integer"}, "col": {"type": "integer"},
                        "open_top": {"type": "boolean"}}}

PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["assessment", "clamped_task_object", "clamped_at_destination", "target", "destination",
                 "grasp_style", "action"],
    "properties": {
        "assessment": {"type": "string"},
        "clamped_task_object": {"type": ["boolean", "null"]},
        "clamped_at_destination": {"type": ["boolean", "null"]},
        "target": {"anyOf": [_POINT, {"type": "null"}]},
        "destination": {"anyOf": [_DEST, {"type": "null"}]},
        "grasp_style": {"type": "string", "enum": ["center", "rim"]},
        "action": {"type": "string", "enum": sorted(set(GRASP_ACTIONS + LIFT_ACTIONS))},
    },
}

VERDICT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdict", "seen", "reason"],
    "properties": {"verdict": {"type": "string", "enum": ["confirm", "wrong_object", "abstain"]},
                   "seen": {"type": "string"}, "reason": {"type": "string"}},
}

CUSTODY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdict", "reason"],
    "properties": {"verdict": {"type": "string", "enum": ["confirm", "not_held", "abstain"]},
                   "reason": {"type": "string"}},
}


def plan_prompt(instruction: str, event: str, size: int) -> str:
    lift = event == "H1_settled_clamp"
    lines = [f'Instruction given to the robot: "{instruction}"', "", f"Event: {EVENTS[event]}", "",
             f"Image 1 is the fixed front scene camera ({size}x{size} pixels). Image 2 is the wrist camera on the "
             "gripper.", ""]
    if lift:
        lines += [
            "Answer:",
            "- clamped_task_object: true if the gripper is holding an object that the instruction needs moved, false "
            "if it holds something else or nothing useful, null if you cannot tell.",
            "- clamped_at_destination: true if that held object is already where the instruction wants it, false if "
            "not, null if you cannot tell.",
            "- target: the held object, named with words from the instruction (label), and one pixel (row, col) in "
            "image 1 that lies on that object; null if unsure.",
        ]
        actions = ("action: 'lift_and_place' (lift the held object and put it at the destination), "
                   "'lift_and_handback' (lift it and return control to the policy) or 'handback' (do nothing).")
    else:
        lines += [
            "Answer:",
            "- target: the object the gripper should pick up next to carry out the instruction, named with words "
            "from the instruction (label), and one pixel (row, col) in image 1 that lies ON that object's body (not "
            "on the robot, the table, a shadow or another object); null if it is not visible or you are unsure.",
            "- clamped_task_object and clamped_at_destination: null for this event.",
        ]
        actions = ("action: 'grasp_and_place' (grasp the target, then put it at the destination), "
                   "'grasp_and_handback' (grasp and lift the target, then return control to the policy) or "
                   "'handback' (do nothing).")
    lines += [
        "- destination: where the instruction wants that object, if it is visible: label (words from the "
        "instruction), one pixel (row, col) in image 1 at the spot where the object should be released, and "
        "open_top = true only if the object can be lowered onto/into it straight from above with nothing covering "
        "it; false for anything enclosed or that must first be opened or slid out. open_top is ALWAYS false for a "
        "drawer or any other furniture or appliance compartment (cabinet, shelf, microwave, oven), a caddy, and any "
        "enclosed or lidded container, even when it is open. null if the instruction gives no destination for this "
        "object or it is not visible.",
        "- grasp_style: 'center' for compact objects, 'rim' for wide open vessels best taken by their edge.",
        f"- {actions}",
        "- assessment: one short sentence on what you see.",
    ]
    return "\n".join(lines)


def size_text(size: Optional[Mapping[str, Any]]) -> str:
    """The perception-measured size of the marked object (fix2; generic, no per-object table)."""
    if not size or size.get("height") is None or not size.get("extent_xy"):
        return ""
    h, (a, b) = 100.0 * float(size["height"]), (100.0 * float(x) for x in size["extent_xy"][:2])
    return (f"\nThe depth camera measures the object under the mark as about {h:.1f} cm tall (top above the table) "
            f"with a visible footprint of about {a:.1f} x {b:.1f} cm (partly hidden objects can measure smaller). "
            f"If that height or size clearly does not fit a {{label}}, the mark is not on the {{label}}.")


def witness_prompt(instruction: str, label: str, held: bool, size: Optional[Mapping[str, Any]] = None) -> str:
    what = (f"Is the gripper at the mark holding the {label}?" if held else
            f"Is the CENTER of the mark on the {label} itself (not on the robot, the table, a shadow or another "
            "object)?")
    measured = "" if held else size_text(size).replace("{label}", label)
    return (f'Instruction given to the robot: "{instruction}"\n\nA red ring with a cross marks one point. Image 1 is '
            "the full front scene camera with the mark; image 2 is a 4x zoom around the same mark in the same frame."
            f"{measured}\n\n{what}\nverdict: 'confirm' if yes, 'wrong_object' if it is on something else, 'abstain' "
            "if you cannot tell. seen: what the mark is actually on. reason: one short sentence.")


_CUSTODY_VIEWS = {
    "before": "a zoom of the fixed front scene camera BEFORE the grasp, centred where the {label} was",
    "after": "the SAME zoom window of the same camera NOW, after the gripper closed there and lifted about 6 cm",
    "wrist": "the wrist camera on the gripper, now",
}


def custody_prompt(instruction: str, label: str, views: Sequence[str] = ("before", "after")) -> str:
    """Describes exactly the images sent, in order (no wrist claim when the benchmark sends no wrist RGB)."""
    lines = [f"Image {i}: " + _CUSTODY_VIEWS[v].format(label=label) + "." for i, v in enumerate(views, 1)]
    return (f'Instruction given to the robot: "{instruction}"\n\nThe robot has just tried to pick up the {label} and '
            "lift it.\n" + "\n".join(lines) + f"\n\nDid the {label} move up together with the gripper, i.e. is it "
            "held between the fingers and lifted clear of the surface it stood on? Compare the images: an object "
            "that stayed in place (or fell back) is not held.\nverdict: 'confirm' if clearly yes, 'not_held' if "
            "clearly not, 'abstain' if you cannot tell. reason: one short sentence.")


_STOP = {"the", "and", "put", "both", "pick", "place", "turn", "into", "onto", "then", "with", "from", "into", "on",
         "in", "it", "of", "to", "up", "a", "an", "at", "its", "them", "close", "open"}


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOP}


def label_ok(label: str, instruction: str) -> bool:
    """The label must reuse at least one content word of the instruction (generic check, no per-task table)."""
    return bool(content_words(label) & content_words(instruction))


def validate_plan(obj: Mapping[str, Any], event: str, instruction: str, size: int) -> Optional[str]:
    allowed = LIFT_ACTIONS if event == "H1_settled_clamp" else GRASP_ACTIONS
    act = obj.get("action")
    if act not in allowed:
        return f"action {act!r} not allowed for this event (use one of {list(allowed)})"
    for key in ("target", "destination"):
        pt = obj.get(key)
        if pt is None:
            continue
        if not isinstance(pt, Mapping):
            return f"{key} must be an object or null"
        try:
            r, c = int(pt["row"]), int(pt["col"])
        except (KeyError, TypeError, ValueError):
            return f"{key} needs integer row and col"
        if not (0 <= r < size and 0 <= c < size):
            return f"{key} pixel ({r}, {c}) is outside the {size}x{size} image"
    if act != "handback" and obj.get("target") is None:
        return "a grasp/lift action needs a target"
    return None


@dataclass
class CallResult:
    role: str
    status: str                       # "ok" or a TRANSPORT value
    outcome: Optional[str] = None     # SEMANTIC value when status == "ok"
    obj: Optional[dict] = None
    record: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        rec = {k: v for k, v in self.record.items() if k in _RECORD_KEYS and k != "status"}
        return {**rec, "operator_status": self.record.get("status"), "role": self.role, "status": self.status,
                "outcome": self.outcome, "reply": self.obj}


_RECORD_KEYS = ("call", "wall_s", "model_ids_reported", "effort_requested", "effort_reported", "cli_version",
                "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "tool_items",
                "isolation_ok", "isolation_problems", "reconnects", "reply_sha256", "image_sha256", "request_sha256",
                "output_schema_sha256", "error", "status", "pin", "backend", "exit_code", "queue_s")


class ScriptedPlanner:
    """Offline stand-in for CodexPlanner (same ask_json/close surface). Replies are consumed in order; a reply
    of the form {"__transport__": "timeout"} raises the matching transport error."""

    backend = "scripted"

    def __init__(self, responses: Sequence[Any], call_budget: int = 0) -> None:
        self.responses = list(responses)
        self.call_budget = int(call_budget)
        self.calls = 0
        self.records: list = []
        self.prompts: list = []

    def ask_json(self, prompt, images=(), system=None, schema=None, validate=None, retries: int = 0) -> dict:
        from .vendor.codex_planner import BudgetExceeded, CodexCallError
        if self.call_budget and self.calls >= self.call_budget:
            raise BudgetExceeded("scripted budget spent")
        self.calls += 1
        imgs = list(images)
        self.prompts.append({"prompt": prompt, "n_images": len(imgs), "schema": schema})
        rec = {"call": self.calls, "status": "ok", "backend": "scripted", "model_ids_reported": ["scripted"],
               "wall_s": 0.0, "n_images": len(imgs)}
        self.records.append(rec)
        if not self.responses:
            rec["status"] = "error"
            raise CodexCallError("scripted planner has no more replies")
        reply = self.responses.pop(0)
        if isinstance(reply, str):
            reply = json.loads(reply)
        if isinstance(reply, Mapping) and "__transport__" in reply:
            kind = reply["__transport__"]
            rec["status"] = kind
            if kind == "timeout":
                raise TimeoutError("scripted timeout")
            raise CodexCallError(f"scripted transport error {kind}")
        if validate is not None:
            err = validate(reply)
            if err:
                raise CodexCallError(f"no valid JSON reply after 1 attempts: {err}")
        return dict(reply)

    def close(self) -> None:
        pass


class PlannerSession:
    """One per episode. Charges the per-episode cap before every call; runs calls in a worker thread so the
    harness server's event loop keeps serving other episodes; kills the codex process on cancellation."""

    def __init__(self, params: Optional[Mapping[str, Any]] = None, log_path: Optional[str] = None,
                 episode_id: Optional[str] = None) -> None:
        p = dict(DEFAULTS)
        p.update(dict(params or {}))
        unknown = set(p) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown planner params {sorted(unknown)}")
        if "astra" in str(p["model"]).lower():
            raise ValueError("Astra models need the user's explicit confirmation for each use; not configured here")
        self.p = p
        self.log_path, self.episode_id = log_path, episode_id
        self._op = None
        self.calls = 0
        self.history: list = []

    def _operator(self):
        if self._op is None:
            if self.p["backend"] == "scripted":
                self._op = ScriptedPlanner(self.p["scripted_responses"], call_budget=int(self.p["max_calls_episode"]))
            elif self.p["backend"] == "codex":
                from .vendor.codex_planner import CodexPlanner
                self._op = CodexPlanner(model=self.p["model"], effort=self.p["effort"],
                                        timeout_s=float(self.p["timeout_s"]),
                                        call_budget=int(self.p["max_calls_episode"]), log_path=self.log_path,
                                        binary=self.p["codex_binary"], scratch_root=self.p["scratch_root"],
                                        episode_id=self.episode_id)
            else:
                raise ValueError(f"unknown planner backend {self.p['backend']!r}")
        return self._op

    @property
    def remaining(self) -> int:
        return max(0, int(self.p["max_calls_episode"]) - self.calls)

    async def ask(self, role: str, prompt: str, images: Sequence[np.ndarray], schema: Mapping[str, Any],
                  validate: Optional[Callable[[Mapping[str, Any]], Optional[str]]] = None) -> CallResult:
        from .vendor.codex_planner import BudgetExceeded, CodexCallError, CodexIsolationError
        if self.calls >= int(self.p["max_calls_episode"]):
            res = CallResult(role, "budget_refused", record={"error": "per-episode planner budget spent"})
            self.history.append(res)
            return res
        self.calls += 1
        try:
            op = self._operator()
        except Exception as exc:  # noqa: BLE001 - construction failure is transport, never semantics
            res = CallResult(role, "transport_error", record={"error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            self.history.append(res)
            return res
        sem = _semaphore(int(self.p["max_concurrent"]))
        n_before = len(getattr(op, "records", []))

        def call():
            t0 = time.monotonic()
            with sem:
                queued = time.monotonic() - t0
                return queued, op.ask_json(prompt, images=list(images), system=SYSTEM, schema=dict(schema),
                                           validate=validate, retries=0)

        status, obj, err = "ok", None, None
        queued = None
        try:
            queued, obj = await anyio.to_thread.run_sync(call, abandon_on_cancel=True)
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, anyio.get_cancelled_exc_class()) or not isinstance(exc, Exception):
                op.close()          # the episode ended or was cancelled: kill the running codex process group
                raise
            if isinstance(exc, BudgetExceeded):
                status = "budget_refused"
            elif isinstance(exc, TimeoutError):
                status = "timeout"
            elif isinstance(exc, CodexIsolationError):
                status = "isolation_error"
            elif isinstance(exc, CodexCallError) and "no valid JSON reply" in str(exc):
                status = "parse_error"
            else:
                status = "transport_error"
            err = f"{type(exc).__name__}: {str(exc)[:240]}"
        records = getattr(op, "records", [])
        rec = dict(records[-1]) if len(records) > n_before else {}
        if err:
            rec["error"] = err
        if queued is not None:
            rec["queue_s"] = round(queued, 3)
        res = CallResult(role, status, obj=obj, record=rec)
        self.history.append(res)
        return res

    def close(self) -> None:
        if self._op is not None:
            self._op.close()

    def summary(self) -> dict:
        out = {"calls": self.calls, "by_status": {}, "by_outcome": {}, "wall_s": 0.0, "model_ids": []}
        for r in self.history:
            out["by_status"][r.status] = out["by_status"].get(r.status, 0) + 1
            if r.outcome:
                out["by_outcome"][r.outcome] = out["by_outcome"].get(r.outcome, 0) + 1
            out["wall_s"] += float(r.record.get("wall_s") or 0.0)
            for m in r.record.get("model_ids_reported") or []:
                if m not in out["model_ids"]:
                    out["model_ids"].append(m)
        out["wall_s"] = round(out["wall_s"], 3)
        return out
