"""Harness VLA zero-shot controller: a turn-based planner REPL over a fixed primitive library (paper §2.1-2.3).

Each turn the planner (a pinned NRP VLM standing in for Claude Code / Codex) gets the task language, the
coordinate-free robot state, the step budget left, the results of its last commands and the current camera
images, and answers with JSON commands: perception queries (`segment` = SAM3 + depth back-projection,
`back_project`), analytic primitives (move_to, move_pose, rotate_wrist, rotate_pitch, set_gripper, release) or
the VLA primitive (`vla_act`: the frozen policy with the planner's prompt and an early-return predicate).
Zero-shot: no Task Specific Memory, no Global Memory, no reference-seed exploration, no reset.

When the planner finishes, its turn budget is spent, or the LLM fails repeatedly, the arm holds still until the
episode ends (RPent stops the episode on `finish`; handing control back to the native policy would mix the
baseline into the method).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import numpy as np

from harnessstudio.core.errors import HarnessError
from harnessstudio.tools.llm import image_data_url

from . import perception as P
from . import primitives as prim
from .memory import read_memory, render
from .prompts import FIRST, system_prompt

TIER1_SCRIPT = [
    '{"thought": "localize the target first", "commands": [{"action": "segment", "prompt": "black bowl"}, '
    '{"action": "back_project", "row": 100, "col": 120}]}',
    'not json at all',
    '{"thought": "approach", "commands": [{"action": "move_to", "xyz": [0.1, -0.05, 1.05], "gripper": -1, "max_steps": 3}]}',
    '{"thought": "grasp", "commands": [{"action": "vla_act", "prompt": "pick up the bowl", "max_chunks": 1, '
    '"stop": "object_lifted"}, {"action": "rotate_wrist", "delta_yaw": 0.2}]}',
    '{"thought": "carry and drop", "commands": [{"action": "set_gripper", "gripper": 1, "steps": 1}, '
    '{"action": "release", "max_steps": 2}]}',
    '{"thought": "done", "commands": [{"action": "finish", "reason": "tier 1 script"}]}',
]

PERCEPTION = ("segment", "back_project")
MAX_COMMANDS = 3


def parse_reply(text: str) -> tuple[Optional[dict], str]:
    """First JSON object in the reply (after removing <think> blocks and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            if "commands" not in obj and "action" in obj:
                obj = {"thought": "", "commands": [obj]}
            cmds = obj.get("commands")
            if isinstance(cmds, dict):
                cmds = [cmds]
            if isinstance(cmds, list) and cmds and all(isinstance(c, dict) for c in cmds):
                return {"thought": str(obj.get("thought", ""))[:600], "commands": cmds}, ""
    return None, 'reply must be one JSON object {"thought": "...", "commands": [{"action": ...}, ...]}'


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _upsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


class HarnessVLA:
    TIER1_LLM_RESPONSES = TIER1_SCRIPT

    def __init__(self, max_turns: int = 40, max_tool_calls: int = 60, image_scale: int = 2, wrist_image: bool = True,
                 history_turns: int = 8, llm_retries: int = 2, sam3_url: str = "http://127.0.0.1:8117",
                 max_tokens: int = 800, memory: bool = False, memory_chars: int = 9000) -> None:
        self.max_turns, self.max_tool_calls = int(max_turns), int(max_tool_calls)
        self.image_scale, self.wrist_image = int(image_scale), bool(wrist_image)
        self.history_turns, self.llm_retries, self.max_tokens = int(history_turns), int(llm_retries), int(max_tokens)
        # Few-shot (memory) mode: task memory from the development stage (ctx.artifacts["memory"], read-only).
        self.memory, self.memory_chars = bool(memory), int(memory_chars)
        # Endpoint of the SAM3 server (a deployment detail, not a method parameter): env var overrides the default.
        self.sam3 = P.Sam3Client(os.environ.get("HS_HARNESS_VLA_SAM3_URL", sam3_url))
        self.tool_calls = 0
        self.repeats = 0
        self._last_reply: Optional[str] = None
        self.landmarks: dict = {}   # the planner's localization table (RPent PERCEPTION_ALGORITHM step 6)
        self.gripper_cmd = -1.0
        self.log: list = []

    # -- observation for the planner ---------------------------------------------------------
    def _images(self, obs: dict, overlays: list) -> list:
        parts = []
        av = P.calibration_image(obs, "agentview")
        if av is not None:
            parts.append({"type": "text", "text": f"agentview image ({av.shape[0] * self.image_scale}x"
                                                  f"{av.shape[1] * self.image_scale}, calibration frame):"})
            parts.append({"type": "image_url", "image_url": {"url": image_data_url(_upsample(av, self.image_scale))}})
        if self.wrist_image:
            wr = P.calibration_image(obs, "wrist")
            if wr is not None:
                parts.append({"type": "text", "text": "wrist camera image (moves with the gripper):"})
                parts.append({"type": "image_url", "image_url": {"url": image_data_url(_upsample(wr, self.image_scale))}})
        for label, img in overlays[-3:]:
            parts.append({"type": "text", "text": f"SAM3 overlay for {label} (mask in red):"})
            parts.append({"type": "image_url", "image_url": {"url": image_data_url(_upsample(img, self.image_scale))}})
        return parts

    def _state_text(self, ctx) -> str:
        state = prim.proprio(ctx)
        out = {"robot_state": state, "steps_left": int(ctx.remaining), "gripper_command": self.gripper_cmd,
               "status": "goal NOT reached yet (the episode ends by itself on success)"}
        eef = state.get("eef_pos")
        if self.landmarks and eef and len(eef) >= 3:
            table = {}
            for label, lm in list(self.landmarks.items())[-8:]:
                x0, x1 = lm["world_x_range"]
                y0, y1 = lm["world_y_range"]
                table[label] = {**{k: lm[k] for k in ("world_xyz", "world_z_top", "seen_at_step")},
                                "eef_minus_center": [round(eef[i] - lm["world_xyz"][i], 3) for i in range(3)],
                                "eef_above_its_footprint": bool(x0 <= eef[0] <= x1 and y0 <= eef[1] <= y1),
                                "eef_height_over_its_top": round(eef[2] - lm["world_z_top"], 3)}
            out["localization_table"] = table
        return json.dumps(out)

    @staticmethod
    def _strip_images(messages: list) -> list:
        out = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                c = [p if p.get("type") != "image_url" else {"type": "text", "text": "[old image omitted]"} for p in c]
            out.append({"role": m["role"], "content": c})
        return out

    def _repeats_failure(self, cmd: dict) -> bool:
        """Anti-loop guard: an identical motion command right after it failed/stalled is refused (no env steps)."""
        last = getattr(self, "_last", None)
        return (last is not None and last[1] and str(cmd.get("action")) not in PERCEPTION
                and json.dumps(cmd, sort_keys=True) == last[0])

    # -- one command --------------------------------------------------------------------------
    async def _run_command(self, ctx, cmd: dict, overlays: list) -> dict:
        action = str(cmd.get("action", "")).strip()
        args = {k: v for k, v in cmd.items() if k != "action"}
        obs = ctx.observe()
        if action in PERCEPTION:
            if self.tool_calls >= self.max_tool_calls:
                return {"error": f"perception budget of {self.max_tool_calls} calls is spent"}
            self.tool_calls += 1
            ctx.accounting.tool_calls += 1
            camera = str(args.get("camera", "agentview"))
            if action == "segment":
                res, overlay = P.segment(obs, self.sam3, camera=camera, prompt=str(args.get("prompt") or ""),
                                         point=args.get("point"), min_score=float(args.get("min_score", 0.2)),
                                         scale=float(self.image_scale))
                if overlay is not None:
                    overlays.append((f"'{args.get('prompt') or args.get('point')}' ({camera})", overlay))
                if res.get("world_xyz") and res.get("world_x_range"):
                    label = str(args.get("prompt") or f"point {args.get('point')}")
                    self.landmarks[label] = {k: res[k] for k in ("world_xyz", "world_x_range", "world_y_range",
                                                                  "world_z_top")}
                    self.landmarks[label]["seen_at_step"] = int(ctx.steps)
                return res
            return P.back_project(obs, camera=camera, row=args.get("row"), col=args.get("col"),
                                  row_range=args.get("row_range"), col_range=args.get("col_range"),
                                  z_min=args.get("z_min"), z_max=args.get("z_max"), scale=float(self.image_scale))
        if action not in prim.PRIMITIVES:
            return {"error": f"unknown action {action!r}; allowed: {sorted(list(prim.PRIMITIVES) + list(PERCEPTION) + ['finish'])}"}
        try:
            result = await prim.PRIMITIVES[action](ctx, **args)
        except HarnessError:
            raise  # budgets, stops, policy and contract errors belong to the framework
        except Exception as exc:  # noqa: BLE001 - a bad command must not crash the episode
            return {"error": f"bad arguments or failure in {action}: {type(exc).__name__}: {str(exc)[:200]}"}
        if "error" not in result:
            if action in ("move_to", "move_pose", "rotate_wrist", "rotate_pitch", "set_gripper"):
                g = args.get("gripper", -1.0 if action in ("move_to", "move_pose") else 1.0)
                self.gripper_cmd = prim._gripper(g)  # noqa: SLF001
            elif action == "release":
                self.gripper_cmd = -1.0
            elif action == "vla_act":
                self.gripper_cmd = 1.0 if (result.get("final_gripper_opening") or 0.08) < 0.07 else -1.0
        return result

    async def _hold(self, ctx) -> None:
        """Stop: keep the pose and the last gripper command until the episode ends (no policy calls)."""
        a = np.zeros(int(ctx.embodiment.action_dim or 7), dtype=np.float32)
        try:
            a[ctx.embodiment.dims("gripper")] = self.gripper_cmd
        except KeyError:
            pass
        while not ctx.ended:
            n = min(50, ctx.remaining)
            if n <= 0:
                break
            await ctx.execute(np.repeat(a[None, :], n, axis=0))

    @staticmethod
    def _debug_dump(ctx, tag: str) -> None:
        """Development aid: HS_HARNESS_VLA_DEBUG_DIR=<dir> saves the public observation (images, depth, camera,
        state) as .npz. Off by default; never read back by the method."""
        root = os.environ.get("HS_HARNESS_VLA_DEBUG_DIR")
        if not root:
            return
        try:
            obs = ctx.observe()
            flat = {"instruction": np.asarray(str(obs.get("task_description", "")))}
            for group in ("images", "depth"):
                for k, v in (obs.get(group) or {}).items():
                    flat[f"{group}.{k}"] = np.asarray(v)
            for cam, meta in (obs.get("camera") or {}).items():
                for k, v in meta.items():
                    flat[f"camera.{cam}.{k}"] = np.asarray(v)
            if obs.get("states") is not None:
                flat["states"] = np.asarray(obs["states"])
            os.makedirs(root, exist_ok=True)
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{ctx.instruction}_{tag}_{time.time_ns()}")
            np.savez_compressed(os.path.join(root, name[:150] + ".npz"), **flat)
        except Exception:  # noqa: BLE001 - debugging must never affect the episode
            pass

    # -- the REPL -----------------------------------------------------------------------------
    async def run(self, ctx) -> None:
        llm = ctx.tools["llm"]
        horizon = ctx.steps + ctx.remaining
        system = {"role": "system", "content": system_prompt(horizon, memory=self.memory)}
        first = [{"type": "text", "text": FIRST.format(instruction=ctx.instruction)}]
        if self.memory:
            entries = read_memory(ctx.artifacts.get("memory"))
            text = render(entries, ctx.instruction, max_chars=self.memory_chars)
            ctx.record("task_memory", entries_total=len(entries), found=bool(text), chars=len(text))
            first.append({"type": "text", "text": text or "TASK MEMORY: none for this task language (no reference-"
                                                          "scene exploration recorded); act zero-shot."})
        first += [{"type": "text", "text": self._state_text(ctx)}] + self._images(ctx.observe(), [])
        history: list = [{"role": "user", "content": first}]
        self._debug_dump(ctx, "start")
        failures = 0
        finish_reason = "turn_budget"
        for turn in range(self.max_turns):
            if ctx.ended:
                return
            msgs = [system] + self._strip_images(history[:-1]) + history[-1:]
            if len(history) > 2 * self.history_turns + 1:  # keep the task message and the last turns
                msgs = [system, self._strip_images(history[:1])[0]] + self._strip_images(
                    history[-2 * self.history_turns:-1]) + history[-1:]
            t0 = time.monotonic()
            try:
                reply = await llm(msgs, operator="planner", max_tokens=self.max_tokens)
            except Exception as exc:  # noqa: BLE001 - provider errors, budget
                failures += 1
                ctx.record("llm_error", turn=turn, error=f"{type(exc).__name__}: {str(exc)[:200]}")
                if failures > self.llm_retries or "budget" in str(exc).lower():
                    finish_reason = "llm_failure"
                    break
                continue
            plan, err = parse_reply(reply)
            if plan is not None and json.dumps(plan["commands"], sort_keys=True) == self._last_reply:
                # Degenerate repetition (the same commands as the previous reply, whose results the planner already
                # has): do not execute, do not add the duplicate to the context (copying feeds the loop), and say so.
                self.repeats += 1
                ctx.record("repeat_rejected", turn=turn, commands=plan["commands"])
                note = {"type": "text", "text": "NOTE: your last reply repeated your previous commands exactly; it was "
                                                "NOT executed. Their results are above. Decide something different "
                                                "based on them."}
                last = history[-1]
                if isinstance(last.get("content"), list):
                    last["content"] = [p for p in last["content"] if not str(p.get("text", "")).startswith("NOTE: your last")] + [note]
                continue
            if plan is None:
                ctx.record("parse_error", turn=turn, reply=reply[:500])
                history += [{"role": "assistant", "content": reply[:2000]},
                            {"role": "user", "content": [{"type": "text", "text": "ERROR: " + err},
                                                         {"type": "text", "text": self._state_text(ctx)}]
                             + self._images(ctx.observe(), [])}]
                continue
            results, overlays, stop = [], [], False
            for cmd in plan["commands"][:MAX_COMMANDS]:
                if ctx.ended:
                    break  # the goal fired (or the horizon hit) during the previous command: record the turn
                if str(cmd.get("action")) == "finish":
                    stop = True
                    finish_reason = "finish: " + str(cmd.get("reason", ""))[:200]
                    break
                if self._repeats_failure(cmd):
                    res = {"error": "this exact command failed or stalled last time; it would fail again. Choose a "
                                    "different command (other target, waypoint, approach side or primitive)."}
                else:
                    res = _jsonable(await self._run_command(ctx, cmd, overlays))
                self._last = (json.dumps(cmd, sort_keys=True), ("error" in res) or ("stalled" in res))
                results.append({"command": cmd, "result": res})
                if "error" in res:
                    break
            ctx.record("turn", turn=turn, thought=plan["thought"], results=results,
                       llm_seconds=round(time.monotonic() - t0, 2), steps=ctx.steps)
            self._debug_dump(ctx, f"turn{turn:02d}")
            if stop or ctx.ended:
                break
            dropped = len(plan["commands"]) - len(results)
            note = (f" ({dropped} later command(s) not executed)" if dropped > 0 and results
                    and "error" in results[-1]["result"] else "")
            self._last_reply = json.dumps(plan["commands"], sort_keys=True)
            history += [{"role": "assistant", "content": json.dumps(plan)},
                        {"role": "user", "content": [
                            {"type": "text", "text": "RESULTS" + note + ": " + json.dumps(results)},
                            {"type": "text", "text": self._state_text(ctx)}] + self._images(ctx.observe(), overlays)}]
        if ctx.ended and finish_reason == "turn_budget":
            finish_reason = "episode_ended"
        ctx.record("planner_stop", reason=finish_reason, tool_calls=self.tool_calls, repeats_rejected=self.repeats)
        if not ctx.ended:
            await self._hold(ctx)
