"""RoboHarness controller: the authors' evaluation rollout (src/soma_eval.py `rollout`) step for step.

Per environment step (original order):
  1. current task = active subtask (TaskDecomposeState) or the benchmark instruction
  2. every `perceive_every` (10) steps, unless a rollback is running: Perception.process_frame on the agent-view
     frame -> (frame, refined task, flags); the refined task replaces the instruction for this step (unless
     subtasks are active); flags may start Encore (rollback 40 + buffer 20), set subtasks or key-step timeouts
  3. action: rollback action while a rollback runs (accumulator cleared), else LeRobot `select_action`:
     a queue of `n_action_steps` (10) actions refilled from one policy chunk when empty, computed from the
     current (possibly modified) frame and task; executed policy actions are added to the accumulator
  4. step; key-step timeout may start a rollback (reverse 50, buffer 25)
Episode start: Dual-Memory RAG retrieval with the original encoder on the first frame + task (top-1 success,
top-2 failure, cosine >= 0.4) from the frozen memory artifact.

Success signals: the original passes the environment's per-step success to TaskDecomposeState and
KeyStepRetryState. The LIBERO env terminates on success, so inside a running episode that signal is always
False; this port passes False and never reads evaluator state.

Development (`hs prepare`, ctx.development and no memory artifact): the released memory pass. Perception is off
(the released MAX_RETRIES = 0 path, so actions equal the native policy's) and the agent-view frame is recorded
every `keyframe_every` steps (the original's periodic keyframes) for prepare.Preparer to diagnose and embed.
"""
from __future__ import annotations

import base64
import io
import os
from collections import deque
from typing import Any, Optional

import numpy as np

from .control_flow import KeyStepRetryState, RollbackState, TaskDecomposeState
from .memory import MemoryBank
from .perception import Perception, ToolClient
from . import vlm as V

TIER1_SCRIPT = [
    # brain (step 0): overlay + distractor removal, refined task
    '{"reasoning": "two similar objects", "refined_task": "pick up the green highlighted block", '
    '"tool_chain": ["visual_overlay", "remove_distractor"], "params": {"visual_overlay": {"target_object": "the block"}, '
    '"remove_distractor": {"objects_to_remove": ["the cup on the left"]}}}',
    "[100, 120, 400, 500]",                     # detector (overlay target)
    "no box here",                              # detector (distractor): falls back to pure SAM3
    "{'valid': false, 'reason': 'single quotes'}",   # verifier reply that is not JSON -> passes (original)
    # brain (step 10): flow tools only
    '```json\n{"reasoning": "stuck", "refined_task": "", "tool_chain": ["encore", "task_decompose", "key_step_retry"], '
    '"params": {"task_decompose": {"subtasks": ["reach the block", "lift the block"]}, '
    '"key_step_retry": {"key_steps": {"start": 0, "end": 3, "timeout_grace": 1}}}}\n```',
    'not json',                                 # brain: unparseable -> no intervention
]


def _image(obs: dict, key: str) -> tuple[Optional[str], Optional[np.ndarray]]:
    images = obs.get("images") or {}
    if key in images:
        return key, np.asarray(images[key])
    for k, v in images.items():
        return k, np.asarray(v)
    return None, None


def _jpeg_b64(img: np.ndarray, quality: int) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


class RoboHarness:
    TIER1_LLM_RESPONSES = TIER1_SCRIPT

    def __init__(self, max_attempts: int = 3, perceive_every: int = 10, n_action_steps: int = 10,
                 modification_active_limit: int = 40, encore_reverse_steps: int = 40, encore_buffer_steps: int = 20,
                 retry_reverse_steps: int = 50, retry_buffer_steps: int = 25, success_top_k: int = 1,
                 failure_top_k: int = 2, retrieval_threshold: float = 0.4, image_key: str = "agentview",
                 tool_url: str = "http://127.0.0.1:8131", tool_timeout: float = 120.0, keyframe_every: int = 5,
                 keyframe_quality: int = 90) -> None:
        self.max_attempts, self.perceive_every, self.n_action_steps = int(max_attempts), int(perceive_every), int(n_action_steps)
        self.modification_active_limit = int(modification_active_limit)
        self.encore = (int(encore_reverse_steps), int(encore_buffer_steps))
        self.retry = (int(retry_reverse_steps), int(retry_buffer_steps))
        self.success_top_k, self.failure_top_k = int(success_top_k), int(failure_top_k)
        self.retrieval_threshold = float(retrieval_threshold)
        self.image_key = image_key
        # Tool server endpoint is a deployment detail: the env var overrides the default.
        self.tools = ToolClient(os.environ.get("HS_ROBOHARNESS_TOOL_URL", tool_url), timeout=tool_timeout)
        self.keyframe_every, self.keyframe_quality = int(keyframe_every), int(keyframe_quality)

    async def _llm(self, ctx, messages: list, operator: str) -> str:
        """Original _generate contract: a provider error yields "{}" (no intervention), never an exception."""
        try:
            return await ctx.tools["llm"](messages, operator=operator, max_tokens=V.MAX_TOKENS[operator])
        except Exception as exc:  # noqa: BLE001
            ctx.record("rh_llm_error", operator=operator, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return "{}"

    async def _retrieve(self, ctx, frame: np.ndarray, task: str) -> dict:
        bank = MemoryBank.load(ctx.artifacts.get("memory"))
        try:
            ctx.accounting.tool_calls += 1
            q = await self.tools.embed(frame, task)
        except Exception as exc:  # noqa: BLE001 - original: "[SOMA] Init Failed" -> empty RAG context
            ctx.record("rh_rag", error=f"{type(exc).__name__}: {str(exc)[:200]}", memory=bank.stats())
            return {}
        succ = bank.retrieve(q, top_k=self.success_top_k, partition="success", threshold=self.retrieval_threshold)
        fail = bank.retrieve(q, top_k=self.failure_top_k, partition="failure", threshold=self.retrieval_threshold)
        ctx.record("rh_rag", memory=bank.stats(),
                   success=[{"id": r.get("id"), "task": r.get("task"), "score": round(r["score"], 4)} for r in succ],
                   failure=[{"id": r.get("id"), "task": r.get("task"), "score": round(r["score"], 4),
                             "diagnosis": str(r.get("diagnosis", ""))[:300]} for r in fail])
        return {"success": succ, "failure": fail}

    async def run(self, ctx) -> None:
        memory_pass = bool(getattr(ctx, "development", False)) and not ctx.artifacts.get("memory")
        perception = Perception(lambda m, op: self._llm(ctx, m, op), self.tools,
                                max_attempts=0 if memory_pass else self.max_attempts,
                                modification_active_limit=self.modification_active_limit, record=ctx.record,
                                count_tool=lambda: setattr(ctx.accounting, "tool_calls", ctx.accounting.tool_calls + 1))
        rollback, task_ctrl, retry_ctrl = RollbackState(), TaskDecomposeState(), KeyStepRetryState()
        base_task = ctx.instruction
        obs = ctx.observe()
        key, frame0 = _image(obs, self.image_key)
        rag: dict = {}
        if memory_pass:
            ctx.record("rh_memory_pass", task=base_task, image_key=key, keyframe_every=self.keyframe_every)
        elif frame0 is not None:
            rag = await self._retrieve(ctx, frame0, base_task)
        action_dim = int(ctx.embodiment.action_dim or 7)
        accumulated = np.zeros((1, 7), dtype=np.float32)
        queue: deque = deque()
        step = 0
        while not ctx.ended:
            current_task = task_ctrl.current_task(default_task=base_task)
            obs = ctx.observe()
            key, frame = _image(obs, self.image_key)
            if memory_pass and step % self.keyframe_every == 0 and frame is not None:
                ctx.record("rh_keyframe", step=step, jpeg_b64=_jpeg_b64(frame, self.keyframe_quality))
            override = None
            if step % self.perceive_every == 0 and not rollback.active and frame is not None and not memory_pass:
                processed, refined, flags = await perception.process_frame(frame, current_task, step, rag)
                if flags.get("image_modified", False):
                    override = processed
                if refined and isinstance(refined, str) and not task_ctrl.subtasks:
                    current_task = refined
                if flags.get("encore", False):
                    rollback.start_from_accumulated(accumulated, reverse_steps=self.encore[0],
                                                    buffer_steps=self.encore[1], reason="SOMA_Encore")
                    ctx.record("rh_rollback", step=step, reason="encore", reverse_action=rollback.reverse_action[0].tolist())
                if "subtasks" in flags:
                    task_ctrl.update_from_control_flags(flags)
                if "key_steps" in flags:
                    retry_ctrl.update_from_control_flags(flags)
            if rollback.active:
                action = rollback.step_action(1)[0]
                accumulated.fill(0.0)
            else:
                if not queue:
                    payload = None
                    if override is not None and key is not None:
                        payload = {"images": {**(obs.get("images") or {}), key: override}}
                    chunk = (await ctx.policy.sample(k=1, obs=payload, instruction=current_task))[0].actions
                    chunk = np.asarray(chunk, dtype=np.float32)
                    queue.extend(chunk[: self.n_action_steps])
                    if payload is not None or current_task != base_task:
                        ctx.record("rh_decision", step=step, instruction=current_task, image_modified=payload is not None)
                action = np.asarray(queue.popleft(), dtype=np.float32)
                accumulated += action[:7].reshape(1, -1) if action.shape[0] >= 7 else 0.0
            if action.shape[0] != action_dim:
                action = action[:action_dim]
            if await ctx.execute(action[None, :]) == 0:
                break
            task_ctrl.maybe_advance(success_any=False)
            if retry_ctrl.should_trigger(step=step, success_any=False):
                retry_ctrl.mark_triggered()
                rollback.start_from_accumulated(accumulated, reverse_steps=self.retry[0],
                                                buffer_steps=self.retry[1], reason="Timeout_Retry")
                ctx.record("rh_rollback", step=step, reason="key_step_timeout", key_steps=retry_ctrl.key_steps)
            step += 1
