"""RoboHarness perception orchestrator (the paper's Brain + Limbs), ported from src/soma_perception.py
(PerceptionModule.process_frame) and src/soma_tools.py (RoboHarness repo, MIT, 716defc).

Per call: the Brain (VLM) reads the frame, the task and the retrieved failure diagnoses and returns a tool chain;
visual tools edit the frame through the tool server (SAM3 masks; the VLM detects a box first, as the original
service does); a modified frame is checked by the VLM verifier and, if rejected, the Brain re-plans with the
rejection reason, up to `max_attempts` times. Visual tools are allowed while the visual step counter is
<= 40 (it grows by 10 per call and resets when Encore fires). All of this is the original logic.

The one released value we change is the attempt count: the released file sets MAX_RETRIES = 0, which skips the
loop entirely (every call returns the frame unchanged). `max_attempts` restores the documented behaviour
("supports blocking retries upon verification failure"); see paper.yaml.
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
from typing import Any, Awaitable, Callable, Optional

import anyio
import numpy as np

from . import vlm as V

VISUAL_TOOLS = ["visual_overlay", "remove_distractor", "replace_texture", "replace_background"]
FLOW_TOOLS = ["encore", "key_step_retry", "task_decompose"]


def png_b64(image: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_b64_image(b64: str) -> np.ndarray:
    from PIL import Image
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


class ToolClient:
    """HTTP client for scripts/rh_tool_server.py (the original Sam3HttpClient plus the retrieval encoder).
    A failed or unsuccessful call returns the input image, as the original client does."""

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self.url, self.timeout = url.rstrip("/"), float(timeout)

    def _post_sync(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    async def post(self, path: str, payload: dict) -> dict:
        return await anyio.to_thread.run_sync(self._post_sync, path, payload)

    async def image_tool(self, path: str, image: np.ndarray, **fields: Any) -> tuple[np.ndarray, dict]:
        out = await self.post(path, {"image": png_b64(image), **fields})
        if out.get("success") and out.get("image"):
            return decode_b64_image(out["image"]), out
        return image, out

    async def embed(self, image: np.ndarray, text: str) -> np.ndarray:
        out = await self.post("/embed", {"image": png_b64(image), "text": text})
        return np.asarray(out["embedding"], dtype=np.float32)


LLM = Callable[[list, str], Awaitable[str]]   # (messages, operator) -> reply text ("{}" on provider failure)


class Perception:
    def __init__(self, llm: LLM, tools: ToolClient, max_attempts: int = 3, modification_active_limit: int = 40,
                 record: Optional[Callable[..., None]] = None, count_tool: Optional[Callable[[], None]] = None) -> None:
        self.llm, self.tools = llm, tools
        self.max_attempts = int(max_attempts)
        self.retry_count = 0
        self.visual_tool_steps = 0
        self.modification_active_limit = int(modification_active_limit)
        self._record = record or (lambda *a, **k: None)
        self._count_tool = count_tool or (lambda: None)

    async def _detect(self, image: np.ndarray, prompt: str) -> Optional[list]:
        reply = await self.llm(V.detect_messages(image, prompt), V.DETECTOR)
        return V.parse_detection(reply, int(image.shape[0]), int(image.shape[1]))

    async def _mask_tool(self, path: str, image: np.ndarray, prompt: str, **fields: Any) -> tuple[np.ndarray, dict]:
        """Original server-side _get_mask: VLM box (Strategy A) when available, then the SAM3 tool."""
        bbox = await self._detect(image, prompt) if prompt else None
        self._count_tool()
        try:
            out_img, out = await self.tools.image_tool(path, image, bbox=bbox, **fields)
        except Exception as exc:  # noqa: BLE001 - the original logs the tool failure and keeps the image
            out_img, out = image, {"success": False, "message": f"{type(exc).__name__}: {str(exc)[:160]}"}
        return out_img, {"tool": path.strip("/"), "prompt": prompt, "bbox": bbox,
                         "success": bool(out.get("success")), "message": str(out.get("message", ""))[:120]}

    async def _verify(self, original: np.ndarray, modified: np.ndarray, original_task: str, refined_task: str):
        reply = await self.llm(V.verify_messages(original, modified, original_task, refined_task), V.VERIFIER)
        return V.parse_verification(reply)

    async def process_frame(self, image: np.ndarray, current_task: str, step: int,
                            rag_context: Optional[dict] = None) -> tuple[np.ndarray, str, dict]:
        rag_context = rag_context or {}
        rag_str_base = ""
        rag_hints: dict = {}
        failures = rag_context.get("failure") or []
        successes = rag_context.get("success") or []
        if failures:
            diags = [str(f.get("diagnosis", "")) for f in failures[:2] if f.get("diagnosis")]
            if diags:
                rag_str_base = f"Warning! Past failures causes: {'; '.join(diags)}."

        def _has_tex(item: dict, key: str) -> bool:
            info = item.get("info") if isinstance(item, dict) else None
            assets = info.get("assets") if isinstance(info, dict) else None
            return isinstance(assets, dict) and bool(assets.get(key))

        rag_hints["success_has_object_texture"] = bool(successes) and _has_tex(successes[0], "object_texture_png_b64")
        rag_hints["success_has_floor_texture"] = bool(successes) and _has_tex(successes[0], "floor_texture_png_b64")
        rag_hints["failure_has_object_texture"] = bool(failures) and _has_tex(failures[0], "object_texture_png_b64")
        rag_hints["failure_has_floor_texture"] = bool(failures) and _has_tex(failures[0], "floor_texture_png_b64")

        current_attempt = 0
        final_processed_img = image.copy()
        final_refined_task = current_task
        final_control_flags: dict = {"encore": False}
        feedback_prompt = ""
        attempts_log: list = []

        while current_attempt < self.max_attempts:
            current_attempt += 1
            current_rag_str = rag_str_base
            if feedback_prompt:
                current_rag_str += (f"\n[IMMEDIATE FEEDBACK] Your previous plan was rejected. Reason: {feedback_prompt}. "
                                    "Please fix the visual modification.")
            t0 = time.monotonic()
            try:
                reply = await self.llm(V.orchestrate_messages(image, current_task, current_rag_str, rag_hints), V.BRAIN)
                plan = V.extract_json(reply) or {}
            except Exception as exc:  # noqa: BLE001 - original: "Planning failed" -> leave the loop
                attempts_log.append({"attempt": current_attempt, "planning_error": f"{type(exc).__name__}: {str(exc)[:160]}"})
                break
            if not isinstance(plan, dict):
                plan = {}
            tool_chain = plan.get("tool_chain", []) or []
            if not isinstance(tool_chain, list):
                tool_chain = []
            params = plan.get("params", {}) if isinstance(plan.get("params"), dict) else {}
            refined_task_text = plan.get("refined_task", current_task)
            task_plan = plan.get("task_plan", {}) if isinstance(plan.get("task_plan"), dict) else {}

            allow_visual = self.visual_tool_steps <= self.modification_active_limit
            tool_chain = [t for t in tool_chain if (t not in VISUAL_TOOLS) or allow_visual]

            temp_img = image.copy()
            temp_flags: dict = {"encore": False}
            if "key_steps" in task_plan:
                temp_flags["key_steps"] = task_plan.get("key_steps")
            if "subtasks" in task_plan:
                temp_flags["subtasks"] = task_plan.get("subtasks")
            image_modified_in_this_loop = False
            tool_log: list = []

            for tool_name in tool_chain:
                try:
                    tool_specific_params = params.get(tool_name, params) if isinstance(tool_name, str) else params
                    if not isinstance(tool_specific_params, dict):
                        raise TypeError(f"params for {tool_name} are not an object")
                    if tool_name in VISUAL_TOOLS:
                        if tool_name == "remove_distractor":
                            targets = tool_specific_params.get("objects_to_remove")
                            if targets is None:
                                single_target = tool_specific_params.get("object_to_remove")
                                if single_target:
                                    targets = [single_target]
                            if isinstance(targets, list):
                                for tgt in targets:
                                    if isinstance(tgt, str) and tgt:
                                        temp_img, info = await self._mask_tool("/remove_distractor", temp_img, tgt,
                                                                               object_to_remove=tgt)
                                        tool_log.append(info)
                                        image_modified_in_this_loop = True
                        elif tool_name == "visual_overlay":
                            target = tool_specific_params.get("target_object")
                            if isinstance(target, str) and target:
                                temp_img, info = await self._mask_tool("/visual_overlay", temp_img, target,
                                                                       target_object=target, color=[0, 255, 0], alpha=0.4)
                                tool_log.append(info)
                                image_modified_in_this_loop = True
                        elif tool_name in ("replace_texture", "replace_background"):
                            # Texture assets come from retrieved memory entries (info.assets). The released logger
                            # never stores any, so these tools only run if a memory entry carries a texture.
                            source = tool_specific_params.get("source", "rag_success")
                            pick_list = successes if source == "rag_success" else failures
                            default_key = "object_texture_png_b64" if tool_name == "replace_texture" else "floor_texture_png_b64"
                            texture_key = tool_specific_params.get("texture_key", default_key)
                            tex = None
                            if pick_list:
                                info0 = pick_list[0].get("info", {})
                                assets = info0.get("assets", {}) if isinstance(info0, dict) else {}
                                tex = assets.get(texture_key) if isinstance(assets, dict) else None
                            if tool_name == "replace_texture":
                                target = tool_specific_params.get("target_object")
                                if isinstance(target, str) and target and tex:
                                    temp_img, info = await self._mask_tool("/replace_texture", temp_img, target,
                                                                           target_object=target, texture_image=tex)
                                    tool_log.append(info)
                                    image_modified_in_this_loop = True
                            else:
                                region_prompt = tool_specific_params.get("region_prompt", "floor")
                                alpha = float(tool_specific_params.get("alpha", 1.0))
                                if isinstance(region_prompt, str) and region_prompt and tex:
                                    temp_img, info = await self._mask_tool("/replace_background", temp_img, region_prompt,
                                                                           region_prompt=region_prompt, texture_image=tex,
                                                                           alpha=alpha)
                                    tool_log.append(info)
                                    image_modified_in_this_loop = True
                    elif tool_name in ("instruction_refine", "chaining_step"):
                        pass
                    elif tool_name == "encore":
                        temp_flags["encore"] = True
                        self.retry_count += 1
                    elif tool_name == "key_step_retry":
                        temp_flags["key_step_retry"] = True
                        if "key_steps" in tool_specific_params:
                            temp_flags["key_steps"] = tool_specific_params.get("key_steps")
                    elif tool_name == "task_decompose":
                        temp_flags["task_decompose"] = True
                        subtasks = tool_specific_params.get("subtasks")
                        if isinstance(subtasks, list) and subtasks:
                            temp_flags["subtasks"] = subtasks
                except Exception as exc:  # noqa: BLE001 - original: log and continue with the next tool
                    tool_log.append({"tool": str(tool_name), "error": f"{type(exc).__name__}: {str(exc)[:160]}"})

            verification_passed, reject_reason = True, ""
            if image_modified_in_this_loop and allow_visual:
                try:
                    is_valid, reason = await self._verify(image, temp_img, current_task, refined_task_text)
                except Exception as exc:  # noqa: BLE001 - original: do not block on verification errors
                    is_valid, reason = True, f"Verification error: {exc}"
                if not is_valid:
                    verification_passed, reject_reason = False, reason
            attempts_log.append({"attempt": current_attempt, "tool_chain": tool_chain, "refined_task": refined_task_text,
                                 "reasoning": str(plan.get("reasoning", ""))[:400], "tools": tool_log,
                                 "image_modified": image_modified_in_this_loop, "verified": verification_passed,
                                 "reject_reason": str(reject_reason)[:200], "seconds": round(time.monotonic() - t0, 2)})
            if verification_passed:
                final_processed_img = temp_img
                final_refined_task = refined_task_text
                final_control_flags = temp_flags
                if image_modified_in_this_loop:
                    final_control_flags["image_modified"] = True
                break
            feedback_prompt = reject_reason

        self.visual_tool_steps += 10   # the original adds 10 whether or not the frame was modified
        if final_control_flags.get("encore", False):
            self.visual_tool_steps = 0
        self._record("rh_perception", step=int(step), task=current_task, rag=rag_str_base[:600],
                     attempts=attempts_log, refined_task=final_refined_task,
                     flags={k: v for k, v in final_control_flags.items()},
                     visual_tool_steps=self.visual_tool_steps)
        return final_processed_img, final_refined_task, final_control_flags
