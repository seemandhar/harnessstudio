"""The RoboHarness "Brain": prompts and parsers ported verbatim from src/soma_vlm.py and the verification prompt
in src/soma_perception.py (RoboHarness repo, MIT, commit 716defc).

Only the transport changes: the original calls an OpenAI-compatible client directly; here each prompt is built as
a message list and sent through HarnessStudio's pinned operators (budgeted and traced), one operator per original
sampling setting:
    brain     orchestrate_perception / failure report / success summary   temperature 0.01, max_tokens 1024
    detector  detect_object (box for SAM3 mask selection)                  temperature 0.0,  max_tokens 128
    verifier  _verify_modification                                         temperature 0.1,  max_tokens 256
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Optional, Union

import numpy as np

from harnessstudio.tools.llm import image_data_url

BRAIN, DETECTOR, VERIFIER = "brain", "detector", "verifier"
MAX_TOKENS = {BRAIN: 1024, DETECTOR: 128, VERIFIER: 256}


def encode_image(image: Union[np.ndarray, bytes, str]) -> str:
    """PNG data URL, as the original _encode_image (float images in [0, 1] are scaled to uint8 first)."""
    if isinstance(image, str):
        return image if image.startswith("data:image") else "data:image/jpeg;base64," + image
    if isinstance(image, (bytes, bytearray)):
        kind = "png" if bytes(image[:8]) == b"\x89PNG\r\n\x1a\n" else "jpeg"
        return f"data:image/{kind};base64," + base64.b64encode(image).decode("ascii")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return image_data_url(arr)


def extract_json(text: str) -> Optional[dict]:
    """Original _extract_json (soma_vlm.py): direct parse, fenced block, outermost braces."""
    text = text or ""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        if "```json" in text:
            t = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            t = text.split("```")[1].split("```")[0]
        else:
            t = text
        return json.loads(t.strip())
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


ORCHESTRATE_SYSTEM = (
    "You are the visual perception cortex of a robot manipulation system (SOMA).\n"
    "Goal: Preprocess input image and task prompt to help robotic policy.\n"
    "TOOLS:\n"
    "1. 'remove_distractor': Remove ALL confusing objects/distractors. Params: objects_to_remove (List[str])\n"
    "2. 'visual_overlay': Highlight objects (e.g. 'red cup'). Params: target_object\n"
    "3. 'replace_texture': Replace object texture from success memory. Params: target_object, source='rag_success'\n"
    "4. 'replace_background': Replace background. Params: region_prompt, source='rag_success'\n"
    "5. 'key_step_retry': Flag to trigger key step retry in policy. Params: key_steps (optional list of steps)\n"
    "6. 'task_decompose': Flag to trigger task decomposition in policy. Params: subtasks (optional list of subtask descriptions)\n"
    "7. 'encore': No image modification, but refine task description. Params: refined_task\n"
    "OUTPUT JSON: { \"reasoning\": \"...\", \"refined_task\": \"...\", \"tool_chain\": [\"tool_name_1\"], \"params\": {...} }"
    "Tips:"
    "1. Use 'visual_overlay' if you identify key objects. Use 'remove_distractor' if there are misleading elements. Use 'replace_texture' if success memory has better visual features. Use 'replace_background' if the background is noisy. Use 'key_step_retry' if key steps need to be retried. Use 'task_decompose' if the task should be broken down into subtasks. Use 'encore' if you want to refine the task description without modifying the image. Always provide reasoning in the output.\n"
    "2. RAG Context and Hints can guide your decision. For example, if success memory has clear object texture but failure doesn't, consider using 'replace_texture' to enhance the current image.\n"
    "3. If the image is already clear and well-aligned with the task, you can choose to do no modification but still provide a refined task description.\n"
    "4. If use visual_overlay, replace the origin object description in refined_task with the highlighted version (e.g. 'cup' -> 'green highlighted cup') to help downstream policy attend to it. We use Green highlight by default. If use else tools, do NOT output 'highlighted' word in the task description.\n"
    "5. You CAN and SHOULD combine tools. For example, use 'remove_distractor' to clear vision AND 'visual_overlay' to guide the policy. The policy will focus on the 'refined_task' after all tools are applied.\n"
    "6. You SHOULD give more detailed object descriptions of visual_overlay and remove_distractor in params to help the vision module execute them accurately. E.g., 'visual_overlay': {'target_object': 'the green cup'}, 'remove_distractor': {'objects_to_remove': ['the red cup on the left', 'the red cup on the right']}.\n"
    "7. If visual_overlay and remove_distractor are both triggered, execute visual_overlay first to ensure the overlay is not removed by remove_distractor.\n"
    "8. Always keep the refined_task concise and focused on the main manipulation goal. If the original task is 'Pick up the red cup and place it on the plate', a good refined_task could be 'Pick up the highlighted green cup and place it on the plate', NOT 'Pick up the red cup and place it on the plate while ignoring the blue book and the yellow bowl'."
    "9. If the background is too noisy and hinders object recognition, consider using 'replace_background' to simplify the scene."
    "10. If the original task description is noisy or ambiguous, refine it to be clearer and more specific."
    "11. CRITICAL: If there are multiple distractors of the same type (e.g. multiple wrong bowls), you MUST list EACH of them individually in the 'objects_to_remove' list. Do NOT use a single generic phrase like 'all red bowls'. Instead, use specific descriptions for each one so the vision system can target them precisely (e.g. ['red bowl on top left', 'red bowl on bottom right'])."
)


def orchestrate_messages(image: Any, task_desc: str, rag_context: str = "", rag_hints: Optional[dict] = None) -> list:
    """Original orchestrate_perception message list."""
    hints_str = ""
    if rag_hints:
        if rag_hints.get("success_has_object_texture"):
            hints_str += "- Success memory has object texture available.\n"
        if rag_hints.get("failure_has_object_texture"):
            hints_str += "- Failure memory suggests object texture issue.\n"
    user_prompt = f"Current Task: {task_desc}\nContext: {rag_context}\nHints: {hints_str}\nAnalyze image. Do we need intervention?"
    return [{"role": "system", "content": ORCHESTRATE_SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": user_prompt},
                                         {"type": "image_url", "image_url": {"url": encode_image(image)}}]}]


FAILURE_SYSTEM = (
    "Role: Robot Failure Analyst.\n"
    "Return JSON keys: 'intrinsic_analysis' (direct_cause, observation), 'mcp_remediation' (tool_name, parameters).\n"
    "direct_cause examples: 'Visual Ambiguity', 'Distractor Interference', 'Control Error'.\n"
    "mcp_remediation tools: 'visual_overlay', 'remove_distractor', 'enhance_contrast'."
)
SUCCESS_SYSTEM = (
    "Role: Robotics Perception Analyst.\n"
    "Return JSON keys: 'visual_context_snapshot' (object_color, lighting), 'execution_summary' (one sentence why it worked)."
)


def failure_report_messages(images: list, task_prompt: str, anchor_example: Optional[dict] = None) -> list:
    anchor_text = ""
    if anchor_example:
        anchor_text = f"\nReference Success: {anchor_example.get('execution_summary', '')}"
    content = [{"type": "text", "text": f"Task: {task_prompt}\nAnalyze these keyframes of a failed episode.{anchor_text}"}]
    content += [{"type": "image_url", "image_url": {"url": encode_image(img)}} for img in images]
    return [{"role": "system", "content": FAILURE_SYSTEM}, {"role": "user", "content": content}]


def success_description_messages(images: list, task_prompt: str) -> list:
    content = [{"type": "text", "text": f"Task: {task_prompt}\nAnalyze these success keyframes."}]
    content += [{"type": "image_url", "image_url": {"url": encode_image(img)}} for img in images]
    return [{"role": "system", "content": SUCCESS_SYSTEM}, {"role": "user", "content": content}]


def diagnosis_from_failure_report(report: Optional[dict]) -> str:
    """Original ExperienceLogger._run_failure_diagnosis formatting."""
    analysis = (report or {}).get("intrinsic_analysis", {})
    if not isinstance(analysis, dict):
        analysis = {}
    return f"{analysis.get('direct_cause', 'Unknown')}: {analysis.get('observation', '')}"


def summary_from_success_report(report: Optional[dict]) -> str:
    return (report or {}).get("execution_summary", "Task completed successfully.")


DETECT_SYSTEM = "You are a robotic vision assistant. Output the bounding box of the target object."


def detect_messages(image: Any, prompt: str) -> list:
    user_prompt = (
        f"Detect the bounding box for: '{prompt}'. "
        "Return ONLY the box coordinates in JSON format: [ymin, xmin, ymax, xmax] (normalized 0-1000). "
        "Example: [150, 200, 450, 600]"
    )
    return [{"role": "system", "content": DETECT_SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": user_prompt},
                                         {"type": "image_url", "image_url": {"url": encode_image(image)}}]}]


def parse_detection(content: str, h: int, w: int) -> Optional[list]:
    """Original detect_object parsing: first four integers as [ymin, xmin, ymax, xmax] / 1000 -> [x1, y1, x2, y2]."""
    numbers = re.findall(r"\d+", content or "")
    if len(numbers) >= 4 and h > 0 and w > 0:
        ymin = float(numbers[0]) / 1000 * h
        xmin = float(numbers[1]) / 1000 * w
        ymax = float(numbers[2]) / 1000 * h
        xmax = float(numbers[3]) / 1000 * w
        return [int(xmin), int(ymin), int(xmax), int(ymax)]
    return None


def verify_messages(original_img: Any, modified_img: Any, original_task: str, refined_task: str) -> list:
    """Original PerceptionModule._verify_modification prompt (two images, then the text)."""
    prompt = (
        f"I have performed a visual modification to assist a robot policy.\n"
        f"Original Task: '{original_task}'\n"
        f"Refined Task: '{refined_task}'\n\n"
        f"Image 1 is the Original View.\n"
        f"Image 2 is the Modified View (with overlay/removal).\n\n"
        f"Please verify:\n"
        f"1. Does the visual modification correctly highlight the target or remove the distractor described in the task suggestions?\n"
        f"2. Is the refined task instruction clear and accurate?\n"
        f"If the modification is wrong, misleading, or unnecessary, return 'valid': false.\n"
        f"Respond strictly in JSON format: {{'valid': boolean, 'reason': 'string'}}."
    )
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": encode_image(original_img)}},
                                         {"type": "image_url", "image_url": {"url": encode_image(modified_img)}},
                                         {"type": "text", "text": prompt}]}]


def parse_verification(content: str) -> tuple[bool, str]:
    """Original parsing: substring between the first '{' and the last '}' through json.loads. Any exception
    (including invalid JSON) passes the modification, as the original's `except` branch does."""
    try:
        content = content or ""
        result = json.loads(content[content.find("{"):content.rfind("}") + 1])
        return bool(result.get("valid", False)), str(result.get("reason", "No reason provided"))
    except Exception as e:  # noqa: BLE001
        return True, f"Verification error: {e}"
