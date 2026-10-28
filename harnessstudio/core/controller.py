"""Controller slot (milestone 2): agent harnesses that own the episode's decision loop.

A controller is built from ``slots["controller"]`` and gets ONE call per episode:
``run(tools)``. It drives the robot only through ``tools.act(prompt, steps)``, which runs the
normal pipeline (proposer / selector / executor) for whole policy decisions. It sees the public
observation, the original instruction and its own LLM operator; never simulator state, reward,
success, case, seed or arm. When ``run`` returns early, the episode finishes natively.

Arms:
- baseline: no controller.
- null (shadow mode): the controller runs and pays for its LLM calls, but every act() executes
  native decisions with the original instruction, so the null stays comparable to the baseline.
- candidate: act() uses the controller's prompt and step cap.

Budget: the candidate's policy-call bound uses ``min_act_steps``; every decision an act() runs
executes at least that many steps unless the episode ends first.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol

from .arms import Arm
from .errors import MethodContractError, RegistrationError
from .llm import LLMBudgetExhausted, build_operator, validate_llm_params


class Controller(Protocol):
    def run(self, tools: "ControllerTools") -> None: ...


def validate_controller_params(params: Mapping[str, Any]) -> None:
    if type(params.get("min_act_steps")) is not int or params["min_act_steps"] < 1:
        raise RegistrationError("A controller must declare integer min_act_steps >= 1")
    if params.get("llm") is not None:
        if not isinstance(params["llm"], Mapping):
            raise RegistrationError("controller llm params must be a mapping")
        validate_llm_params(params["llm"])


class ControllerTools:
    def __init__(self, *, chunk, state, arm: Arm, spec, original: str, params: Mapping[str, Any],
                 public_of, trace=None):
        self._chunk, self._state, self._arm, self._spec = chunk, state, arm, spec
        self._public_of, self._trace = public_of, trace
        self.instruction = original
        self.min_act_steps = int(params["min_act_steps"])
        self.horizon = spec.horizon
        self.chunk_len = spec.chunk_len
        self.memory: dict = {}
        self._llm = build_operator(params.get("llm"))
        self.acts, self.prompts, self.stopped = 0, [], None

    # --- read-only state --------------------------------------------------------------------
    @property
    def steps(self) -> int:
        return self._state["steps"]

    @property
    def remaining(self) -> int:
        return self.horizon - self._state["steps"]

    @property
    def ended(self) -> bool:
        return self._state["ended"]

    def observe(self):
        """Public observation (the policy's public keys only)."""
        return self._public_of(self._state["observation"])

    def image(self, key: str):
        return self.observe().values[key]

    # --- actions ----------------------------------------------------------------------------
    def act(self, prompt: Optional[str] = None, steps: Optional[int] = None) -> int:
        """Run whole policy decisions until at least ``steps`` steps executed (default: one
        decision) or the episode ends. Returns the steps executed. In the null arm the prompt and
        step cap are ignored (native decisions, original instruction)."""
        if self._state["ended"]:
            return 0
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise MethodContractError("act() prompt must be a nonempty string or None")
        if steps is not None and (type(steps) is not int or steps < 1):
            raise MethodContractError("act() steps must be a positive integer or None")
        candidate = self._arm is Arm.CANDIDATE
        self.acts += 1
        self.prompts.append(prompt)
        if self._trace is not None:
            self._trace({"event": "controller_act", "act": self.acts, "steps_before": self.steps,
                         "prompt": prompt, "requested_steps": steps, "applied": candidate})
        target = steps if steps is not None else 1
        done = 0
        while done < target and not self._state["ended"]:
            cap = None
            if candidate and steps is not None:
                cap = max(self.min_act_steps, target - done)
                cap = cap if cap < self.chunk_len else None
            done += self._chunk(prompt if candidate else None, cap)
            if steps is None:
                break
        return done

    def llm(self, messages, *, max_tokens: Optional[int] = None) -> str:
        """One budgeted LLM call. Raises LLMBudgetExhausted when the declared budget is spent."""
        if self._llm is None:
            raise MethodContractError("This controller declared no llm params")
        text = self._llm(messages, max_tokens=max_tokens)
        if self._trace is not None:
            self._trace({"event": "controller_llm", "steps": self.steps, **self._llm.log[-1]})
        return text

    @staticmethod
    def image_data_url(array) -> str:
        """PNG data URL for an HxW or HxWx3 uint8 image (for vision-capable models)."""
        import base64
        return "data:image/png;base64," + base64.b64encode(png_bytes(array)).decode()

    # --- accounting -------------------------------------------------------------------------
    def summary(self) -> dict:
        return {"acts": self.acts, "distinct_prompts": sorted({p for p in self.prompts if p is not None}),
                "stopped": self.stopped, "llm": None if self._llm is None else self._llm.summary()}


def run_controller(controller, tools: ControllerTools) -> None:
    """Run a controller; a spent LLM budget ends control (the episode then finishes natively)."""
    try:
        controller.run(tools)
        tools.stopped = tools.stopped or "returned"
    except LLMBudgetExhausted:
        tools.stopped = "llm_budget"


def png_bytes(array) -> bytes:
    """Minimal PNG encoder (standard library + NumPy): grayscale HxW or RGB HxWx3, uint8."""
    import struct
    import zlib
    import numpy as np
    a = np.asarray(array)
    while a.ndim > 3 or (a.ndim == 3 and a.shape[-1] not in (1, 3) and a.shape[0] == 1):
        a = a[0]
    a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3):
        raise ValueError(f"Cannot encode an image of shape {a.shape}")
    color = 2 if a.ndim == 3 else 0
    height, width = a.shape[:2]
    raw = b"".join(b"\x00" + a[row].tobytes() for row in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
