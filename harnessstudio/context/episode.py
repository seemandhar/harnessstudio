"""EpisodeContext: the only object a method or component receives (spec 5.1).

Everything prefixed with an underscore belongs to the framework (arm, case, raw observation, native
stream). It is an API boundary, not a sandbox: plugin code is reviewed, and tier 1 checks what
reaches the method.
"""
from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..contracts import Contract
from ..core.arms import Arm
from ..core.errors import BudgetExceeded, ContractError, MethodStopped, PolicyError
from ..core.method import Budgets
from ..core.noise import crn_seed
from ..policy_ext import protocol
from .bridge import Bridge
from .policy import PolicyHandle, Proposal
from .trace import fingerprint, jsonable


@dataclass(frozen=True)
class CaseKey:
    benchmark: str
    task: str
    episode_idx: int


@dataclass(frozen=True)
class Embodiment:
    """Public facts about the action space, from the policy's HELLO action_spec (in vector order)."""
    action_layout: Mapping[str, tuple] = field(default_factory=dict)   # name -> (start, end)
    action_dim: Optional[int] = None
    control_mode: str = "unknown"
    eef_path: tuple = ()               # keys into the public observation, e.g. ("states",)
    eef_slice: Optional[tuple] = None  # e.g. (0, 3)
    eef_delta_scale: Optional[float] = None   # metres per unit position action (analytic skills)
    gripper_open: float = -1.0
    gripper_close: float = 1.0

    @classmethod
    def from_action_spec(cls, spec: Mapping[str, Mapping[str, Any]], control_mode: str = "unknown",
                         fallback_dim: Optional[int] = None, eef: Optional[Mapping[str, Any]] = None,
                         fallback_layout: Optional[Mapping[str, Any]] = None) -> "Embodiment":
        layout, offset = {}, 0
        for name, dim in (spec or {}).items():
            n = int(dim.get("dims", 0))
            if n > 0:
                layout[name] = (offset, offset + n)
                offset += n
        if len(layout) <= 1 and fallback_layout:  # policy declares one raw block: use the contract's names
            layout = {k: (int(v[0]), int(v[1])) for k, v in fallback_layout.items()}
            offset = max(e for _, e in layout.values())
        eef = dict(eef or {})
        sl = eef.get("slice")
        return cls(dict(layout), offset or fallback_dim, control_mode, tuple(eef.get("path") or ()),
                   tuple(sl) if sl else None)

    def eef_position(self, obs: Mapping[str, Any]) -> Optional[np.ndarray]:
        """End-effector position from a public observation, or None if this benchmark does not expose it."""
        if not self.eef_path:
            return None
        value: Any = obs
        for key in self.eef_path:
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        return arr[self.eef_slice[0]:self.eef_slice[1]] if self.eef_slice else arr

    def dims(self, *names: str) -> list:
        missing = [n for n in names if n not in self.action_layout]
        if missing:
            raise KeyError(f"Action layout lacks {missing}; has {sorted(self.action_layout)}")
        return [d for n in names for d in range(*self.action_layout[n])]


@dataclass
class Accounting:
    steps: int = 0
    decisions: int = 0
    method_policy_samples: int = 0
    native_policy_samples: int = 0
    shadow_policy_samples: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    tool_calls: int = 0
    policy_seconds: float = 0.0
    method_seconds: float = 0.0
    termination: str = "running"
    extra: dict = field(default_factory=dict)


class _PolicyProxy:
    """ctx.policy: budgeted, seeded access to the frozen policy for methods."""

    def __init__(self, ctx: "EpisodeContext") -> None:
        self._ctx = ctx

    @property
    def capabilities(self) -> frozenset:
        return self._ctx._profile.capabilities

    async def act(self, obs: Optional[Mapping[str, Any]] = None, instruction: Optional[str] = None) -> np.ndarray:
        """One native-style request. Returns the chunk [H, A] (or [1, A] with server chunking)."""
        ctx = self._ctx
        ctx._charge(1)
        payload = ctx._policy_input(obs, instruction)
        if ctx._shadow:
            chunk = await ctx._shadow_handle().act(payload)
            ctx.accounting.shadow_policy_samples += 1
            return chunk
        chunk = await ctx._policy.act(payload)
        ctx._decision += 1
        ctx._log("act", input_sha256=fingerprint(payload), action_sha256=fingerprint(chunk))
        return chunk

    async def sample(self, k: int = 1, obs: Optional[Mapping[str, Any]] = None, instruction: Optional[str] = None,
                     sampler: Optional[Mapping[str, Any]] = None, readouts: Sequence[str] = ()) -> list[Proposal]:
        """K seeded samples for the current decision. Sample 0 uses the native noise stream."""
        ctx = self._ctx
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        ctx._charge(k)
        payload = ctx._policy_input(obs, instruction)
        if ctx._shadow:
            seeds = [ctx._seed(10**6 + ctx._shadow_decision, i) for i in range(k)]
            ctx._shadow_decision += 1
            out = await ctx._shadow_handle().sample(payload, seeds, sampler, readouts)
            ctx.accounting.shadow_policy_samples += k
            return out
        seeds = [ctx._seed(ctx._decision, i) for i in range(k)]
        out = await ctx._policy.sample(payload, seeds, sampler, readouts)
        ctx._decision += 1
        ctx._log("sample", decision=ctx._decision - 1, seeds=seeds, input_sha256=fingerprint(payload),
                 actions_sha256=[fingerprint(p.actions) for p in out])
        return out


class EpisodeContext:
    def __init__(self, *, case: CaseKey, contract: Contract, horizon: int, arm: Arm, namespace: str,
                 policy: PolicyHandle, budgets: Budgets, bridge: Bridge, decision_loop: Any = None,
                 shadow: bool = False, artifacts: Optional[Mapping[str, Any]] = None,
                 tools: Optional[Mapping[str, Any]] = None, embodiment: Optional[Embodiment] = None) -> None:
        self._case, self._contract, self._horizon, self._arm = case, contract, horizon, arm
        self._namespace, self._policy, self._budgets, self._bridge = namespace, policy, budgets, bridge
        self._profile = policy.profile
        self._loop = decision_loop
        self._shadow = shadow
        self._shadow_policy: Optional[PolicyHandle] = None
        self._shadow_open = None  # coroutine factory set by the server
        self._raw: dict = {}
        self._public: dict = {}
        self._decision = 0
        self._shadow_decision = 0
        self._native_queue: deque = deque()
        self._stop_reason: Optional[str] = None
        self._events: list = []
        self._action_hashes: list = []
        self._obs_hashes: list = []       # raw observation the action was computed from, per step
        self._method_spend = 0
        self.accounting = Accounting()
        self.memory: dict = {}                       # per-episode scratch
        self.artifacts = dict(artifacts or {})       # frozen method artifacts (read-only by convention)
        self.tools = dict(tools or {})
        self.policy = _PolicyProxy(self)
        self.embodiment = embodiment or Embodiment(action_dim=contract.action_dim, control_mode=contract.control_mode)

    # -- public, read-only state ------------------------------------------------
    def observe(self) -> dict:
        return copy.deepcopy(self._public)

    @property
    def instruction(self) -> str:
        text = self._public.get(self._contract.instruction_key)
        if not isinstance(text, str):
            raise ContractError("The instruction must be a string")
        return text

    @property
    def steps(self) -> int:
        return self.accounting.steps

    @property
    def remaining(self) -> int:
        return max(0, self._horizon - self.accounting.steps)

    @property
    def ended(self) -> bool:
        return self._stop_reason is not None or self.remaining == 0

    # -- public actions -------------------------------------------------------------
    async def execute(self, actions: Any) -> int:
        """Run actions one per step. Returns how many ran (fewer if the episode ends)."""
        self._check_active()
        chunk = protocol.as_chunk(actions)
        if self._contract.action_dim is not None and chunk.shape[1] != self._contract.action_dim:
            raise ContractError(f"Actions have {chunk.shape[1]} dims; the contract says {self._contract.action_dim}")
        executed = 0
        for action in chunk:
            if self.ended:
                break
            if self._shadow:
                action = await self._native_action()
            await self._emit(action)
            executed += 1
        return executed

    async def decide(self, instruction: Optional[str] = None) -> int:
        """Run the method's DecisionLoop once (one policy decision). Returns steps executed."""
        self._check_active()
        if self._shadow:
            if not self._native_queue:
                await self._fill_native()
            executed = 0
            while self._native_queue and not self.ended:
                await self._emit(self._native_queue.popleft())
                executed += 1
            return executed
        if self._loop is None:
            raise RuntimeError("This method has no DecisionLoop")
        return await self._loop.decide(self, instruction)

    def record(self, event: str, **fields: Any) -> None:
        self._log("event", name=str(event), **fields)

    def stop(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stop() needs a reason")
        self._stop_reason = reason
        self._log("stop", reason=reason)

    # -- framework internals ----------------------------------------------------
    def _check_active(self) -> None:
        if self._stop_reason is not None:
            raise MethodStopped(f"Method stopped: {self._stop_reason}")

    def _set_observation(self, raw: Mapping[str, Any]) -> None:
        self._raw = dict(raw)
        self._public = self._contract.filter_public(raw)

    def _seed(self, decision: int, sample: int) -> int:
        c = self._case
        return crn_seed(self._namespace, c.benchmark, c.task, c.episode_idx, decision, sample)

    def _charge(self, samples: int) -> None:
        self._check_active()
        cap = self._budgets.policy_calls
        if cap is not None and self._method_spend + samples > cap:
            raise BudgetExceeded(f"policy budget {cap} exhausted")
        self._method_spend += samples
        self.accounting.method_policy_samples += samples

    def _policy_input(self, obs: Optional[Mapping[str, Any]], instruction: Optional[str]) -> dict:
        """The raw observation with method overrides on public keys (spec 16.2)."""
        payload = dict(self._raw)
        if obs is not None:
            extra = set(obs) - set(self._contract.public_keys)
            if extra:
                raise ContractError(f"Methods may only override public keys, not {sorted(extra)}")
            payload.update(obs)
        if instruction is not None:
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be a non-empty string")
            payload[self._contract.instruction_key] = instruction
        return payload

    async def _native_chunk(self) -> np.ndarray:
        """One native decision at the current observation (seeded sample 0 when available)."""
        started = time.monotonic()
        if self._profile.seeded:
            seed = self._seed(self._decision, 0)
            chunk = (await self._policy.sample(self._raw, [seed]))[0].actions
        else:
            chunk = await self._policy.act(self._raw)
        self.accounting.policy_seconds += time.monotonic() - started
        self._decision += 1
        self.accounting.native_policy_samples += 1
        self._log("native", decision=self._decision - 1, input_sha256=fingerprint(self._raw),
                  action_sha256=fingerprint(chunk))
        return chunk

    async def _fill_native(self) -> None:
        chunk = await self._native_chunk()
        self._native_queue.extend(np.array(a) for a in chunk[: self._profile.native_length(chunk)])

    async def _native_action(self) -> np.ndarray:
        if not self._native_queue:
            await self._fill_native()
        return self._native_queue.popleft()

    async def _emit(self, action: np.ndarray) -> None:
        """Hand one action to the benchmark (through the bridge) and wait for the next observation."""
        action = np.asarray(action)
        if not np.isfinite(action).all():
            raise PolicyError("Refusing to execute a non-finite action")
        await self._bridge.emit(action)

    def _note_sent(self, action: np.ndarray) -> None:
        """Called by the server after an action left for the benchmark."""
        self.accounting.steps += 1
        self._action_hashes.append(fingerprint(np.asarray(action)))
        self._obs_hashes.append(fingerprint(self._raw))

    def _shadow_handle(self) -> PolicyHandle:
        if self._shadow_policy is None:
            raise RuntimeError("Shadow policy session is not open")
        return self._shadow_policy

    def _log(self, kind: str, **fields: Any) -> None:
        self._events.append({"t": kind, "step": self.accounting.steps, **jsonable(fields)})
