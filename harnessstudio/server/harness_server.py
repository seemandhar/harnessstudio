"""The harness server (spec 3.1): a vla-eval ModelServer that sits between the benchmark and the policy.

To the benchmark it is an ordinary policy server. Per episode it opens its own connection to the
policy server, builds an EpisodeContext and runs the method as a task bridged to the benchmark's
observation/action loop. The benchmark's task payload and results stay with the framework.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from anyio.abc import TaskGroup

import vla_eval
from vla_eval.connection import Connection
from vla_eval.model_servers.base import ModelServer, SessionContext
from vla_eval.specs import DimSpec

from .. import __version__
from ..components.decision_loop import DecisionLoop
from ..context.bridge import Bridge
from ..context.episode import CaseKey, Embodiment, EpisodeContext
from ..context.policy import PolicyHandle, PolicyProfile
from ..context.trace import write_record
from ..contracts import Contract
from ..core.arms import Arm
from ..core.errors import ContractError, RegistrationError
from ..core.method import MethodConfig, validate

log = logging.getLogger(__name__)


@dataclass
class _Episode:
    case: CaseKey
    ctx: EpisodeContext
    bridge: Bridge
    controller: Any
    native: bool
    started: bool = False
    t0: float = field(default_factory=time.monotonic)


class HarnessServer(ModelServer):
    def __init__(self, method: MethodConfig, *, policy_url: str, contract: Contract, arm: Arm | str,
                 profile: PolicyProfile, out_dir: Optional[str | Path] = None, namespace: str = "hs",
                 policy_timeout: float = 600.0, scripted_llm: bool = False) -> None:
        self.method = validate(method)
        self.scripted_llm = scripted_llm
        missing = set(method.required) - set(profile.capabilities)
        if missing:
            raise RegistrationError(f"Method {method.name} needs capabilities {sorted(missing)} "
                                    f"that this policy does not have")
        self.policy_url, self.contract, self.arm = policy_url, contract, Arm(arm)
        self.profile, self.namespace, self.policy_timeout = profile, namespace, policy_timeout
        self.out_dir = Path(out_dir) if out_dir else None
        self.records: list[dict] = []
        self._episodes: dict[str, _Episode] = {}
        self._probe: dict = {}
        self._tg: Optional[TaskGroup] = None

    # -- lifecycle --------------------------------------------------------------
    async def on_serve_start(self, tg: TaskGroup) -> None:
        self._tg = tg
        conn = Connection(self.policy_url, timeout=60.0, max_retries=8, backoff_base=1.5)
        await conn.connect()
        self._probe = dict(conn.server_info)
        await conn.close()
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "manifest.json").write_text(json.dumps(self.manifest(), indent=2, default=str))

    def manifest(self) -> dict:
        return {"harnessstudio": __version__, "vla_eval": vla_eval.__version__, "arm": self.arm.value,
                "method": self.method.to_json(), "method_fingerprint": self.method.fingerprint(),
                "contract": asdict(self.contract), "profile": {**asdict(self.profile),
                                                               "capabilities": sorted(self.profile.capabilities)},
                "policy_url": self.policy_url, "policy_hello": self._probe, "namespace": self.namespace}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {k: DimSpec.from_dict(v) for k, v in (self._probe.get("action_spec") or {}).items()}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {k: DimSpec.from_dict(v) for k, v in (self._probe.get("observation_spec") or {}).items()}

    def get_observation_params(self) -> dict[str, Any]:
        return dict(self._probe.get("observation_params") or {})

    # -- episodes ------------------------------------------------------------------
    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        if config.get("mode") == "live":
            raise RuntimeError("HarnessStudio v2.0 supports sync mode only")
        leftover = self._episodes.pop(ctx.session_id, None)
        if leftover is not None:  # the previous episode never got EPISODE_END (spec 16.5)
            await self._finalize(leftover, None, "no_episode_end")
        task = config.get("task") or {}
        case = CaseKey(self.contract.name, str(task.get("name", "")), int(task.get("episode_idx", 0)))
        policy = PolicyHandle(self.policy_url, self.profile, self.policy_timeout)
        await policy.open(config)
        bridge = Bridge()
        shadow = self.arm is Arm.NULL and self.method.controller is not None and self.method.null == "shadow"
        embodiment = Embodiment.from_action_spec(self._probe.get("action_spec") or {}, self.contract.control_mode,
                                                 self.contract.action_dim, self.contract.eef_position)
        ectx = EpisodeContext(case=case, contract=self.contract, horizon=self.contract.horizon(task), arm=self.arm,
                              namespace=self.namespace, policy=policy, budgets=self.method.budgets, bridge=bridge,
                              decision_loop=DecisionLoop(self.method, embodiment), shadow=shadow,
                              embodiment=embodiment)
        if shadow:
            ectx._shadow_policy = PolicyHandle(self.policy_url, self.profile, self.policy_timeout)
            await ectx._shadow_policy.open(config)
        controller = self.method.controller.build() if self.method.controller else None
        if self.method.operators:
            from ..tools.llm import LLMTool, build_operators
            scripted = None
            if self.scripted_llm:
                scripted = getattr(controller, "TIER1_LLM_RESPONSES", None)
                if not scripted:
                    raise RegistrationError("Methods with operators must define TIER1_LLM_RESPONSES for tier 1")
            ectx.tools["llm"] = LLMTool(ectx, build_operators(self.method.operators, scripted),
                                        self.method.budgets.llm_calls, self.method.budgets.max_tokens)
        self._episodes[ctx.session_id] = _Episode(case, ectx, bridge, controller, native=self.arm is Arm.BASELINE)

    async def on_observation(self, obs: dict[str, Any], ctx: SessionContext) -> None:
        ep = self._episodes.get(ctx.session_id)
        if ep is None:
            raise RuntimeError("Observation before EPISODE_START")
        ectx = ep.ctx
        try:
            ectx._set_observation(obs)
            if ep.native:
                action = await ectx._native_action()
            else:
                if not ep.started:
                    ep.started = True
                    assert self._tg is not None, "on_serve_start did not run"
                    ep.bridge.start(self._tg, lambda: self._run_method(ep))
                    action = await ep.bridge.first()
                else:
                    action = await ep.bridge.next()
                if action is None:
                    if ep.bridge.error is not None:
                        raise ep.bridge.error
                    ep.native = True  # method finished, stopped or ran out of budget: finish natively
                    ectx.accounting.termination = ep.bridge.termination or "method_returned"
                    action = await ectx._native_action()
        except Exception as exc:
            self._episodes.pop(ctx.session_id, None)
            await self._finalize(ep, None, f"{type(exc).__name__}: {exc}")
            raise
        action = np.asarray(action)
        ectx._note_sent(action)
        await ctx.send_action({"actions": action})

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        ep = self._episodes.pop(ctx.session_id, None)
        if ep is not None:
            await self._finalize(ep, result or None, None if result else "empty_result")

    async def _run_method(self, ep: _Episode) -> None:
        ectx = ep.ctx
        started = time.monotonic()
        try:
            if ep.controller is None:
                await ectx._loop.run(ectx)
            elif self.arm is Arm.NULL and self.method.null == "custom":
                await ep.controller.run_null(ectx)
            else:
                await ep.controller.run(ectx)
        finally:
            ectx.accounting.method_seconds += time.monotonic() - started

    async def _finalize(self, ep: _Episode, result: Optional[Mapping[str, Any]], invalid: Optional[str]) -> None:
        ep.bridge.cancel()
        ectx = ep.ctx
        if ep.bridge.error is not None and invalid is None:
            invalid = f"method_error: {type(ep.bridge.error).__name__}: {ep.bridge.error}"
        acc = ectx.accounting
        if acc.termination == "running":
            acc.termination = "native" if self.arm is Arm.BASELINE else (ep.bridge.termination or "environment")
        acc.policy_seconds = ectx._policy.seconds + (ectx._shadow_policy.seconds if ectx._shadow_policy else 0.0)
        await ectx._policy.close(result)
        if ectx._shadow_policy is not None:
            await ectx._shadow_policy.close(result)
        metrics = dict((result or {}).get("metrics") or {})
        record = {
            "benchmark": ep.case.benchmark, "task": ep.case.task, "episode_idx": ep.case.episode_idx,
            "method": self.method.name, "method_fingerprint": self.method.fingerprint(), "arm": self.arm.value,
            "valid": invalid is None, "invalid_reason": invalid, "metrics": metrics,
            "success": bool(metrics.get("success", False)) if invalid is None else None,
            "accounting": asdict(acc), "action_sha256": ectx._action_hashes, "events": ectx._events,
            "wall_seconds": time.monotonic() - ep.t0,
        }
        self.records.append(record)
        if self.out_dir:
            write_record(self.out_dir, record)
