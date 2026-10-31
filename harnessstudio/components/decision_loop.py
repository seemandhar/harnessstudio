"""The default controller (spec 5.3): one decision runs

    ObservationOp -> InstructionOp -> Gate -> Proposer (+ Sampler in the policy server)
    -> Selector -> ActionOp -> Executor, with Monitor checked after every executed step.

Derived-null rules (spec 8.1) are applied here, from the framework-private arm. In the null arm every
component still runs (so the null pays the same compute), but the policy sees the original
observation and instruction, sample 0 executes for its native length, and cuts are ignored.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ..core import arms
from ..core.arms import Arm
from ..core.method import MethodConfig


class DecisionLoop:
    def __init__(self, config: MethodConfig) -> None:
        build = lambda slot: config.slots[slot].build() if slot in config.slots else None  # noqa: E731
        self.observation_op = build("observation_op")
        self.instruction_op = build("instruction_op")
        self.gate = build("gate")
        self.proposer = build("proposer")
        self.selector = build("selector")
        self.action_op = build("action_op")
        self.executor = build("executor")
        self.monitor = build("monitor")
        self.sampler_ref = config.slots["sampler"].to_json() if "sampler" in config.slots else None
        self.num_samples = int(getattr(self.proposer, "num_samples", 1)) if self.proposer else 1
        self.history: list[dict] = []

    async def run(self, ctx) -> None:
        while not ctx.ended:
            await ctx.decide()

    async def decide(self, ctx, instruction: Optional[str] = None) -> int:
        arm: Arm = ctx._arm
        candidate = arm is Arm.CANDIDATE
        obs = ctx.observe()

        obs_override = self.observation_op.transform(obs) if self.observation_op else None
        text = instruction or ctx.instruction
        if self.instruction_op:
            text = self.instruction_op.rewrite(text, obs, ctx.memory)
        k = self.num_samples
        if self.gate:
            k = int(self.gate.plan(obs, self.history).get("num_samples", k))
        k = arms.num_samples(arm, k)

        policy_obs = obs_override if candidate else None
        policy_text = arms.instruction_used(arm, ctx.instruction, text)
        if policy_text == ctx.instruction:
            policy_text = None  # no override: the policy gets the raw observation unchanged
        if ctx._profile.seeded:
            proposals = await ctx.policy.sample(
                k, obs=policy_obs, instruction=policy_text,
                sampler=self.sampler_ref if arms.sampler_effective(arm) else None)
        else:
            if k > 1:
                raise RuntimeError("Multiple samples need a seeded policy")
            from ..context.policy import Proposal
            proposals = [Proposal.make(await ctx.policy.act(obs=policy_obs, instruction=policy_text))]

        chosen = 0
        if self.selector and len(proposals) > 1:
            chosen = int(self.selector.select([p.actions for p in proposals], [p.extras for p in proposals], obs))
            if not 0 <= chosen < len(proposals):
                raise ValueError(f"Selector returned index {chosen} for {len(proposals)} proposals")
        index = arms.executed_index(arm, chosen)
        native = proposals[index].actions
        chunk = native
        if self.action_op:
            transformed = np.asarray(self.action_op.transform(np.array(native), obs, ctx.memory))
            if candidate:
                chunk = transformed

        native_k = ctx._profile.native_length(native)
        k_exec = native_k
        if self.executor:
            planned = int(self.executor.plan(np.array(chunk), dict(proposals[index].extras), obs))
            if candidate:
                k_exec = max(1, min(planned, len(chunk)))
        executed = 0
        for i in range(k_exec):
            if ctx.ended:
                break
            await ctx._emit(chunk[i])
            executed += 1
            after = ctx.observe()
            wants = True
            if self.executor and getattr(self.executor, "per_step", False) and executed < k_exec:
                wants = bool(self.executor.continue_after(executed, after))
            if self.monitor and self.monitor.check(after, self.history) == "interrupt":
                wants = False
            if not arms.honor_cut(arm, wants):
                break
        if self.executor:
            self.executor.record_executed(np.array(chunk[:executed]))
        self.history.append({"decision": len(self.history), "samples": len(proposals), "chosen": chosen,
                             "executed": executed})
        ctx.accounting.decisions += 1
        return executed
