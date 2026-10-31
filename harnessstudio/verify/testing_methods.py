"""Small methods used by tier 1 and the test suite. They also serve as minimal examples of each kind
of method: slot-filling (derived null), controller (shadow null), budgets, and failure handling."""
from __future__ import annotations

import numpy as np

from ..core.method import Budgets, Fidelity, MethodConfig, ModuleRef

_T1 = frozenset({"act", "chunk", "seeded_sampling", "isolated_streams"})


class Pool:
    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples


class PickLast:
    def select(self, chunks, extras, obs) -> int:
        return len(chunks) - 1


class CutAfter:
    """Executor that plans the whole chunk but stops after `n` steps (per-step)."""
    min_k = 1
    per_step = True

    def __init__(self, n: int = 2) -> None:
        self.n = n

    def plan(self, chunk, extras, obs) -> int:
        return len(chunk)

    def continue_after(self, executed, obs) -> bool:
        return executed < self.n

    def record_executed(self, executed_actions) -> None:
        pass


class ZeroThenDecide:
    """Controller: checks it only sees public keys, runs two zero actions, then lets the policy act."""

    def __init__(self, action_dim: int = 4) -> None:
        self.action_dim = action_dim

    async def run(self, ctx) -> None:
        obs = ctx.observe()
        if "secret_goal" in obs:
            raise AssertionError("private key reached the method")
        await ctx.policy.sample(2)  # paid exploration call
        await ctx.execute(np.zeros((2, self.action_dim), dtype=np.float32))
        while not ctx.ended:
            await ctx.decide()


class Crash:
    async def run(self, ctx) -> None:
        await ctx.decide()
        raise ValueError("deliberate method failure")


def pick_last_k4() -> MethodConfig:
    return MethodConfig(name="test-pick-last-k4", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        required=_T1, slots={"proposer": ModuleRef(f"{__name__}:Pool", {"num_samples": 4}),
                                             "selector": ModuleRef(f"{__name__}:PickLast")})


def cut_after_2() -> MethodConfig:
    return MethodConfig(name="test-cut-after-2", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        required=frozenset({"act", "chunk"}),
                        slots={"executor": ModuleRef(f"{__name__}:CutAfter", {"n": 2})})


def zero_controller() -> MethodConfig:
    return MethodConfig(name="test-zero-controller", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        required=_T1, controller=ModuleRef(f"{__name__}:ZeroThenDecide"), null="shadow",
                        budgets=Budgets(policy_calls=1000))


def crash_controller() -> MethodConfig:
    return MethodConfig(name="test-crash", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        required=frozenset({"act"}), controller=ModuleRef(f"{__name__}:Crash"), null="shadow",
                        budgets=Budgets(policy_calls=10))


def budget_3() -> MethodConfig:
    return MethodConfig(name="test-budget-3", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        required=_T1, slots={"proposer": ModuleRef(f"{__name__}:Pool", {"num_samples": 3}),
                                             "selector": ModuleRef(f"{__name__}:PickLast")},
                        budgets=Budgets(policy_calls=3))
