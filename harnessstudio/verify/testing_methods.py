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

    def __init__(self, action_dim: int = 7) -> None:
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


class LLMPlanner:
    """Controller: asks the LLM for subtasks, drives the policy with each, then finishes natively."""
    TIER1_LLM_RESPONSES = ['["move left", "move right"]']

    async def run(self, ctx) -> None:
        import json
        plan = await ctx.tools["llm"]([{"role": "user", "content": f"Split into subtasks: {ctx.instruction}"}])
        for subtask in json.loads(plan):
            executed = 0
            while executed < 3 and not ctx.ended:
                executed += await ctx.decide(instruction=subtask)
        while not ctx.ended:
            await ctx.decide()


def llm_planner() -> MethodConfig:
    from ..core.method import OperatorPin
    return MethodConfig(name="test-llm-planner", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        required=frozenset({"act", "chunk"}), controller=ModuleRef(f"{__name__}:LLMPlanner"),
                        null="shadow", budgets=Budgets(policy_calls=1000, llm_calls=2, max_tokens=64),
                        operators=(OperatorPin("planner", "test-model", "nrp", "NRP_API_KEY"),))


class SkillDemo:
    """Controller: analytic move to a nearby point, gripper close, then the policy as a skill."""

    async def run(self, ctx) -> None:
        from ..skills import eef_delta as sk
        start = ctx.embodiment.eef_position(ctx.observe())
        res = await sk.move_by(ctx, [0.3, -0.2], tolerance=0.05, max_steps=6)
        ctx.record("move", ok=res.ok, steps=res.steps, error=res.final.get("error"), start=list(map(float, start)))
        await sk.set_gripper(ctx, "close", steps=1)
        await sk.vla_act(ctx, steps=100)


def skill_demo() -> MethodConfig:
    return MethodConfig(name="test-skill-demo", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        required=frozenset({"act", "chunk"}), controller=ModuleRef(f"{__name__}:SkillDemo"),
                        null="shadow", budgets=Budgets(policy_calls=100))


class TrajectoryPreparer:
    """Keeps successful development trajectories (state, action per step)."""
    collect_arm = "baseline"
    record = ("state",)

    def build(self, records, out_dir):
        from ..components.memory import write_trajectories
        write_trajectories(records, out_dir / "trajectories.jsonl", only_success=False)
        return {"trajectories": "trajectories.jsonl"}


class NearestBlend:
    """ActionOp: blends the first action toward the nearest stored development action (RTCF-like toy)."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha, self.store = alpha, None

    def load_artifacts(self, artifacts) -> None:
        from ..components.memory import TrajectoryStore
        self.store = TrajectoryStore.load(artifacts["trajectories"])

    def transform(self, chunk, obs, memory):
        hit = self.store.nearest(obs["state"], k=1) if self.store is not None else []
        if not hit:
            return chunk
        out = np.array(chunk, dtype=np.float32)
        out[0] = (1 - self.alpha) * out[0] + self.alpha * np.asarray(hit[0]["action"], dtype=np.float32)
        return out


def memory_blend() -> MethodConfig:
    return MethodConfig(name="test-memory-blend", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        required=frozenset({"act", "chunk"}),
                        slots={"action_op": ModuleRef(f"{__name__}:NearestBlend", {"alpha": 0.5})},
                        prepare=ModuleRef(f"{__name__}:TrajectoryPreparer"))
