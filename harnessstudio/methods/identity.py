"""identity: native seeded GR00T through the slot pipeline with no slots (spec §7)."""
from ..core.capabilities import caps
from ..core.method import Fidelity, MethodConfig, ModuleRef


class PassThroughExecutor:
    """Executes the full native chunk one step at a time. Used only by the tier-2 per-step check
    (identity with stepping_override='per_step'); never cuts."""
    identity = {"module": "pass_through"}

    def __init__(self, min_k=16, per_step=True):
        self.min_k, self.per_step = min_k, per_step

    def plan(self, chunk, extras, obs):
        return len(chunk.actions)

    def continue_after(self, executed, obs):
        return True

    def record_executed(self, executed_actions):
        pass


def per_step_variant(config: MethodConfig, chunk_len: int = 16) -> MethodConfig:
    import dataclasses
    return dataclasses.replace(config, slots={"executor": ModuleRef(
        "harnessstudio.methods.identity:PassThroughExecutor", {"min_k": chunk_len, "per_step": True})},
        description=config.description + " [per-step stepping override for tier-2 check 4]")


def config() -> MethodConfig:
    return MethodConfig(name="identity", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        substitutions=(), required=caps("action_chunk", "seeded_sampling"), slots={},
                        description="Native policy; candidate and null equal the seeded baseline.")
