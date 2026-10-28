"""EpisodeOp slot adapter over harnessstudio.runtime.operators and runtime.memory.

InstructionRewrite asks a frozen language operator (Unified's NRPInstructionOperator: verified
model identity, frozen catalog, prompt and token budget, no automatic fallback, never Astra) to
rewrite the instruction once at episode start, with advice retrieved from frozen DEVELOPMENT
memory cards. The operator client is injected (``client_factory``), so this module never reads
credentials itself.

Not runnable on the milestone-1 GR00T backend: its worker environment strips *_API_KEY variables
(harnessstudio.bench.registry.child_environment) and the GR00T adapter does not yet rewrite the instruction
key. Both are deferred; the module is unit-tested with a fake client.
"""
from __future__ import annotations

import importlib

from ..runtime.contracts import PolicySpec
from ..runtime.memory import FrozenMemory, MemoryCard
from ..runtime.operators import NRPInstructionOperator


class InstructionRewrite:
    def __init__(self, client_factory: str, model: str, expected_returned_model: str, catalog_sha256: str,
                 policy_id: str, checkpoint_identity: str, memory_cards=(), max_tokens: int = 256,
                 max_calls: int = 1):
        module, _, attr = client_factory.partition(":")
        client = getattr(importlib.import_module(module), attr)()
        self.operator = NRPInstructionOperator(client, model=model, expected_returned_model=expected_returned_model,
                                               catalog_sha256=catalog_sha256, max_tokens=max_tokens)
        self.memory = FrozenMemory([MemoryCard(**card) for card in memory_cards])
        self.policy = PolicySpec(policy_id, checkpoint_identity)
        self.max_calls, self.calls, self.tokens = int(max_calls), 0, 0
        self.identity = dict(self.operator.identity, adapter="harnessstudio.episode_op",
                             memory_sha256=self.memory.sha256, max_calls=self.max_calls)

    def start_episode(self, instruction, obs) -> str:
        if self.calls >= self.max_calls:
            raise RuntimeError("Frozen operator-call budget exhausted")
        self.calls += 1
        memories = self.memory.retrieve(instruction, self.policy)
        text, tokens = self.operator.rewrite(obs, instruction, memories)
        self.tokens += tokens
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Operator returned an empty instruction")
        return text.strip()
