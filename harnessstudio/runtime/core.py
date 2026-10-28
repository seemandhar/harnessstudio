"""Bounded Sense → Context → Propose → Select runtime, independent of evaluation."""
from dataclasses import asdict, dataclass
import time

from .contracts import Budget, PublicObservation, canonical_hash, public_hash
from .memory import FrozenMemory
from .modules import IdentitySelector


@dataclass(frozen=True)
class Decision:
    chunk: object
    execute_steps: int
    trace: dict


class UnifiedHarness:
    def __init__(self, policy, contract, *, budget=None, selector=None, monitor=None,
                 memory=None, operator=None, trace_sink=None, parity_guard=None):
        self.policy, self.contract = policy, contract
        self.budget = budget or Budget()
        self.selector = selector or IdentitySelector()
        self.monitor, self.operator = monitor, operator
        self.memory = memory or FrozenMemory()
        self.trace_sink = trace_sink
        self.parity_guard = parity_guard
        self._started = False
        if self.budget.samples_per_decision > 1 and (
                not {"independent_sampling", "isolated_sampling"} <= policy.spec.capabilities
                or not callable(getattr(policy, "infer_alternative", None))):
            raise ValueError("Extra proposals require isolated RNG streams and an alternative-sampling port")
        for module in (self.selector, monitor, operator):
            if module and not module.required_capabilities <= policy.spec.capabilities:
                raise ValueError("Module requires unavailable policy/observation capabilities")
        if monitor and monitor.key not in contract.public_keys:
            raise ValueError("Monitor key is outside public observation contract")
        if operator and (self.budget.max_operator_calls < 1 or self.budget.max_operator_tokens < 1):
            raise ValueError("Explicit operator call/token budgets are required")
        self.manifest = {"unified_version": "1.1.0", "policy": {
            **asdict(policy.spec), "capabilities": sorted(policy.spec.capabilities)},
            "contract": asdict(contract), "budget": asdict(self.budget),
            "selector": self.selector.identity, "monitor": getattr(monitor, "identity", None),
            "operator": getattr(operator, "identity", None), "memory_sha256": self.memory.sha256,
            "parity_reference_sha256": canonical_hash([asdict(r) for r in parity_guard.decisions]) if parity_guard else None,
            "training_steps": 0}
        self.manifest_sha256 = canonical_hash(self.manifest)

    def start_episode(self, instruction, *, arm="candidate"):
        if arm not in {"baseline", "null", "candidate"} or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Explicit arm and nonempty public instruction required")
        if self.parity_guard and arm!='null':
            raise ValueError('Native parity guard is only for an action-identical null arm')
        # The evaluator owns native policy reset/start_episode; never reset a
        # simulator or policy RNG as a side effect of wrapper construction.
        self.instruction, self.original_instruction, self.arm = instruction, instruction, arm
        self.policy_calls = self.operator_calls = self.operator_tokens = 0
        self._next_step = 0
        self._started = True
        if self.monitor: self.monitor.reset()

    def propose(self, raw_observation, *, step):
        if not self._started or type(step) is not int or step != self._next_step:
            raise ValueError("Decisions must follow the native execution cadence")
        if step >= self.contract.horizon:
            raise ValueError("Official policy horizon exhausted")
        start = time.perf_counter()
        observation = PublicObservation.from_mapping(raw_observation, self.contract.public_keys)
        execute_steps = min(self.contract.native_chunk_steps, self.contract.horizon-step)
        sample_count = 1 if self.arm == "baseline" else self.budget.samples_per_decision
        if self.policy_calls + sample_count > self.budget.max_policy_calls:
            raise RuntimeError("Frozen policy-call budget exhausted")
        if self.parity_guard:
            self.parity_guard.before_inference(step,observation.sha256,
                canonical_hash(self.instruction),execute_steps)
        memories = self.memory.retrieve(self.original_instruction, self.policy.spec)
        monitor_signal = self.monitor.observe(observation) if self.monitor else None
        if self.operator and step == 0 and self.arm != "baseline":
            if self.operator_calls >= self.budget.max_operator_calls:
                raise RuntimeError("Frozen operator-call budget exhausted")
            self.operator_calls += 1
            proposed, tokens = self.operator.rewrite(
                PublicObservation.from_mapping(observation.values, self.contract.public_keys),
                self.original_instruction, memories)
            if type(tokens) is not int or tokens < 0:
                raise ValueError("Operator must account actual token use")
            self.operator_tokens += tokens
            if self.operator_tokens > self.budget.max_operator_tokens:
                raise RuntimeError("Frozen operator-token budget exhausted")
            if not isinstance(proposed, str) or not proposed.strip():
                raise ValueError("Invalid instruction proposal")
            if self.arm == "candidate": self.instruction = proposed
        chunks = []
        inference_start = time.perf_counter()
        for sample in range(sample_count):
            self.policy_calls += 1
            # Native preprocessing may mutate arrays/lists. Every proposal gets
            # the same original public data, with no cross-proposal aliasing.
            proposal_observation = PublicObservation.from_mapping(
                observation.values, self.contract.public_keys)
            chunk = (self.policy.infer(proposal_observation, self.instruction) if sample == 0 else
                     self.policy.infer_alternative(proposal_observation, self.instruction, sample))
            chunk.validate(self.contract.action_dim, execute_steps)
            chunks.append(chunk)
        inference_seconds = time.perf_counter()-inference_start
        selected = self.selector.select(chunks, execute_steps)
        if type(selected) is not int or not 0 <= selected < len(chunks):
            raise ValueError("Selector must return an existing proposal index")
        # Null executes proposal zero after performing candidate selection work.
        # Alternative samples must not advance the primary policy RNG/queue.
        # This capability declaration still needs native/null trace verification.
        chosen = selected if self.arm == "candidate" else 0
        if self.parity_guard:
            self.parity_guard.before_action(step,public_hash(chunks[chosen].actions))
        self._next_step = step + execute_steps
        trace = {"manifest_sha256": self.manifest_sha256, "step": step, "arm": self.arm,
                 "public_input_sha256": observation.sha256,
                 "instruction_sha256": canonical_hash(self.instruction),
                 "proposal_action_sha256": [public_hash(c.actions) for c in chunks],
                 "selected_index": chosen, "selector_proposed_index": selected,
                 "execute_steps": execute_steps, "policy_calls": sample_count,
                 "operator_calls_total": self.operator_calls, "operator_tokens_total": self.operator_tokens,
                 "memory_ids": [c.id for c in memories], "monitor": monitor_signal,
                 "inference_seconds": inference_seconds, "total_seconds": time.perf_counter()-start,
                 "null_action_parity": "requires measured adapter/native trace audit" if self.arm == "null" else None}
        if self.trace_sink: self.trace_sink(trace)
        return Decision(chunks[chosen], execute_steps, trace)
