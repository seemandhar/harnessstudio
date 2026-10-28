"""Unified bridge (spec §6.3): StudioPolicyPort, selector adapter, StudioUnifiedHarness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .arms import Arm
from .errors import MethodContractError
from .method import SERVER_SLOTS, MethodConfig
from .noise import NoiseSchedule


@dataclass(frozen=True)
class ProposeResult:
    natives: list
    flats: list
    extras: dict
    cost: dict


class PolicyClient(Protocol):
    def get_action_seeded(self, observation, seed: int) -> tuple: ...
    def propose(self, observation, *, arm: str, decision_index: int, step_index: int,
                seeds: list, want: list) -> ProposeResult: ...


def build_worker_modules(method: MethodConfig) -> dict:
    """Fresh instances for one (episode, arm) (spec §4.7)."""
    return {name: ref.build() for name, ref in method.slots.items() if name not in SERVER_SLOTS and name != "controller"}


def method_steps_per_step(method: MethodConfig) -> bool:
    ref = method.slots.get("executor")
    return bool(ref and ref.params.get("per_step", False))


def wanted_extras(method: MethodConfig, arm: Arm) -> list:
    return [] if arm is Arm.BASELINE or "sampler" not in method.slots else ["sampler"]


class StudioPolicyPort:
    """Evaluator-side port. Holds the noise schedule; one RPC per decision."""
    def __init__(self, client: PolicyClient, spec, *, arm: Arm, noise: NoiseSchedule, samples: int, want: list):
        self.spec = spec
        self._client, self._arm, self._noise = client, arm, noise
        self._samples, self._want = samples, list(want)
        self._decision = None
        self.last_response = None
        self.last_seeds = None

    def begin_decision(self, decision_index: int, step_index: int) -> None:
        self._decision = (decision_index, step_index)
        self.last_response = None

    def infer(self, observation, instruction):
        from harnessstudio.runtime import ActionChunk
        if self._decision is None:
            raise RuntimeError("begin_decision() must precede infer()")
        d, s = self._decision
        seeds = self._noise.seeds(d, self._samples)
        values = dict(observation.values)
        if self._arm is Arm.BASELINE:
            native, flat = self._client.get_action_seeded(values, seeds[0])
            response = ProposeResult([native], [flat], {}, {"policy_calls": 1})
        else:
            response = self._client.propose(values, arm=self._arm.value, decision_index=d, step_index=s,
                                            seeds=seeds, want=self._want)
            if len(response.natives) != len(seeds) or len(response.flats) != len(seeds):
                raise RuntimeError("Policy server returned the wrong number of samples")
        self.last_response, self.last_seeds = response, seeds
        return ActionChunk(response.flats[0])

    def infer_alternative(self, observation, instruction, index):
        from harnessstudio.runtime import ActionChunk
        if self.last_response is None or not 0 < index < len(self.last_response.flats):
            raise RuntimeError("Alternative sample requested outside the current decision")
        return ActionChunk(self.last_response.flats[index])


class UnifiedSelectorAdapter:
    """Fits a HarnessStudio Selector into Unified's select(chunks, execute_steps)."""
    required_capabilities = frozenset()

    def __init__(self, selector, port: StudioPolicyPort):
        self._selector, self._port = selector, port
        self.identity = {"module": "harnessstudio.selector", "inner": getattr(selector, "identity", None)}
        self._public = None
        self.policy_info = {}

    def bind(self, public) -> None:
        self._public = public

    def select(self, chunks, execute_steps):
        extras = dict(self._port.last_response.extras, policy=self.policy_info)
        index = self._selector.select(list(chunks), extras, self._public)
        if type(index) is not int or not 0 <= index < len(chunks):
            raise MethodContractError(f"Selector returned {index!r}, not an index into {len(chunks)} samples")
        return index


_HARNESS_CLASS = None


def studio_harness_class():
    """UnifiedHarness subclass: variable execution (commit) and first-divergence guarding (§4.5)."""
    global _HARNESS_CLASS
    if _HARNESS_CLASS is not None:
        return _HARNESS_CLASS
    from harnessstudio.runtime import PublicObservation, UnifiedHarness
    from harnessstudio.runtime.parity import ParityError

    class StudioUnifiedHarness(UnifiedHarness):
        divergence = None

        def propose(self, raw_observation, *, step):
            if self.parity_guard is not None:
                observed = PublicObservation.from_mapping(raw_observation, self.contract.public_keys).sha256
                try:
                    expected = self.parity_guard.expected(step).public_input_sha256
                except ParityError:
                    expected = None
                if expected != observed:
                    self.divergence = {"step": step,
                                       "reason": "input" if expected is not None else "no_baseline_decision"}
                    self.parity_guard = None
            decision = super().propose(raw_observation, step=step)
            self._pending = (step, decision.execute_steps)
            return decision

        def commit(self, executed: int) -> None:
            step, maximum = getattr(self, "_pending", (None, None))
            if step is None or type(executed) is not int or not 0 < executed <= maximum:
                raise ValueError("commit must follow propose with 0 < executed <= native maximum")
            self._next_step = step + executed
            self._pending = (None, None)
            if self.trace_sink:
                self.trace_sink({"event": "horizon_commit", "step": step, "executed_steps": executed,
                                 "native_max_steps": maximum, "manifest_sha256": self.manifest_sha256})

    _HARNESS_CLASS = StudioUnifiedHarness
    return _HARNESS_CLASS
