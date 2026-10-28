"""Worker-side episode loop (spec §5.1). Only the evaluator steps the simulator."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from . import runtime as rt
from .arms import Arm, executed_k, honor_cut, instruction_used, num_samples
from .budget import max_policy_calls
from .errors import MethodContractError
from .method import MethodConfig
from .noise import NoiseSchedule


@dataclass(frozen=True)
class StepResult:
    observation: Mapping[str, Any]
    success: bool
    executed: int
    terminated: bool


class EnvPort(Protocol):
    def execute(self, native_actions: Any, start: int, count: int) -> StepResult: ...


@dataclass(frozen=True)
class EpisodeSpec:
    case_id: str
    horizon: int
    chunk_len: int
    action_dim: int
    control_hz: float
    benchmark: str
    public_keys: tuple
    noise_namespace: str
    instruction_of: Callable[[Mapping[str, Any]], str]
    with_instruction: Optional[Callable[[Mapping[str, Any], str], Mapping[str, Any]]]
    input_hashes: Callable[[Mapping[str, Any]], dict]
    action_hashes: Callable[[Any], Any]
    method_manifest_sha256: str = ""
    action_layout: Mapping[str, Any] = field(default_factory=dict)  # action key -> (start, end) in a flat row
    proprio_eef: Optional[tuple] = None  # (public key, (start, end) or None): end-effector position


def _extras_sha256(extras: dict) -> str:
    from harnessstudio.runtime.contracts import public_hash  # array-aware (dtype, shape, bytes)
    return public_hash(extras)


def run_episode(method: MethodConfig, arm: Arm, spec: EpisodeSpec, initial_observation, env: EnvPort,
                client, policy_spec, *, trace: Callable[[dict], None],
                baseline_core_traces: Optional[Sequence[dict]] = None,
                before_decision: Optional[Callable[[Mapping[str, Any]], None]] = None) -> dict:
    """Run one episode of one arm. Raises on infrastructure or contract failures; the caller
    classifies (spec §10). Returns the arm-level record (benchmark fields added by the backend)."""
    from harnessstudio.runtime import Budget, PublicObservation, RuntimeContract
    from harnessstudio.runtime.parity import ParityGuard

    modules = {} if arm is Arm.BASELINE else rt.build_worker_modules(method)
    proposer, selector = modules.get("proposer"), modules.get("selector")
    executor, episode_op = modules.get("executor"), modules.get("episode_op")
    # The controller runs in the null arm too (shadow mode: same LLM calls, native actions).
    controller = None
    if arm is not Arm.BASELINE and "controller" in method.slots:
        controller = method.slots["controller"].build()
    k = proposer.num_samples if proposer is not None else 1
    if executor is not None and (executor.min_k != method.slots["executor"].params["min_k"]
                                 or executor.per_step != method.slots["executor"].params["per_step"]):
        raise MethodContractError("Executor attributes differ from its declared params")
    declared_min_k = method.slots["executor"].params["min_k"] if "executor" in method.slots else spec.chunk_len
    if declared_min_k > spec.chunk_len:
        raise MethodContractError("Executor min_k exceeds the policy chunk length")
    per_step = rt.method_steps_per_step(method)
    samples = num_samples(arm, k)
    arm_min_k = declared_min_k if arm is Arm.CANDIDATE else spec.chunk_len
    if arm is Arm.CANDIDATE and "controller" in method.slots:
        arm_min_k = min(arm_min_k, method.slots["controller"].params["min_act_steps"])
    budget = Budget(samples_per_decision=samples,
                    max_policy_calls=max_policy_calls(num_samples=samples, min_k=arm_min_k, horizon=spec.horizon))
    contract = RuntimeContract(spec.benchmark, spec.horizon, spec.control_hz, spec.chunk_len,
                               spec.action_dim, tuple(spec.public_keys))
    port = rt.StudioPolicyPort(client, policy_spec, arm=arm, noise=NoiseSchedule(spec.noise_namespace, spec.case_id),
                               samples=samples, want=rt.wanted_extras(method, arm))
    adapter = rt.UnifiedSelectorAdapter(selector, port) if selector is not None else None
    if adapter is not None:
        adapter.policy_info = {"action_layout": dict(spec.action_layout), "proprio_eef": spec.proprio_eef}
    guard = None
    if arm is Arm.NULL and baseline_core_traces is not None:
        guard = ParityGuard([r for r in baseline_core_traces if "event" not in r])
    core: list = []
    runtime = rt.studio_harness_class()(port, contract, budget=budget, selector=adapter,
                                        trace_sink=core.append, parity_guard=guard)

    original = spec.instruction_of(initial_observation)
    rewrite = None
    if episode_op is not None:
        rewrite = episode_op.start_episode(original, PublicObservation.from_mapping(initial_observation, spec.public_keys))
        if not isinstance(rewrite, str) or not rewrite.strip():
            raise MethodContractError("EpisodeOp must return a nonempty instruction")
    used = instruction_used(arm, original, rewrite if rewrite is not None else original)
    if used != original and spec.with_instruction is None:
        raise MethodContractError("This adapter cannot change the instruction")
    runtime.start_episode(used, arm=arm.value)

    record = {"arm": arm.value, "method": method.name, "method_version": method.version,
              "samples_per_decision": samples, "declared_min_k": declared_min_k, "per_step": per_step,
              "max_policy_calls": budget.max_policy_calls, "instruction_used": used,
              "instruction_rewrite": rewrite, "steps": 0, "success": False, "decisions": 0,
              "policy_calls": 0, "policy_seconds": 0.0, "executed_k": [], "selected_k": [],
              "planned_k": [], "cuts": [], "noise_seeds": [], "termination": None}
    state = {"observation": initial_observation, "steps": 0, "decision": 0, "ended": False}

    def view_for(observation, prompt):
        text = used if prompt is None else prompt
        return observation if text == original else spec.with_instruction(observation, text)

    def chunk(prompt=None, max_steps=None):
        """Decide and execute ONE decision (one policy query and its executed segment).
        prompt / max_steps come from a controller's act() in the candidate arm only."""
        steps, decision = state["steps"], state["decision"]
        if prompt is not None and prompt != original and spec.with_instruction is None:
            raise MethodContractError("This adapter cannot change the instruction")
        raw = view_for(state["observation"], prompt)
        public = PublicObservation.from_mapping(raw, spec.public_keys)
        inputs = spec.input_hashes(raw)
        if before_decision is not None:
            before_decision(raw)  # exactly one pre-action frame per decision (RI-2 contract)
        port.begin_decision(decision, steps)
        if adapter is not None:
            adapter.bind(public)
        started = time.monotonic()
        d = runtime.propose(raw, step=steps)
        policy_seconds = time.monotonic() - started
        response = port.last_response
        module_extras = dict(response.extras, policy={"action_layout": dict(spec.action_layout),
                                                       "proprio_eef": spec.proprio_eef})
        chosen = d.trace["selected_index"]
        native = response.natives[chosen]
        planned = spec.chunk_len
        if executor is not None:
            planned = executor.plan(d.chunk, module_extras, public)
            if type(planned) is not int or not declared_min_k <= planned <= spec.chunk_len:
                raise MethodContractError(f"Executor planned k={planned!r}, outside [{declared_min_k}, {spec.chunk_len}]")
        selected = executed_k(arm, planned, spec.chunk_len)
        if max_steps is not None and arm is Arm.CANDIDATE:
            selected = min(selected, max_steps)
        effective = min(selected, spec.horizon - steps)
        trace({"event": "action_dispatch", "chunk": decision, "steps_before": steps, "input": inputs,
               "action": spec.action_hashes(native), "selected_k": selected, "execute_steps": effective,
               "noise_seed": port.last_seeds[chosen], "policy_seconds": policy_seconds,
               "selected_index": chosen, "selector_proposed_index": d.trace["selector_proposed_index"],
               "planned_k": planned, "samples": samples,
               "noise_stream": NoiseSchedule.stream_names(samples)[chosen],
               "extras_sha256": _extras_sha256(response.extras),
               "method_manifest_sha256": spec.method_manifest_sha256,
               "controller_prompt": None if prompt is None else spec.input_hashes({"prompt": prompt})["prompt"]})
        executed, cut, success, terminated = 0, None, False, False
        observation = state["observation"]
        if per_step:
            while executed < effective:
                result = env.execute(native, executed, 1)
                if result.executed != 1:
                    raise RuntimeError("Environment executed a wrong number of steps")
                executed += 1
                observation, terminated = result.observation, result.terminated
                success = success or result.success
                if terminated:
                    break
                if executor is not None and executed < effective:
                    # Called after EVERY executed step so stateful executors see each observation;
                    # a cut is recorded and honoured only once executed >= min_k (budget bound, §4.6).
                    wants = executor.continue_after(executed, PublicObservation.from_mapping(view_for(observation, prompt), spec.public_keys))
                    if wants is not True and wants is not False:
                        raise MethodContractError("continue_after must return a bool")
                    if executed >= declared_min_k:
                        if not wants and cut is None:
                            cut = executed
                        if not honor_cut(arm, wants):
                            break
        else:
            result = env.execute(native, 0, effective)
            executed, observation = result.executed, result.observation
            success, terminated = result.success, result.terminated
        if not 0 < executed <= effective:
            raise RuntimeError("Environment reported an invalid executed count")
        if executor is not None:
            executor.record_executed(d.chunk.actions[:executed])
        runtime.commit(executed)
        state["observation"] = observation
        state["steps"] = steps = steps + executed
        record["success"] = record["success"] or success
        record["executed_k"].append(executed)
        record["selected_k"].append(selected)
        record["planned_k"].append(planned)
        record["cuts"].append(cut)
        record["noise_seeds"].append(port.last_seeds[chosen])
        record["policy_seconds"] += policy_seconds
        trace({"event": "action_complete", "chunk": decision, "executed_steps": executed,
               "steps": steps, "cut_at": cut})
        state["decision"] = decision + 1
        if terminated:
            record["termination"] = "native_terminated"
            state["ended"] = True
        elif steps >= spec.horizon:
            state["ended"] = True
        honoured_cut = arm is Arm.CANDIDATE and (cut is not None or (max_steps is not None and selected < spec.chunk_len))
        if not terminated and executed != effective and not honoured_cut:
            raise RuntimeError("Short dispatch without termination")
        return executed

    if controller is not None:
        from .controller import ControllerTools, run_controller
        tools = ControllerTools(chunk=chunk, state=state, arm=arm, spec=spec, original=original,
                                params=method.slots["controller"].params, trace=trace,
                                public_of=lambda o: PublicObservation.from_mapping(o, spec.public_keys))
        run_controller(controller, tools)
        record["controller"] = tools.summary()
    while not state["ended"]:
        chunk()
    steps, decision = state["steps"], state["decision"]
    if steps <= 0 or (not record["success"] and steps != spec.horizon):
        raise RuntimeError("Incomplete episode cannot be scored as a robot failure")
    record.update(steps=steps, decisions=decision, policy_calls=runtime.policy_calls,
                  divergence=runtime.divergence, core_traces=core,
                  unified_manifest=runtime.manifest, unified_manifest_sha256=runtime.manifest_sha256)
    return record
