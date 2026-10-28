"""Tier-1 CPU contract checks (spec §9.1) on SYNTHETIC fixtures."""
from __future__ import annotations

import dataclasses
import traceback

from ..core import registry
from ..core.arms import Arm
from ..core.capabilities import require, unified_names
from ..core.errors import LeakageError, MethodContractError
from ..core.method import ModuleRef, validate
from ..core.noise import NoiseSchedule, derive_seed
from ..core.pipeline import EpisodeSpec, run_episode
from ..core.slots import DecisionCtx
from ..policies import adapter as policy_adapter
from . import fixtures as fx

FIELDS = ("input", "action", "noise_seed", "execute_steps", "selected_k")


def _spec(case):
    return EpisodeSpec(case_id=case, horizon=64, chunk_len=fx.CHUNK, action_dim=fx.ACTION_DIM, control_hz=20.0,
                       benchmark="synthetic", public_keys=fx.PUBLIC_KEYS, noise_namespace="tier1",
                       instruction_of=fx.instruction_of, with_instruction=fx.with_instruction,
                       input_hashes=fx.input_hashes, action_hashes=fx.action_hashes,
                       action_layout=fx.ACTION_LAYOUT, proprio_eef=fx.PROPRIO_EEF)


def _policy_spec(capabilities):
    from harnessstudio.runtime import PolicySpec
    return PolicySpec("synthetic", "tier1-fixture", unified_names(capabilities))


def _run(method, arm, capabilities, *, case="c1", env=None, baseline=None):
    env = env or fx.SyntheticEnv()
    rows = []
    record = run_episode(method, arm, _spec(case), env.observation(), env,
                         fx.SyntheticClient(), _policy_spec(capabilities), trace=rows.append,
                         baseline_core_traces=baseline)
    return record, [r for r in rows if r["event"] == "action_dispatch"]


def _key(rows):
    return [[r[f] for f in FIELDS] for r in rows]


def _check(name, fn):
    try:
        return {"name": name, "passed": True, "detail": fn()}
    except Exception as exc:  # report every failure; tier 1 must not stop at the first
        return {"name": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4)}


def _with_executor(method, factory, min_k, per_step):
    slots = dict(method.slots)
    slots["executor"] = ModuleRef(factory, {"min_k": min_k, "per_step": per_step})
    slots.pop("sampler", None)
    return dataclasses.replace(method, slots=slots)


def offline(method):
    """Tier 1 never calls a network LLM: a controller's llm params become a scripted fake.
    A controller class may provide TIER1_LLM_RESPONSES (cycled); default is ["ok"]."""
    ref = method.slots.get("controller")
    if ref is None or not ref.params.get("llm") or ref.params["llm"].get("provider") == "fake":
        return method
    import importlib
    module, _, attr = ref.factory.partition(":")
    responses = list(getattr(getattr(importlib.import_module(module), attr), "TIER1_LLM_RESPONSES", ["ok"]))
    llm = {"provider": "fake", "responses": responses, "max_calls": ref.params["llm"]["max_calls"],
           "max_tokens": ref.params["llm"]["max_tokens"]}
    slots = dict(method.slots, controller=ModuleRef(ref.factory, dict(ref.params, llm=llm)))
    return dataclasses.replace(method, slots=slots)


def run_tier1(name: str, policy: str) -> dict:
    method, origin = registry.get(name)
    method = offline(validate(method))
    pa = policy_adapter(policy)
    caps = pa.CAPABILITIES
    checks = [_check("registration", lambda: validate(method).name),
              _check("capabilities", lambda: require(method.required, pa.CAPABILITIES,
                                                     method=name, policy=pa.NAME) or "ok")]

    def null_derivation():
        base, brows = _run(method, Arm.BASELINE, caps)
        null, nrows = _run(method, Arm.NULL, caps, baseline=base["core_traces"])
        if _key(brows) != _key(nrows):
            raise AssertionError("null action_dispatch rows differ from baseline on a noise-free fixture")
        expected = null["samples_per_decision"] * null["decisions"]
        if null["policy_calls"] != expected:
            raise AssertionError(f"null paid {null['policy_calls']} calls, expected {expected}")
        return {"decisions": null["decisions"], "null_policy_calls": null["policy_calls"]}
    checks.append(_check("disabled_is_identity_and_null_derivation", null_derivation))

    def leakage():
        from harnessstudio.runtime import ActionChunk, PublicObservation
        obs = fx.SyntheticEnv().observation()
        dropped = PublicObservation.from_mapping(dict(obs, reward=1.0, success=True), fx.PUBLIC_KEYS)
        if {"reward", "success"} & set(dropped.values):
            raise LeakageError("top-level forbidden keys reached the public observation")
        nested = dict(obs)
        nested["state.gripper_qpos"] = {"success": True}
        try:
            PublicObservation.from_mapping(nested, fx.PUBLIC_KEYS)
        except ValueError:
            pass
        else:
            raise LeakageError("nested forbidden key was not rejected")
        fields = {f.name for f in dataclasses.fields(DecisionCtx)}
        if fields & {"case_id", "task", "trial", "seed", "arm", "episode_seed"}:
            raise LeakageError("DecisionCtx exposes identity fields")
        # Recording proxy (§4.7): modules may read only public keys (a workflow boundary, not a sandbox).
        accessed = set()

        class Recording(dict):
            def __getitem__(self, key):
                accessed.add(key)
                return dict.__getitem__(self, key)

            def get(self, key, default=None):
                accessed.add(key)
                return dict.get(self, key, default)

        public = PublicObservation(Recording(obs), "0" * 64)
        client = fx.SyntheticClient()
        k = method.slots["proposer"].params["num_samples"] if "proposer" in method.slots else 1
        resp = client.propose(obs, arm="candidate", decision_index=0, step_index=0,
                              seeds=NoiseSchedule("tier1", "c1").seeds(0, k),
                              want=["sampler"] if "sampler" in method.slots else [])
        chunks = [ActionChunk(f) for f in resp.flats]
        mods = {n: r.build() for n, r in method.slots.items() if n != "sampler"}
        extras = dict(resp.extras, policy={"action_layout": fx.ACTION_LAYOUT, "proprio_eef": fx.PROPRIO_EEF})
        if "selector" in mods:
            mods["selector"].select(chunks, extras, public)
        if "executor" in mods:
            mods["executor"].plan(chunks[0], extras, public)
            if mods["executor"].per_step:
                mods["executor"].continue_after(1, public)
        if "episode_op" in mods:
            mods["episode_op"].start_episode(fx.INSTRUCTION, public)
        if accessed - set(fx.PUBLIC_KEYS):
            raise LeakageError(f"modules read non-public keys {sorted(accessed - set(fx.PUBLIC_KEYS))}")
        return {"keys_read": sorted(accessed)}
    checks.append(_check("leakage", leakage))

    def budget():
        k = method.slots["proposer"].params["num_samples"] if "proposer" in method.slots else 1
        out = {}
        for label, factory, min_k in (("min_k", "harnessstudio.verify.fixtures:MinKExecutor", 1),
                                      ("cut_everywhere", "harnessstudio.verify.fixtures:CutEverywhereExecutor", 1)):
            record, _ = _run(_with_executor(method, factory, min_k, True), Arm.CANDIDATE, caps)
            bound = k * 64 // min_k
            if record["policy_calls"] > bound or record["steps"] > 64:
                raise AssertionError(f"{label}: {record['policy_calls']} calls > {bound} or steps > 64")
            out[label] = record["policy_calls"]
        try:
            _run(_with_executor(method, "harnessstudio.verify.fixtures:BelowMinKExecutor", 4, False),
                 Arm.CANDIDATE, caps)
        except MethodContractError:
            out["below_min_k"] = "refused"
        else:
            raise AssertionError("an executor planning k < min_k was not refused")
        return out
    checks.append(_check("budget_bound", budget))

    def seed_rule():
        n = NoiseSchedule("tier1", "c1")
        if n.seeds(0, 8)[0] != derive_seed("tier1", "c1", "primary", 0):
            raise AssertionError("sample 0 does not use the primary stream")
        client = fx.SyntheticClient()
        obs = fx.SyntheticEnv().observation()
        one = client.propose(obs, arm="candidate", decision_index=0, step_index=0, seeds=n.seeds(0, 1), want=[])
        many = client.propose(obs, arm="candidate", decision_index=0, step_index=0, seeds=n.seeds(0, 8), want=[])
        if fx.action_hashes(one.natives[0]) != fx.action_hashes(many.natives[0]):
            raise AssertionError("extra samples changed sample 0")
        return "ok"
    checks.append(_check("seed_rule", seed_rule))

    def lifecycle():
        a = {c: _key(_run(method, Arm.CANDIDATE, caps, case=c)[1]) for c in ("c1", "c2")}
        b = {c: _key(_run(method, Arm.CANDIDATE, caps, case=c)[1]) for c in ("c2", "c1")}
        if a != b:
            raise AssertionError("per-case traces depend on case order (state carried across episodes)")
        return "ok"
    checks.append(_check("lifecycle_order_independence", lifecycle))

    def determinism():
        if _key(_run(method, Arm.CANDIDATE, caps)[1]) != _key(_run(method, Arm.CANDIDATE, caps)[1]):
            raise AssertionError("two identical runs differ")
        return "ok"
    checks.append(_check("determinism", determinism))

    if "controller" in method.slots:
        def controller_arms():
            rows = {}
            for arm in (Arm.NULL, Arm.CANDIDATE):
                trace = []
                env = fx.SyntheticEnv()
                record = run_episode(method, arm, _spec("c1"), env.observation(), env, fx.SyntheticClient(),
                                     _policy_spec(caps), trace=trace.append)
                acts = [r for r in trace if r["event"] == "controller_act"]
                dispatch = [r for r in trace if r["event"] == "action_dispatch"]
                applied = [r["controller_prompt"] for r in dispatch if r["controller_prompt"] is not None]
                if arm is Arm.NULL and (applied or any(r["selected_k"] != fx.CHUNK for r in dispatch[:-1])):
                    raise AssertionError("shadow-mode null executed a controller prompt or step cap")
                rows[arm.value] = {"acts": len(acts), "prompted_decisions": len(applied),
                                   "llm_calls": (record["controller"]["llm"] or {}).get("calls"),
                                   "stopped": record["controller"]["stopped"]}
            if rows["null"]["llm_calls"] != rows["candidate"]["llm_calls"] and rows["null"]["acts"] == rows["candidate"]["acts"]:
                raise AssertionError("null did not pay the controller's LLM calls")
            return rows
        checks.append(_check("controller_shadow_null_and_candidate", controller_arms))

    if origin.get("kind") == "plugin":
        def closure():
            import os
            import subprocess
            import tempfile
            from pathlib import Path
            from ..core.manifest import freeze, method_manifest
            m = method_manifest(method, origin, policy=pa.NAME)
            from ..eval.attribution import _default_bench
            bench = _default_bench()
            child = bench / pa.CHILD_PYTHON
            if not child.is_file():
                raise AssertionError(f"child interpreter {child} not found; cannot run the import probe")
            module = origin["ref"].split(":")[0]
            with tempfile.TemporaryDirectory() as tmp:
                frozen = freeze(m, Path(tmp))
                paths = [tmp, str(Path(tmp) / "plugins")]
                probe = subprocess.run(
                    [str(child), "-I", "-c", f"import sys; sys.path[:0]={paths!r}; import {module}"],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=120)
            if probe.returncode != 0:
                raise AssertionError("child import probe failed: " + probe.stderr[-500:])
            return {"files": len(m["closure"]), "frozen_files": len(frozen["files"]), "child": str(child)}
        checks.append(_check("plugin_closure", closure))

    return {"method": name, "policy": pa.NAME, "tier": 1, "synthetic": True,
            "passed": all(c["passed"] for c in checks), "checks": checks}
