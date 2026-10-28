"""Tier-2 checks on one GPU (spec §9.2). Integration evidence only, never a quality result.

Orchestrates backend runs (rh's GR00T backend) on one clean case, classifies
null vs baseline with the M/R/X rule, and runs a native-equality probe for methods with a
server-side Sampler. Every run is added to the GPU ledger (user cap 6.0 GPU-hours).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .. import config
from ..eval.runs import backend, default_table

LEDGER = config.ledger_path()
# Each policy module declares TIER2_CASE (one clean case) and its canonical table.
CASE = {"group": "atomic_seen", "task": "CloseToasterOvenDoor", "trial": 1}  # GR00T default
PER_RUN_HOURS = 0.1


def _check(name, passed, detail):
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_arm(method, arm, out, gpu, *, baseline_run=None, stepping=None, ledger=None, policy="gr00t-n1.5") -> dict:
    from ..eval.ledger import Ledger
    ledger = Ledger(ledger or LEDGER)
    ledger.require(PER_RUN_HOURS)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = out.parent / f"{out.name}.config.json"
    cfg.write_text(json.dumps({"schema": "harnessstudio-run/1", "method": method, "arm": arm,
                               "baseline_run": str(baseline_run) if baseline_run else None,
                               "stepping_override": stepping}))
    pa, script = backend(policy)
    case = pa.TIER2_CASE
    argv = [sys.executable, str(script), "--policy", pa.NAME, "--benchmark", pa.BENCHMARK,
            "--harness", "harnessstudio", "--harness-config", str(cfg), "--canonical-table", str(default_table(policy)),
            "--groups", case["group"], "--tasks", case["task"], "--trials", str(case["trial"]),
            "--gpus", str(gpu), "--max-hours", "0.5", "--output", str(out)]
    proc = subprocess.run(argv, cwd=config.workspace(), capture_output=True, text=True, timeout=3600)
    if out.exists():
        ledger.add(out)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    case_id = f"{case['task']}--{case['trial']:04d}"
    result_path = out / "results" / case_id / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    return {"out": str(out), "returncode": proc.returncode, "status": summary.get("status"),
            "stdout_tail": proc.stdout[-600:], "stderr_tail": proc.stderr[-600:], "result": result}


def _arm_ok(run) -> bool:
    r = run["result"]
    return run["returncode"] == 0 and r.get("status") == "complete" and r.get("validity") == "valid"


def _classify_with_repair(parent, make_other, label, out_root) -> tuple:
    """M passes; R triggers one fresh re-run that must be M; X fails (spec §9.2.5)."""
    from ..eval.attribution import classify_runs
    other = make_other(out_root / f"{label}")
    if not _arm_ok(other):
        return False, {"run": other, "reason": "arm did not complete validly"}
    report = classify_runs(parent, other["out"], out_root / f"{label}-attribution")
    if report["X"] == 0 and report["R"] == 0:
        return True, {"class": "M", "report": report["per_case"]}
    if report["X"] == 0:
        again = make_other(out_root / f"{label}-repair")
        rep2 = classify_runs(parent, again["out"], out_root / f"{label}-repair-attribution") if _arm_ok(again) else None
        ok = bool(rep2) and rep2["M"] == rep2["cases"]
        return ok, {"class": "R then " + ("M" if ok else "not M"), "first": report["per_case"],
                    "repair": rep2 and rep2["per_case"]}
    return False, {"class": "X", "report": report["per_case"]}


def _native_probe(baseline_out, gpu, out_json) -> dict:
    """Acquire a lease, then run the GR00T-env probe comparing native get_action with NoOp,
    Clone (replacement splice) and VelocityReadout samplers on the saved initial observation."""
    from ..bench import resources
    from ..bench.registry import child_environment
    from ..bench.runner_util import topology_and_idle
    manifest = json.loads((Path(baseline_out) / "manifest.json").read_text())
    inventory, _ = topology_and_idle([gpu], manifest["policybench"])
    owner = "benchmark.harnessstudio.native-probe." + uuid.uuid4().hex
    lease_dir = config.leases_dir()
    resources.acquire(lease_dir, owner, [gpu], inventory=inventory)
    try:
        env = child_environment(manifest["policybench"], gpu=gpu)
        python = str(Path(manifest["policybench"]) / "envs/gr00t-n15/bin/python")
        proc = subprocess.run([python, "-m", "harnessstudio.verify.tier2", "--native-probe",
                               "--baseline-run", str(baseline_out), "--out", str(out_json)],
                              cwd=config.workspace(), env=env, capture_output=True, text=True, timeout=1800)
    finally:
        resources.release(lease_dir, owner)
    if proc.returncode != 0 or not Path(out_json).is_file():
        return {"passed": False, "detail": proc.stderr[-1200:]}
    return json.loads(Path(out_json).read_text())


def _native_probe_main(baseline_run, out_json) -> int:
    """Runs INSIDE the GR00T env on the leased GPU."""
    import numpy as np
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    from ..methods.chunktrust_ahs.modules import VelocityReadout
    from ..policies import gr00t_n15 as g
    manifest = json.loads((Path(baseline_run) / "manifest.json").read_text())
    case = manifest["cases"][0]["id"]
    packet = np.load(Path(baseline_run) / "results" / case / "initial_public_reset.npz")
    obs = {k: packet[k] for k in packet.files}
    data = DATA_CONFIG_MAP["panda_omron"]
    policy = Gr00tPolicy(model_path=str(manifest["checkpoint_path"]), modality_config=data.modality_config(),
                         modality_transform=data.transform(), embodiment_tag="new_embodiment",
                         denoising_steps=g.DENOISING_STEPS)
    policy.model.eval()
    policy.model.requires_grad_(False)

    class Clone:
        identity = {"module": "clone"}
        def on_step(self, step, x_t, v_t, ctx):
            return v_t.clone()
        def extras(self):
            return {}
    seed = 108
    native = g.seeded_sample(policy, obs, seed, samplers=[])
    variants = {"noop": g.seeded_sample(policy, obs, seed, samplers=[g.NoOpSampler()]),
                "clone_splice": g.seeded_sample(policy, obs, seed, samplers=[Clone()], effective=True),
                "velocity_readout": g.seeded_sample(policy, obs, seed, samplers=[VelocityReadout()], effective=False),
                "repeat_native": g.seeded_sample(policy, obs, seed, samplers=[])}
    result = {k: all(np.array_equal(native[a], v[a]) for a in native) for k, v in variants.items()}
    Path(out_json).write_text(json.dumps({"passed": all(result.values()), "detail": result, "seed": seed}))
    return 0


def run_tier2(method: str, policy: str, *, gpu=None, ledger=None) -> dict:
    from ..core import registry
    from ..core.runtime import method_steps_per_step
    from ..policies import adapter
    pa = adapter(policy)
    if gpu is None:
        raise ValueError("tier 2 needs --gpu (a measured-free GPU id)")
    method_config, _ = registry.get(method)
    root = config.runs_dir() / "verify" / method / ("tier2-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    checks = []
    base = run_arm(method, "baseline", root / "baseline", gpu, ledger=ledger, policy=policy)
    checks.append(_check("baseline_arm", _arm_ok(base), {k: base[k] for k in ("out", "status", "returncode")}))
    if not _arm_ok(base):
        return {"method": method, "tier": 2, "passed": False, "checks": checks, "root": str(root),
                "error": base["stderr_tail"] or base["stdout_tail"]}
    ok, detail = _classify_with_repair(base["out"], lambda out: run_arm(method, "null", out, gpu,
                                       baseline_run=base["out"], ledger=ledger, policy=policy), "null", root)
    checks.append(_check("null_vs_baseline_MRX", ok, detail))
    cand = run_arm(method, "candidate", root / "candidate", gpu, ledger=ledger, policy=policy)
    checks.append(_check("candidate_arm", _arm_ok(cand), {k: cand[k] for k in ("out", "status", "returncode")}))
    if "sampler" in method_config.slots:
        probe = _native_probe(base["out"], gpu, root / "native_probe.json")  # GR00T-only (sampler needs denoise_step)
        checks.append(_check("native_equality_and_clone_splice", probe.get("passed"), probe.get("detail")))
    if method_steps_per_step(method_config):
        chunk = run_arm("identity", "baseline", root / "identity-chunk", gpu, ledger=ledger, policy=policy)
        ok, detail = (_classify_with_repair(chunk["out"], lambda out: run_arm(
            "identity", "baseline", out, gpu, stepping="per_step", ledger=ledger, policy=policy), "identity-per-step", root)
            if _arm_ok(chunk) else (False, "identity chunk baseline failed"))
        checks.append(_check("per_step_equals_chunk_stepping", ok, detail))
    overhead = {}
    for label, run in (("baseline", base), ("candidate", cand)):
        r = run["result"]
        overhead[label] = {k: r.get(k) for k in ("policy_calls", "policy_seconds", "wall_seconds", "steps", "success")}
    if overhead["baseline"]["policy_calls"]:
        overhead["policy_call_ratio"] = (overhead["candidate"]["policy_calls"] or 0) / overhead["baseline"]["policy_calls"]
    checks.append(_check("overhead_measured", True, overhead))
    from ..eval.ledger import Ledger
    return {"method": method, "tier": 2, "policy": pa.NAME, "case": pa.TIER2_CASE, "root": str(root), "integration_only": True,
            "ledger_gpu_hours": round(Ledger(ledger or LEDGER).used(), 3),
            "passed": all(c["passed"] for c in checks), "checks": checks}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--native-probe", action="store_true")
    p.add_argument("--baseline-run")
    p.add_argument("--out")
    a = p.parse_args()
    if a.native_probe:
        raise SystemExit(_native_probe_main(a.baseline_run, a.out))
    raise SystemExit("internal entry point; use `rh verify METHOD --tier 2`")
