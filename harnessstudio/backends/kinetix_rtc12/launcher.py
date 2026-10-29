"""Parent-side launcher for kinetix-flow runs (HarnessStudio interpreter; never imports JAX).

Freezes the method closure into the run folder, then runs the worker in the authors' pinned env with the
frozen copy first on PYTHONPATH (frozen-source rule). On GPU it takes or checks a lease in the shared
registry (HARNESSSTUDIO_DELEGATED_LEASE is honoured), pins deterministic XLA flags (delta-review B1),
charges the GPU ledger and releases its own lease after the worker exits. Only the variables the worker
needs reach it; every *_API_KEY is stripped.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from ... import config
from ...policies import kinetix_flow as KF

GPU_XLA_FLAGS = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"   # default (unchanged)
ALLOWED_GPU_XLA_FLAGS = (GPU_XLA_FLAGS, "--xla_gpu_autotune_level=0")             # review-a5-tier2plan T2(a)


def gpu_xla_flags() -> str:
    """Effective GPU XLA flags for the kinetix child. HARNESSSTUDIO_KINETIX_XLA_FLAGS may select only one of
    ALLOWED_GPU_XLA_FLAGS; anything else is refused (before any lease is taken)."""
    value = os.environ.get("HARNESSSTUDIO_KINETIX_XLA_FLAGS")
    if value is None:
        return GPU_XLA_FLAGS
    if value not in ALLOWED_GPU_XLA_FLAGS:
        raise ValueError(f"HARNESSSTUDIO_KINETIX_XLA_FLAGS={value!r} is not allowed; use one of {ALLOWED_GPU_XLA_FLAGS}")
    return value


def _child_env(device: str, gpu, frozen_root: Path, xla_flags=None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["PYTHONPATH"] = os.pathsep.join([str(frozen_root), str(frozen_root / "plugins")])
    env["WANDB_MODE"] = "disabled"
    env["PYTHONUNBUFFERED"] = "1"
    env["HARNESSSTUDIO_KINETIX_ROOT"] = str(KF.kinetix_root())
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["JAX_PLATFORMS"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        env["XLA_FLAGS"] = xla_flags if xla_flags is not None else gpu_xla_flags()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return env


def launch(method: str, *, arms, points, rollouts: int, out: Path, device: str = "gpu", gpu=None,
           level_universe=KF.LEVELS, seed: int = 0, ledger=None, max_hours: float = 1.0) -> dict:
    """One case per point; each case is the whole level universe, level-batched (repair 1)."""
    from ...core import registry
    from ...core.manifest import freeze, method_manifest
    from ...core.protocol import require_method, require_protocol
    method_config, origin = registry.get(method)
    require_method(method_config, KF)
    for p in points:
        require_protocol(KF, *(int(x) for x in str(p).split(":")))
    if device not in ("cpu", "gpu") or (device == "gpu" and gpu is None):
        raise ValueError("device must be 'cpu' or 'gpu' with a gpu id")
    # GPU: allowlisted flags, resolved before any lease (T2a). CPU: the child inherits the parent's XLA_FLAGS
    # (pre-existing behaviour); recorded either way (T2b).
    xla_flags = gpu_xla_flags() if device == "gpu" else os.environ.get("XLA_FLAGS")
    # P8: no matmul-precision pin; refuse a silently inherited non-default value (reviewed allowlist is empty).
    precision = os.environ.get("JAX_DEFAULT_MATMUL_PRECISION")
    if precision is not None:
        raise ValueError(f"JAX_DEFAULT_MATMUL_PRECISION={precision!r} is not allowed (JAX default only)")
    child = Path(KF.CHILD_PYTHON)
    if not child.is_file():
        raise ValueError(f"kinetix child interpreter not found: {child}")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    mm = method_manifest(method_config, origin, policy=KF.NAME)
    frozen = freeze(mm, out / "frozen_closure")
    (out / "launch.json").write_text(json.dumps({"method_manifest_sha256": mm["sha256"], "frozen": frozen["root"],
                                                 "device": device, "gpu": gpu, "xla_flags": xla_flags,
                                                 "jax_default_matmul_precision": "default (JAX_DEFAULT_MATMUL_PRECISION unset)"},
                                                indent=1))
    lease_owner, own_lease, led = None, False, None
    if device == "gpu":
        from ...bench import resources
        from ...eval.ledger import Ledger
        from ...eval.runs import DEFAULT_LEDGER
        led = Ledger(ledger or DEFAULT_LEDGER)
        led.require(max_hours)
        lease_owner = config.delegated_lease()
        if lease_owner is None:
            lease_owner = "harnessstudio.kinetix." + uuid.uuid4().hex
            resources.acquire(config.leases_dir(), lease_owner, [int(gpu)])
            own_lease = True
    argv = [str(child), "-m", "harnessstudio.backends.kinetix_rtc12.worker", "--method", method,
            "--arms", *arms, "--points", *[str(p) for p in points], "--level-universe", *level_universe, "--rollouts", str(rollouts), "--seed", str(seed),
            "--out", str(out / "run")]
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=str(out), env=_child_env(device, gpu, Path(frozen["root"]), xla_flags if device == "gpu" else None),
                              capture_output=True, text=True, timeout=int(max_hours * 3600) + 600)
    finally:
        if led is not None and (out / "run").exists():
            led.add(out / "run")
        if own_lease:
            from ...bench import resources
            resources.release(config.leases_dir(), lease_owner)
    (out / "stdout.log").write_text(proc.stdout)
    (out / "stderr.log").write_text(proc.stderr)
    return {"out": str(out), "run": str(out / "run"), "argv": argv, "returncode": proc.returncode, "xla_flags": xla_flags,
            "wall_seconds": time.time() - started, "lease_owner": lease_owner, "stderr_tail": proc.stderr[-800:]}


def _manifest_value(run, key):
    path = Path(run) / "manifest.json"
    return json.loads(path.read_text()).get(key) if path.is_file() else None


def _results(run: Path, arm: str) -> list:
    return [json.loads(p.read_text()) for p in sorted((run / "results" / arm).glob("*/result.json"))]


def run_tier2_kinetix(method: str, *, gpu=None, ledger=None, device: str = "gpu", rollouts=None, root=None) -> dict:
    """Tier 2 on kinetix-flow: baseline, null and candidate on TIER2_CASE; null must be M for every rollout."""
    case = KF.TIER2_CASE
    root = Path(root or config.runs_dir() / "verify" / method / ("tier2-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
    point = f"{case['inference_delay']}:{case['execute_horizon']}"
    res = launch(method, arms=["baseline", "null", "candidate"], points=[point], level_universe=case["universe"],
                 rollouts=rollouts or case["rollouts"], out=root, device=device, gpu=gpu, ledger=ledger)
    checks = [{"name": "worker_completed", "passed": res["returncode"] == 0,
               "detail": dict({k: res[k] for k in ("run", "returncode", "wall_seconds", "xla_flags")},
                              jax_default_matmul_precision=_manifest_value(res["run"], "jax_default_matmul_precision"),
                              null_mode=_manifest_value(res["run"], "null_mode"))}]
    if res["returncode"] != 0:
        return {"method": method, "tier": 2, "policy": KF.NAME, "passed": False, "checks": checks,
                "root": str(root), "error": res["stderr_tail"]}
    run = Path(res["run"])
    attribution = json.loads((run / "attribution.json").read_text())
    mrx = {cid: v.get("null") for cid, v in attribution.items()}
    all_m = all(v and v["X"] == 0 and v["R"] == 0 for v in mrx.values())
    checks.append({"name": "null_vs_baseline_MRX_per_rollout", "passed": all_m, "detail": mrx})
    base, null, cand = (_results(run, a) for a in ("baseline", "null", "candidate"))
    from ...core import registry
    has_sampler = "sampler" in registry.get(method)[0].slots
    if has_sampler:
        ok = all(r["guidance_vjp_steps"] > 0 and r["guidance_vjp_steps"] == 5 * r["decisions"] for r in null + cand)
        checks.append({"name": "null_pays_guidance_materialized", "passed": ok,
                       "detail": {"null": [r["guidance_vjp_steps"] for r in null],
                                  "candidate": [r["guidance_vjp_steps"] for r in cand]}})
    overhead = {arm: [{k: r.get(k) for k in ("policy_calls", "policy_seconds", "decisions", "solved_count", "rollouts_per_level")}
                      for r in rs] for arm, rs in (("baseline", base), ("null", null), ("candidate", cand))}
    checks.append({"name": "overhead_measured", "passed": True, "detail": overhead})
    return {"method": method, "tier": 2, "policy": KF.NAME, "case": case, "device": device, "root": str(root),
            "integration_only": True, "passed": all(c["passed"] for c in checks), "checks": checks}
