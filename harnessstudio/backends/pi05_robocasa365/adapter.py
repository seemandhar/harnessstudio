"""HarnessStudio backend for methods around frozen Pi0.5 (openpi) on RoboCasa365 (run by `rh`).

Reuses the GR00T backend's generic pieces (config validation, method resolution, case building,
placement, freezing, runner) and adds the Pi0.5 specifics: pinned openpi checkout and released
checkpoint, server/episode interpreters and paths, Protocol-N1 seeding through openpi's Policy._rng.

Defaults (override with flags or environment):
  --policy-root   openpi-robocasa checkout   (default: $HARNESSSTUDIO_OPENPI_ROOT or WORKSPACE/../openpi-robocasa)
  --checkpoint    POLICYBENCH/models/pi05-robocasa-release/<release subdirectory>
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
STUDIO_SRC = HERE.parents[2]
GR00T = HERE.parent / "gr00t_robocasa365"
for _p in (str(STUDIO_SRC), str(GR00T), str(GR00T / "copied")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harnessstudio import config  # noqa: E402
from harnessstudio.backends.gr00t_robocasa365 import adapter as gad  # noqa: E402

NAME = "harnessstudio"
PROTOCOL = HERE / "protocol_pi05_n1.json"
RELEASE = config.DATA_DIR / "pi05_release.json"
BASELINE_IDENTITY = "pi05-n1-baseline"
RUNNER_VERSION = "harnessstudio-pi05-2"
# XLA's GPU autotuner times candidate kernels at compile time, so separate server processes could
# compile different kernels and return actions differing by ~2e-3 for the same request and seed
# (probe 2026-10-28). Autotuning off makes every process bitwise identical; paired arms need that.
SERVER_ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_PLATFORMS": "cuda",
              "XLA_FLAGS": "--xla_gpu_autotune_level=0"}


def parser():
    p = gad.parser()
    p.add_argument("--policy-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    return p


def runtime_paths(args, bench):
    from harnessstudio.policies import pi05 as p
    root = getattr(args, "policy_root", None) or os.environ.get("HARNESSSTUDIO_OPENPI_ROOT") \
        or config.workspace().parent / "openpi-robocasa"
    root = Path(root).resolve()
    release = json.loads(RELEASE.read_text())
    checkpoint = Path(getattr(args, "checkpoint", None)
                      or bench / "models/pi05-robocasa-release" / release["checkpoint_subdirectory"]).resolve()
    server_python = root / ".venv-rmh-pi05/bin/python"
    episode_python = bench / "envs/gr00t-n15/bin/python"
    sim_deps = bench / "envs/pi05-sim-deps"
    for path in (root / "src/openpi/policies/robocasa_policy.py", server_python, episode_python):
        if not path.is_file():
            raise ValueError("Missing pinned Pi0.5 runtime: " + str(path))
    if not sim_deps.is_dir():
        raise ValueError("Missing Pi0.5 simulator client dependencies: " + str(sim_deps))
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != p.OPENPI_COMMIT:
        raise ValueError("openpi-robocasa checkout must be at pinned commit " + p.OPENPI_COMMIT)
    return root, checkpoint, server_python, episode_python, sim_deps


def checkpoint_inventory(checkpoint, *, rehash=False):
    from harnessstudio.bench.common import read_json, sha256, stat_fingerprint, write_json
    release = json.loads(RELEASE.read_text())
    names = release["files"]
    observed = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file() and ".cache" not in p.parts}
    if observed != set(names):
        raise ValueError("Pi0.5 checkpoint file set differs from the pinned release")
    cache = config.workspace() / ".state/benchmark-pi05-checkpoints.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    with cache.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read_json(cache) if cache.exists() else {}
        for name, expected in sorted(names.items()):
            path = checkpoint / name
            stamp = stat_fingerprint(path)
            old = previous.get(str(path))
            digest = old["sha256"] if not rehash and old and old["stat"] == stamp else sha256(path)
            if stamp["size"] != expected["size"] or digest != expected["sha256"]:
                raise ValueError("Pi0.5 checkpoint differs from the pinned release: " + str(path))
            result[str(path)] = {"sha256": digest, "stat": stamp}
        previous.update(result)
        write_json(cache, previous)
    return result, release["revision"]


def task_registry(bench, episode_python, paths):
    from harnessstudio.bench.registry import child_environment
    env = child_environment(bench)
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in paths] + [env["PYTHONPATH"]])
    out = subprocess.run([str(episode_python), str(HERE / "worker.py"), "--mode", "registry"], env=env,
                         capture_output=True, text=True, timeout=180)
    if out.returncode:
        raise RuntimeError("Cannot inspect the RoboCasa task registry: " + out.stderr[-2000:])
    rows = [line.split(":", 1)[1] for line in out.stdout.splitlines() if line.startswith("RMH_REGISTRY:")]
    if len(rows) != 1:
        raise RuntimeError("Pi0.5 backend returned no unique task registry")
    return json.loads(rows[0])


def prepare(args, name):
    from harnessstudio.bench import inventory as bcli
    from harnessstudio.bench.common import json_hash, now, read_json, write_json
    from harnessstudio.bench.registry import canonical_pair, csv_words
    from harnessstudio.core.manifest import freeze, method_manifest
    from harnessstudio.core.runtime import method_steps_per_step
    from harnessstudio.policies import pi05 as p
    import runner
    policy, benchmark = canonical_pair(args.policy, args.benchmark)
    if policy != "pi0.5" or benchmark != "robocasa365":
        raise ValueError("The HarnessStudio Pi0.5 backend supports only Pi0.5 / RoboCasa365")
    if args.harness_config is None:
        raise ValueError("--harness-config is required (schema harnessstudio-run/1)")
    if args.canonical_table not in (None, "none"):
        raise ValueError("the Pi0.5 backend has no canonical-table pre-action check; pass --canonical-table none")
    config_path = args.harness_config.resolve()
    hconfig = gad.validate_config(read_json(config_path))
    method, origin = gad.resolve_method(hconfig, p)
    mm = method_manifest(method, origin, policy=policy)
    per_step = method_steps_per_step(method)
    bench = (args.policybench or config.policybench()).resolve()
    root, checkpoint, server_python, episode_python, sim_deps = runtime_paths(args, bench)
    episode_paths = [sim_deps, root / "packages/openpi-client/src", bench / "benchmarks/robocasa365",
                     bench / "benchmarks/robosuite", bench / "scripts"]
    server_paths = [root / "src", root / "packages/openpi-client/src", bench / "benchmarks/robocasa365",
                    bench / "benchmarks/robosuite"]
    registry = task_registry(bench, episode_python, episode_paths)
    groups, cases = gad.build_cases(registry, ["all"] if args.all_groups else csv_words(args.groups),
                                    csv_words(args.tasks), args.num_tasks, args.episodes_per_task, args.trials)
    protocol = json.loads(PROTOCOL.read_text())
    gpu_request = args.gpus or "0"
    hours = args.max_hours if args.max_hours is not None else 1
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("max-hours must be finite and positive")
    files, revision = checkpoint_inventory(checkpoint, rehash=args.rehash_checkpoint)
    checkpoint_identity = json_hash({str(Path(k).relative_to(checkpoint)): v["sha256"] for k, v in files.items()})
    identity = gad.harness_identity(method, hconfig, BASELINE_IDENTITY)
    plan = {"policy": policy, "benchmark": benchmark, "harness": identity, "harness_name": name,
            "harness_interface": "agent_adapter", "harness_config": hconfig, "harness_config_path": str(config_path),
            "harness_config_sha256": gad.canonical_sha256(hconfig), "harness_identity": identity, "arm": hconfig["arm"],
            "baseline_run": hconfig["baseline_run"], "method_manifest": mm, "stepping": "per_step" if per_step else "chunk",
            "mode": "small", "policybench": str(bench), "groups": groups, "cases": cases, "expected_episodes": len(cases),
            "physical_step_upper_bound": sum(c["horizon"] for c in cases),
            "native_policy_call_upper_bound": sum(gad.arm_call_bound(method, hconfig["arm"], c["horizon"], chunk=5) for c in cases),
            "gpu_request": gpu_request, "split": args.split or "pretrain", "videos": True,
            "record_arrays": bool(args.record_arrays), "max_hours": hours, "output": str(args.output.resolve()),
            "official_trials_per_task": 50, "official_sequential_reset_equivalent": False,
            "reset": "ADAPTED: env.reset(seed=7+50*within_group_task_index+trial) for every trial (paired arms)",
            "noise": "ADAPTED: Protocol N1 via openpi Policy._rng = jax.random.key(seed % 2**32), sample 0 shared by all arms",
            "runner_version": RUNNER_VERSION, "protocol": protocol, "protocol_path": str(PROTOCOL),
            "canonical_table_path": None, "canonical_table_sha256": None, "canonical_table": None,
            "openpi_root": str(root), "openpi_commit": p.OPENPI_COMMIT, "checkpoint_path": str(checkpoint),
            "checkpoint_files": files, "checkpoint_identity": checkpoint_identity, "checkpoint_revision": revision,
            "backend": {"worker_rel": "harnessstudio/backends/pi05_robocasa365/worker.py",
                        "server_python": str(server_python), "episode_python": str(episode_python),
                        "server_paths": [str(x) for x in server_paths], "episode_paths": [str(x) for x in episode_paths],
                        "server_env": dict(SERVER_ENV)}}
    plan["protocol_sha256"] = json_hash({"protocol": protocol, "checkpoint_identity": checkpoint_identity,
                                         "stepping": plan["stepping"], "backend": plan["runner_version"]})
    if hconfig["arm"] == "null":
        gad.check_baseline_run(hconfig, method, cases, plan["protocol_sha256"])
    if gpu_request == "available":
        from harnessstudio.bench import resources
        ids = [g["index"] for g in resources.snapshot() if g.get("processes_known") and not g["processes"]
               and g["used_mib"] <= 2048]
        if not ids:
            raise ValueError("No unoccupied GPU available")
    else:
        ids = bcli.parse_gpu_ids(gpu_request)
    ids = ids[:len(cases)]
    slots = runner.plan_slots(ids, args.workers_per_gpu, len(cases))
    if args.dry_run:
        placement = runner.placement_map({k: cases[i::len(slots)] for i, k in enumerate(slots)})
        return dict(plan, method_manifest={"sha256": mm["sha256"], "method": mm["method"]}, placement=placement,
                    dry_run=True, gpu_execution=False, checkpoint_files="<omitted>")
    ids = sorted({int(k.split(".")[0]) for k in slots})
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Output exists. Choose a new folder, or --resume the frozen run.")
    assignments = {key: cases[i::len(slots)] for i, key in enumerate(slots)}
    placement = runner.placement_map(assignments)
    output.mkdir(parents=True, exist_ok=False)
    try:
        frozen = freeze(mm, output / "frozen_closure")
        sources, repos = bcli.source_inventory(bench, [output / "frozen_closure", config_path, PROTOCOL, RELEASE])
        plan.update(schema_version=1, created_at=now(), deadline=time.time() + hours * 3600, gpu_ids=ids,
                    workers_per_gpu=args.workers_per_gpu, calibrated_worker_mib=args.calibrated_worker_mib,
                    assignments=assignments, placement=placement, placement_sha256=gad.canonical_sha256(placement),
                    source_sha256=sources, repositories=repos, frozen_closure=frozen,
                    lease_owner="benchmark.harnessstudio.pi05." + uuid.uuid4().hex)
        plan["manifest_digest"] = json_hash(plan)
        write_json(output / "manifest.json", plan)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return plan


def build_harness():
    return gad.StudioAdapter(NAME, policies=("pi0.5",), prepare_fn=prepare)


if __name__ == "__main__":
    raise SystemExit(gad.StudioAdapter(NAME, policies=("pi0.5",), prepare_fn=prepare).run_cli(sys.argv[1:]))
