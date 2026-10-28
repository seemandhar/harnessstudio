"""HarnessStudio backend for methods around frozen GR00T N1.5 on RoboCasa365 (run by `rh`).

Generalized from ChunkTrust runner v2.2 adapter.py. Registered name (harness_catalog.json):
`harnessstudio`. Harness config (--harness-config JSON):

  {"schema": "harnessstudio-run/1", "method": NAME, "arm": "baseline"|"null"|"candidate",
   "baseline_run": PATH|null, "stepping_override": null|"per_step"}

Why agent_adapter, not policy_factory: the built-in GR00T worker runs the policy in a separate
server whose sampler internals are invisible to a client plugin and enforces execute == 16 and
unseeded resets. This adapter owns its worker/scheduler but keeps the benchmark loop, official
horizons, native success, leases, manifest freezing, sealing and summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent


STUDIO_SRC = HERE.parents[2]  # the directory that contains the harnessstudio package
for _p in (str(STUDIO_SRC), str(HERE), str(HERE / "copied")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harnessstudio import config  # noqa: E402

NAME = "harnessstudio"
DEFAULT_PROTOCOL = HERE / "copied" / "protocol_v2_n1.json"
PROTOCOL_KEYS = {"schema", "runner_version", "observation_stride", "action_chunk", "control_hz",
                 "denoising_steps", "reset", "noise", "parity", "records"}
CONFIG_KEYS = {"schema", "method", "arm", "baseline_run", "stepping_override"}
BASELINE_IDENTITY = "gr00t-n15-n1-baseline"


def parser():
    p = argparse.ArgumentParser(description="HarnessStudio GR00T N1.5 RoboCasa365 evaluation")
    p.add_argument("--policy", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--harness", required=True)
    p.add_argument("--harness-config", type=Path)
    p.add_argument("--harness-source", type=Path, action="append", default=[])
    p.add_argument("--mode", choices=("small",), default="small")
    p.add_argument("--groups", "--task-set", nargs="+")
    p.add_argument("--all", dest="all_groups", action="store_true")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--num-tasks", type=int)
    trials = p.add_mutually_exclusive_group()
    trials.add_argument("--episodes-per-task", "--episodes", type=int)
    trials.add_argument("--trials", nargs="+", help="Explicit official trial IDs, e.g. 2 3")
    p.add_argument("--gpus", "--gpu")
    p.add_argument("--output", type=Path)
    p.add_argument("--protocol", type=Path, help="Common protocol JSON (default copied/protocol_v2_n1.json)")
    p.add_argument("--canonical-table", help="Canonical reset table JSON, or 'none' (non-paired probes only)")
    p.add_argument("--evaluator-step-log", action="store_true")
    p.add_argument("--require-placement-sha256")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--calibrated-worker-mib", type=int)
    p.add_argument("--policybench", type=Path)
    p.add_argument("--split", choices=("pretrain", "target"))
    p.add_argument("--max-hours", type=float)
    p.add_argument("--videos", action="store_true", default=None)
    p.add_argument("--no-videos", dest="videos", action="store_false")
    p.add_argument("--record-arrays", action="store_true", default=None)
    p.add_argument("--rehash-checkpoint", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--status", action="store_true")
    return p


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(config) -> dict:
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("harness config must have exactly keys " + ", ".join(sorted(CONFIG_KEYS)))
    if config["schema"] != "harnessstudio-run/1" or config["arm"] not in ("baseline", "null", "candidate"):
        raise ValueError("unknown harness-config schema or arm")
    if (config["arm"] == "null") != (config["baseline_run"] is not None):
        raise ValueError("baseline_run is required for, and only for, the null arm")
    if config["stepping_override"] not in (None, "per_step"):
        raise ValueError("stepping_override must be null or 'per_step'")
    if config["stepping_override"] and config["method"] != "identity":
        raise ValueError("stepping_override is only allowed for the identity method (tier-2 check 4)")
    return config


def resolve_method(config, policy_module=None):
    from harnessstudio.core import registry
    from harnessstudio.core.capabilities import require
    from harnessstudio.methods.identity import per_step_variant
    from harnessstudio.policies import gr00t_n15
    g = policy_module or gr00t_n15
    method, origin = registry.get(config["method"])
    if config["stepping_override"] == "per_step":
        method = per_step_variant(method, g.CHUNK_LEN)
    require(method.required, g.CAPABILITIES, method=method.name, policy=g.NAME)
    return method, origin


def harness_identity(method, config, baseline=None) -> str:
    if config["arm"] == "baseline":
        return (baseline or BASELINE_IDENTITY) + ("-perstep" if config["stepping_override"] else "")
    return f"{method.name}@{method.version}/{config['arm']}"


def build_cases(task_registry, groups, tasks, num_tasks, episodes_per_task, trials):
    from harnessstudio.bench.registry import choose_cases
    groups, base = choose_cases(task_registry, mode="small", groups=groups, tasks=tasks,
                                num_tasks=num_tasks, episodes_per_task=1, seeded=True)
    if trials is None:
        count = 1 if episodes_per_task is None else episodes_per_task
        if count < 1:
            raise ValueError("episodes-per-task must be positive")
        trials = list(range(count))
    trials = [int(t) for t in trials]
    if not trials or len(trials) != len(set(trials)) or min(trials) < 0 or max(trials) >= 50:
        raise ValueError("trials must be unique official IDs in [0, 50)")
    cases = []
    for row in base:
        for trial in trials:
            cases.append({"task": row["task"], "horizon": row["horizon"], "group": row["group"],
                          "task_index": row["task_index"], "trial": trial,
                          "id": f"{row['task']}--{trial:04d}",
                          "seed": 7 + 50 * row["task_index"] + trial})
    if len(cases) > 32:
        raise ValueError("Small mode is capped at 32 episodes")
    return groups, cases


def load_protocol(path):
    from harnessstudio.bench.common import read_json
    path = Path(path or DEFAULT_PROTOCOL).resolve()
    protocol = read_json(path)
    if set(protocol) != PROTOCOL_KEYS or protocol["schema"] != "chunktrust-protocol/2":
        raise ValueError("protocol must declare exactly " + ", ".join(sorted(PROTOCOL_KEYS)))
    if (protocol["observation_stride"] != 16 or protocol["action_chunk"] != 16 or protocol["control_hz"] != 20
            or protocol["denoising_steps"] != 4):
        raise ValueError("protocol contract constants differ from the frozen GR00T contract")
    noise, par = protocol["noise"], protocol["parity"]
    if noise.get("crn") is not True or not noise.get("namespace"):
        raise ValueError("HarnessStudio requires Protocol N1 (crn true) with a namespace")
    if par.get("strict_initial_observation_hash") is not True or par.get("max_rerenders") != 2 \
            or par.get("fresh_process_retries") != 1:
        raise ValueError("protocol parity must be strict with 2 re-renders and 1 fresh-process retry")
    return path, protocol


def load_table(value):
    import parity
    from harnessstudio.bench.common import sha256
    if value is None:
        raise ValueError("--canonical-table PATH is required (or 'none' for non-paired probes)")
    if value == "none":
        return None, None, None
    path = Path(value).resolve()
    return path, parity.load_table(str(path)), sha256(path)


def check_baseline_run(config, method, cases, protocol_sha):
    from harnessstudio.bench.common import read_json
    base = read_json(Path(config["baseline_run"]) / "manifest.json")
    bc = base.get("harness_config", {})
    if bc.get("arm") != "baseline" or bc.get("method") != method.name or bc.get("stepping_override"):
        raise ValueError("baseline_run must be a chunk-stepping baseline run of the same method")
    if [c["id"] for c in base["cases"]] != [c["id"] for c in cases]:
        raise ValueError("baseline_run case list differs")
    if base.get("protocol_sha256") != protocol_sha:
        raise ValueError("baseline_run protocol differs")
    return base


def arm_call_bound(method, arm, horizon, chunk=16):
    from harnessstudio.core.budget import max_policy_calls
    k = method.slots["proposer"].params["num_samples"] if "proposer" in method.slots else 1
    min_k = min(method.slots["executor"].params["min_k"], chunk) if "executor" in method.slots else chunk
    if "controller" in method.slots:
        min_k = min(min_k, method.slots["controller"].params["min_act_steps"])
    samples = 1 if arm == "baseline" else k
    return max_policy_calls(num_samples=samples, min_k=(min_k if arm == "candidate" else chunk), horizon=horizon)


def prepare(args, name):
    from harnessstudio.bench import inventory as bcli
    from harnessstudio.bench.common import json_hash, now, read_json, write_json
    from harnessstudio.bench.registry import (DEFAULT_POLICYBENCH, canonical_pair,
                                                       checkpoint_inventory, csv_words, load_task_registry)
    from harnessstudio.core.manifest import freeze, method_manifest
    from harnessstudio.core.runtime import method_steps_per_step
    import runner
    from worker import RUNNER_VERSION
    policy, benchmark = canonical_pair(args.policy, args.benchmark)
    if policy != "gr00t-n1.5" or benchmark != "robocasa365":
        raise ValueError("The HarnessStudio GR00T backend supports only GR00T N1.5 / RoboCasa365")
    if args.harness_config is None:
        raise ValueError("--harness-config is required (schema harnessstudio-run/1)")
    config_path = args.harness_config.resolve()
    config = validate_config(read_json(config_path))
    method, origin = resolve_method(config)
    mm = method_manifest(method, origin, policy=policy)
    per_step = method_steps_per_step(method)
    bench = (args.policybench or DEFAULT_POLICYBENCH).resolve()
    registry = load_task_registry(bench)
    groups, cases = build_cases(registry, ["all"] if args.all_groups else csv_words(args.groups),
                                csv_words(args.tasks), args.num_tasks, args.episodes_per_task, args.trials)
    protocol_path, protocol = load_protocol(args.protocol)
    table_path, table, table_sha = load_table(args.canonical_table)
    if table is not None:
        missing = [c["id"] for c in cases if c["id"] not in table["cases"]]
        if missing:
            raise ValueError("canonical table lacks cases: " + ", ".join(missing[:5]))
    gpu_request = args.gpus or "0"
    if gpu_request != "available":
        bcli.parse_gpu_ids(gpu_request)
    hours = args.max_hours if args.max_hours is not None else 1
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("max-hours must be finite and positive")
    identity = harness_identity(method, config)
    # 'harness' is the per-arm identity: runner_util.summarize requires every record's
    # harness to equal manifest['harness'], and assess requires distinct identities across arms.
    # The catalog name lives in 'harness_name' (checked by resume/status).
    plan = {"policy": policy, "benchmark": benchmark, "harness": identity, "harness_name": name,
        "harness_interface": "agent_adapter", "harness_config": config,
        "harness_config_path": str(config_path), "harness_config_sha256": canonical_sha256(config),
        "harness_identity": identity, "arm": config["arm"],
        "baseline_run": config["baseline_run"], "method_manifest": mm,
        "stepping": "per_step" if per_step else "chunk", "mode": "small", "policybench": str(bench),
        "groups": groups, "cases": cases, "expected_episodes": len(cases),
        "physical_step_upper_bound": sum(c["horizon"] for c in cases),
        "native_policy_call_upper_bound": sum(arm_call_bound(method, config["arm"], c["horizon"]) for c in cases),
        "gpu_request": gpu_request, "split": args.split or "pretrain",
        "videos": True,  # every arm (RI-2 scan, plan clarification 8)
        "record_arrays": bool(args.record_arrays), "max_hours": hours,
        "output": str(args.output.resolve()), "official_trials_per_task": 50,
        "official_sequential_reset_equivalent": False,
        "reset": "ADAPTED: env.reset(seed=7+50*within_group_task_index+trial) for identical initial states across arms; official GR00T eval is unseeded",
        "noise": "ADAPTED: Protocol N1 common random numbers (ChunkTrust v2.2 crn), sample 0 identical in all arms",
        "comparison": "paired seeded scenes across baseline / null / candidate on identical case lists; subset screen, development evidence",
        "runner_version": RUNNER_VERSION, "evaluator_step_log": bool(args.evaluator_step_log),
        "protocol": protocol, "protocol_path": str(protocol_path),
        "canonical_table_path": None if table_path is None else str(table_path),
        "canonical_table_sha256": table_sha, "canonical_table": table}
    # Common to all arms of a phase (never arm config): protocol, table, checkpoint, stepping, backend.
    checkpoint, files, revision = checkpoint_inventory(bench, rehash=args.rehash_checkpoint)
    identity_files = {str(Path(p).relative_to(checkpoint)): row["sha256"] for p, row in files.items()}
    checkpoint_identity = json_hash(identity_files)
    plan["protocol_sha256"] = json_hash({"protocol": protocol, "canonical_table_sha256": table_sha,
                                         "checkpoint_identity": checkpoint_identity,
                                         "stepping": plan["stepping"], "backend": RUNNER_VERSION})
    if config["arm"] == "null":
        check_baseline_run(config, method, cases, plan["protocol_sha256"])
    if args.dry_run:
        preview = {}
        if gpu_request != "available":
            ids = bcli.parse_gpu_ids(gpu_request)[:len(cases)]
            slots = runner.plan_slots(ids, args.workers_per_gpu, len(cases))
            placement = runner.placement_map({k: cases[i::len(slots)] for i, k in enumerate(slots)})
            preview = {"placement": placement, "placement_sha256": canonical_sha256(placement)}
        return dict(plan, canonical_table="<omitted in preview>" if table else None,
                    method_manifest={"sha256": mm["sha256"], "method": mm["method"], "closure_files": len(mm["closure"])},
                    checkpoint_identity=checkpoint_identity, **preview, dry_run=True, gpu_execution=False,
                    source_frozen=False)
    if gpu_request == "available":
        from harnessstudio.bench import resources
        ids = [g["index"] for g in resources.snapshot() if g.get("processes_known") and
               not g["processes"] and g["used_mib"] <= 2048]
        if not ids:
            raise ValueError("No unoccupied GPU available")
    else:
        ids = bcli.parse_gpu_ids(gpu_request)
    if args.workers_per_gpu > 1 and not args.calibrated_worker_mib:
        raise ValueError("--workers-per-gpu > 1 requires a measured --calibrated-worker-mib")
    ids = ids[:len(cases)]
    slots = runner.plan_slots(ids, args.workers_per_gpu, len(cases))
    ids = sorted({int(k.split(".")[0]) for k in slots})
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Output exists. Choose a new folder, or --resume the frozen run.")
    assignments = {key: cases[i::len(slots)] for i, key in enumerate(slots)}
    placement = runner.placement_map(assignments)
    placement_sha = canonical_sha256(placement)
    if args.require_placement_sha256 and args.require_placement_sha256 != placement_sha:
        raise ValueError(f"Placement differs from the required map: {placement_sha}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        frozen = freeze(mm, output / "frozen_closure")
        extras = [p.resolve() for p in args.harness_source] + [
            output / "frozen_closure", HERE / "COPIED.json", config_path, protocol_path]
        if table_path is not None:
            extras.append(table_path)
        sources, repos = bcli.source_inventory(bench, extras)
        plan.update(schema_version=1, created_at=now(), deadline=time.time() + hours * 3600,
            gpu_ids=ids, workers_per_gpu=args.workers_per_gpu, calibrated_worker_mib=args.calibrated_worker_mib,
            assignments=assignments, placement=placement, placement_sha256=placement_sha,
            checkpoint_path=checkpoint, checkpoint_files=files, checkpoint_identity=checkpoint_identity,
            checkpoint_revision=revision, source_sha256=sources, repositories=repos, frozen_closure=frozen,
            lease_owner="benchmark.harnessstudio." + uuid.uuid4().hex)
        plan["manifest_digest"] = json_hash(plan)
        write_json(output / "manifest.json", plan)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)  # only the directory this call just created
        raise
    return plan


class BackendSpec:
    """What this backend supports. External pipelines wrap it in their own registration type."""
    def __init__(self, name, policies, benchmarks):
        self.name, self.policies, self.benchmarks = name, tuple(policies), tuple(benchmarks)


class StudioAdapter:
    def __init__(self, name=NAME, *, policies=("gr00t-n1.5",), prepare_fn=None):
        self.spec = BackendSpec(name, policies, ("robocasa365",))
        self.prepare = prepare_fn or prepare

    def run_cli(self, argv):
        from harnessstudio.bench.common import read_json, verify_manifest, verify_manifest_digest
        import runner
        args = parser().parse_args(list(argv))
        if args.harness != self.spec.name:
            raise ValueError("adapter/harness name mismatch")
        if args.output is None:
            if args.resume or args.status:
                raise ValueError("--output is required for --resume or --status")
            args.output = config.runs_dir() / ("run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                                          + "-" + uuid.uuid4().hex[:8])
        try:
            manifest_path = args.output.resolve() / "manifest.json"
            if args.resume or args.status:
                manifest = (verify_manifest_digest(read_json(manifest_path)) if args.status
                            else verify_manifest(manifest_path))
                if manifest.get("harness_name") != self.spec.name:
                    raise ValueError("Run belongs to another harness")
                if args.status:
                    result = runner.summarize(manifest)
                    print(json.dumps({k: v for k, v in result.items() if k != "completed_evidence_sha256"}, indent=2))
                    return 0
                if (args.harness_config or args.harness_source or args.rehash_checkpoint or args.protocol
                        or args.canonical_table or args.require_placement_sha256
                        or args.workers_per_gpu != 1 or args.calibrated_worker_mib):
                    raise ValueError("Resume uses frozen source/config/checkpoint")
            else:
                manifest = self.prepare(args, self.spec.name)
                if args.dry_run:
                    print(json.dumps(manifest, indent=2, default=str))
                    return 0
            print(f"{manifest['policy']} / {manifest['benchmark']} / {manifest['harness_identity']}: "
                  f"{len(manifest['cases'])} episodes -> {manifest['output']}", flush=True)
            code = runner.run(manifest_path, resume=args.resume)
            result = runner.summarize(manifest)
            print(f"{result['status']}: {result['successes']}/{result['completed_episodes']} successes; "
                  f"{result['completed_episodes']}/{result['expected_episodes']} episodes complete. "
                  f"Saved summary.json and episodes.csv in {manifest['output']}", flush=True)
            return code
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            print("Evaluation stopped: " + str(exc), file=sys.stderr)
            return 2


def build_harness():
    return StudioAdapter()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        internal = argparse.ArgumentParser()
        internal.add_argument("--manifest", type=Path, required=True)
        internal.add_argument("--gpu", type=int, required=True)
        internal.add_argument("--slot", type=int, default=0)
        a = internal.parse_args(sys.argv[2:])
        import runner
        raise SystemExit(runner.worker(a.manifest, a.gpu, a.slot))
    raise SystemExit(StudioAdapter().run_cli(sys.argv[1:]))
