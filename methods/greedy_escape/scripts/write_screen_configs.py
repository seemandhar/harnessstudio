#!/usr/bin/env python3
"""Write the greedy-escape tier-2 / grid-screen / shadow-null / init 4-9 confirmation vla-eval configs and the exact
`hs` argv for each job (review runs/scalepp-20261102/reviews/proposals-r1-review.json GE4, GE6, GE7).

    python write_screen_configs.py --eval-dir EVAL_DIR [--null-scenes t0i1,t2i2,...]

Every config equals the development map's (runs/scalepp-20261102/eval/devbase/scheduler.py CONFIG_TMPL: stock
libero/10.yaml, env seed 7, num_steps_wait 10, max_steps 520 by the stock mapping, record_video true) plus
`params.send_state: true` (the controller's public finger width; behaviour-neutral for OpenVLA). Jobs keep the map's
task -> policy server assignment (eval/devbase/assignments.json) so each candidate episode pairs with map records from
the same server (P-20261031-09). Scenes outside a contiguous prefix of a task's episodes run through the Campaign
bench extension LIBEROCampaignSelect (episode_offset + match_sequential_resets, mounted read-only) so they get the
fixture placement of the map's sequential run (P-20261101-12: the evaluator checks obs[0]/first action against the
paired map record). Prefix scenes (grid init 0-3) use the stock LIBEROBenchmark exactly as the map did.

Writes EVAL_DIR/configs/*.yaml and EVAL_DIR/plan.json. Reads no outcomes. Never addresses libero_10 init >= 10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude")
RUN = ROOT / "runs/scalepp-20261102"
HS = ROOT / "harnessstudio/.venv/bin/hs"
VLA = Path("/home/sejain/research/cvpr27/vla-eval-v0.8.0")
BENCH_EXT = ROOT / "robometaharness/integrations/harnessstudio_v2/bench_ext"
LIBERO_CONFIG = "/home/sejain/research/cvpr27/hs2-gpu/libero_config"
IMAGE = "ghcr.io/allenai/vla-evaluation-harness/libero:latest"
SERVERS = {"g4": {"gpu": "4", "url": "ws://127.0.0.1:8170"}, "g5": {"gpu": "5", "url": "ws://127.0.0.1:8171"},
           "g6": {"gpu": "6", "url": "ws://127.0.0.1:8172"}, "g7": {"gpu": "7", "url": "ws://127.0.0.1:8173"}}
TASKS = ["put both the alphabet soup and the tomato sauce in the basket",
         "put both the cream cheese box and the butter in the basket",
         "turn on the stove and put the moka pot on it",
         "put the black bowl in the bottom drawer of the cabinet and close it",
         "put the white mug on the left plate and put the yellow and white mug on the right plate",
         "pick up the book and place it in the back compartment of the caddy",
         "put the white mug on the plate and put the chocolate pudding to the right of the plate",
         "put both the alphabet soup and the cream cheese box in the basket",
         "put both moka pots on the stove",
         "put the yellow and white mug in the microwave and close it"]
# Grid scenes whose greedy map record fires the frozen trigger (tests/test_replay_devmap.py, recomputed with the
# frozen StallMonitor from the greedy records; equals the proposal's list).
NULL_SCENES = ["t0i0", "t0i1", "t2i0", "t2i2", "t3i0", "t3i3", "t4i3", "t7i0", "t7i2", "t8i2"]
FINGERPRINT = "f03731f11b84f97306ba5080b1def6b72e64468503a25813f46a700e6f9f9fe1"
# Source review S5: before any job on the g7 server (map tasks t6, t9) the evaluator relaunches scalepp-openvla-g7
# exactly like the map and records these identities; jobs carrying "preconditions": ["S5"] wait for them.
S5_G7_RELAUNCH = {
    "when": "before the t6/t9 grid and confirm jobs; only after the evaluator's own ksample server on GPU 7 is stopped "
            "(never a foreign process); GPU 7 leased by the evaluator",
    "launch": ["systemd-run", "--user", "--unit", "scalepp-openvla-g7", "-p", "MemoryMax=48G",
               str(RUN / "eval/services/launch_openvla_unit_ldlp.sh"), "8173", "7"],
    "launcher_sha256": "2c5643c800e6556af7d52d6d7a55c144b69f2e3d938971cbdd8a8503f2f6f0c5",
    "server_config": "/home/sejain/research/cvpr27/hs2-gpu/configs/openvla_libero10_hs.yaml",
    "server_config_sha256": "ca24e559f8c3bfb693d54002456316d76e88c14fa677733d047de80c088cd78f",
    "record_at_launch": {
        "openvla_hs.py_sha256": "e7c6a4c302fece60b338b9a6c14c7ac0cea4dea2057ff72b74c0abfa0aab174b (openvla-ksample NOT merged)",
        "hs_scale/decoder.py_sha256": "619d9b960bd84878a34825c1e35e7f182cade2515cc51721a01fa5fe14256d60",
        "mapped_cublasLt": "libcublasLt.so.12.6.4.1 from /usr/local/cuda-12.6 (grep /proc/PID/maps after the first request)",
        "pid_unit_lease": "PID, unit name and lease owner in eval/services/services.jsonl and server_record.json"},
    "cross_process_check": "the 8 untriggered t6/t9 grid scenes must be bitwise equal to their greedy map records "
                           "(screen_summary retention / S3 gates)"}
TIER2_SCENE = "t7i2"   # earliest grid fire (step 42): exercises a whole 40-decision window and the shadow null


def yaml_dump(obj, indent=0) -> str:
    """Minimal YAML writer (dicts/lists/scalars) so the script needs no third-party package."""
    pad = "  " * indent
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:")
                out.append(yaml_dump(v, indent + 1))
            else:
                out.append(f"{pad}{k}: {json.dumps(v)}")
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict):
                lines = yaml_dump(v, indent + 1).splitlines()
                out.append(f"{pad}- {lines[0].strip()}")
                out.extend(lines[1:])
            else:
                out.append(f"{pad}- {json.dumps(v)}")
    return "\n".join(out)


def config(job_id: str, gpu: str, out_dir: Path, task: int, first_init: int, n: int) -> tuple[dict, str | None]:
    """vla-eval config; returns (config, --benchmark-overrides JSON or None)."""
    if first_init + n > 10:
        raise SystemExit("development map scenes only (init 0-9)")
    params = {"suite": "libero_10", "seed": 7, "num_steps_wait": 10, "send_state": True}
    docker = {"image": IMAGE, "gpus": gpu, "user": "host",
              "env": ["HOME=/tmp", "LIBERO_CONFIG_PATH=/etc/libero"],
              "volumes": [f"{LIBERO_CONFIG}:/etc/libero:ro"]}
    bench = {"benchmark": "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark", "subname": "libero_10",
             "episodes_per_task": n, "params": params, "recording": {"record_video": True}}
    overrides = json.dumps({"episodes_per_task": n, "tasks": [TASKS[task]]})
    if first_init:
        bench["benchmark"] = "hs2_campaign_bench:LIBEROCampaignSelect"
        params.update(task_ids=[task], episode_offset=first_init, match_sequential_resets=True)
        docker["env"].append("PYTHONPATH=/rmh_hs2_bench")
        docker["volumes"].append(f"{BENCH_EXT}:/rmh_hs2_bench:ro")
        overrides = None                      # task selected by task_ids
    cfg = {"extends": str(VLA / "configs/benchmarks/libero/10.yaml"), "server": {"url": "ws://localhost:1"},
           "docker": docker, "output_dir": str(out_dir / "vla-eval-unused"), "benchmarks": [bench]}
    return cfg, overrides


def job(eval_dir: Path, phase: str, arms: str, task: int, first_init: int, n: int, assign: dict) -> dict:
    srv = assign[f"t{task}"]
    jid = f"{phase}-t{task}" + (f"i{first_init}" if phase in ("null", "tier2") else "")
    out = eval_dir / phase / jid
    cfg, overrides = config(jid, SERVERS[srv]["gpu"], out, task, first_init, n)
    path = eval_dir / "configs" / f"{jid}.yaml"
    head = (f"# greedy-escape {phase} {jid}: libero_10 task {task}, init {first_init}-{first_init + n - 1}, env seed 7, "
            f"server {srv}. Written by methods/greedy_escape/scripts/write_screen_configs.py.\n")
    path.write_text(head + yaml_dump(cfg) + "\n")
    common = ["--policy-url", SERVERS[srv]["url"], "--contract", "libero", "--profile", "openvla-libero-hs",
              "--port", "0", "--action-timeout", "300", "--out", str(out)]
    if phase == "tier2":
        argv = [str(HS), "verify", "greedy-escape", "--tier", "2", "--config", str(path)] + common
    else:
        argv = [str(HS), "eval", "greedy-escape", "--arms", arms, "--config", str(path)] + common
    if overrides:
        argv += ["--benchmark-overrides", overrides]
    return {"id": jid, "phase": phase, "arms": arms, "task": task, "inits": list(range(first_init, first_init + n)),
            "server": srv, "gpu": SERVERS[srv]["gpu"], "policy_url": SERVERS[srv]["url"], "config": str(path),
            "out": str(out), "cwd": str(VLA), "env_unset": ["PYTHONPATH", "CUDA_VISIBLE_DEVICES"], "argv": argv}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(RUN / "eval/greedy-escape"))
    ap.add_argument("--null-scenes", default=",".join(NULL_SCENES))
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    eval_dir = (eval_dir if eval_dir.is_absolute() else ROOT / eval_dir).resolve()   # jobs run with cwd = vla-eval
    (eval_dir / "configs").mkdir(parents=True, exist_ok=True)
    assign = json.loads((RUN / "eval/devbase/assignments.json").read_text())
    jobs = []
    t, i = int(TIER2_SCENE[1]), int(TIER2_SCENE[3])
    jobs.append(job(eval_dir, "tier2", "baseline,null,candidate", t, i, 1, assign))
    for t in range(10):
        jobs.append(job(eval_dir, "grid", "candidate", t, 0, 4, assign))
    for scene in [s for s in args.null_scenes.split(",") if s]:
        t, i = int(scene[1]), int(scene[3])
        if i > 3:
            raise SystemExit("the shadow null covers grid scenes (init 0-3) only")
        jobs.append(job(eval_dir, "null", "null", t, i, 1, assign))
    for t in range(10):
        jobs.append(job(eval_dir, "confirm", "candidate", t, 4, 6, assign))
    for j in jobs:
        if j["server"] == "g7":
            j["preconditions"] = ["S5"]
    plan = {"method": "greedy-escape", "method_fingerprint": FINGERPRINT,
            "preconditions": {"S5": S5_G7_RELAUNCH},
            "disclosure": ("window decisions use the SCALE map's CRN seeds but a different observation and decoder state: "
                           "window actions are not paired with the SCALE record; SCALE pairing is outcome-level by scene "
                           "only (source review F6)"), "written_by": "harnessstudio/methods/greedy_escape/scripts/write_screen_configs.py",
            "eval_dir": str(eval_dir), "task_server": assign, "servers": SERVERS, "null_scenes": args.null_scenes.split(","),
            "jobs": jobs}
    (eval_dir / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(jobs)} jobs -> {eval_dir / 'plan.json'}")


if __name__ == "__main__":
    main()
