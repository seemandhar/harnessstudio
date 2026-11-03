#!/usr/bin/env python3
"""gpp evaluator plan: tier 2 (t8i2, t3i6, t7i1) and the amendment-B gpp-core screen (review proposals-r2-review.json
screen_specs_amendment_B.gpp_core): candidate + directional sham + shadow null on the fire set F_gpp (17 greedy scenes,
recomputed here with the frozen ONLINE rules from the map greedy records and hashed) + a 10-scene retention witness
(per task the lowest init not in F_gpp; candidate only, must equal the greedy record bitwise).

    harnessstudio/.venv/bin/python harnessstudio/methods/gpp/scripts/write_screen_configs.py \
        --eval-dir runs/scalepp-20261102/eval/gpp-core [--entry gpp-core|gpp-h2] [--servers t0=g4,t1=g5,...]

Configs are greedy-escape's write_screen_configs.config (imported read-only): the map config + send_state: true, one
scene per job, LIBEROCampaignSelect (episode_offset + match_sequential_resets) for init > 0, case namespace '', CRN
namespace 'hs' (greedy ignores seeds), max_steps 520 (stock), num_steps_wait 10, record_video true. Default servers =
the map's task -> server assignment (eval/devbase/assignments.json; P-20261031-09); the evaluator may override per task
(--servers) and records the assignment in plan.json. Jobs on g7 carry greedy-escape's S5 relaunch precondition.
`--entry gpp-h2` writes the time-guard fallback plan (fire set = the 11 H2 scenes; sham gpp-h2-sham).
Reads no outcomes. Never addresses libero_10 init >= 10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude")
RUN = ROOT / "runs/scalepp-20261102"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "harnessstudio/methods/greedy_escape/scripts"))
import fire_set as FS  # noqa: E402
import write_screen_configs as W  # noqa: E402   (greedy-escape's frozen config writer, read-only)

TIER2 = ["t8i2", "t3i6", "t7i1"]
SHAMS = {"gpp-core": "gpp-core-sham", "gpp-h2": "gpp-h2-sham"}


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(entry: str) -> str:
    from harnessstudio.core import registry
    return registry.load(entry).fingerprint()


def job(eval_dir: Path, phase: str, method: str, arms: str, scene: str, servers: dict, verify: bool = False) -> dict:
    t, i = int(scene[1]), int(scene[3:])
    srv = servers[f"t{t}"]
    jid = f"{phase}-{scene}"
    out = eval_dir / phase / jid
    cfg, overrides = W.config(jid, W.SERVERS[srv]["gpu"], out, t, i, 1)
    path = eval_dir / "configs" / f"{jid}.yaml"
    path.write_text(f"# gpp {phase} {jid} ({method}, arms {arms}): libero_10 task {t} init {i}, env seed 7, server {srv}. "
                    "Written by methods/gpp/scripts/write_screen_configs.py.\n" + W.yaml_dump(cfg) + "\n")
    common = ["--policy-url", W.SERVERS[srv]["url"], "--contract", "libero", "--profile", "openvla-libero-hs",
              "--namespace", "hs", "--case-namespace", "", "--port", "0", "--action-timeout", "300", "--out", str(out)]
    if verify:
        argv = [str(W.HS), "verify", method, "--tier", "2", "--config", str(path)] + common
    else:
        argv = [str(W.HS), "eval", method, "--arms", arms, "--config", str(path)] + common
    if overrides:
        argv += ["--benchmark-overrides", overrides]
    j = {"id": jid, "phase": phase, "method": method, "arms": arms, "task": t, "inits": [i], "scene": scene,
         "server": srv, "gpu": W.SERVERS[srv]["gpu"], "policy_url": W.SERVERS[srv]["url"], "config": str(path),
         "out": str(out), "cwd": str(W.VLA), "env_unset": ["PYTHONPATH", "CUDA_VISIBLE_DEVICES"], "argv": argv}
    if srv == "g7":
        j["preconditions"] = ["S5"]
    return j


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(RUN / "eval/gpp-core"))
    ap.add_argument("--entry", default="gpp-core", choices=sorted(SHAMS))
    ap.add_argument("--servers", default="", help="task overrides, e.g. t6=g6,t9=g5 (recorded in plan.json)")
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    eval_dir = (eval_dir if eval_dir.is_absolute() else ROOT / eval_dir).resolve()
    (eval_dir / "configs").mkdir(parents=True, exist_ok=True)
    servers = json.loads((RUN / "eval/devbase/assignments.json").read_text())
    for kv in filter(None, args.servers.split(",")):
        k, v = kv.split("=")
        if v not in W.SERVERS:
            raise SystemExit(f"unknown server {v}")
        servers[k] = v
    table = FS.fire_table(args.entry)
    table_text = json.dumps(table, indent=1, sort_keys=True)
    (eval_dir / "fire_set.json").write_text(table_text)
    F = table["fire_set"]
    retention = []
    for t in range(10):
        free = [i for i in range(10) if f"t{t}i{i}" not in F]
        retention.append(f"t{t}i{free[0]}")
    sham = SHAMS[args.entry]
    jobs = []
    for scene in TIER2 if args.entry == "gpp-core" else [s for s in TIER2 if table["scenes"][s]["h2"]]:
        jobs.append(job(eval_dir, "tier2", args.entry, "baseline,null,candidate", scene, servers, verify=True))
    for scene in F:
        jobs.append(job(eval_dir, "candidate", args.entry, "candidate", scene, servers))
        jobs.append(job(eval_dir, "sham", sham, "candidate", scene, servers))
        jobs.append(job(eval_dir, "null", args.entry, "null", scene, servers))
    for scene in retention:
        jobs.append(job(eval_dir, "retention", args.entry, "candidate", scene, servers))
    plan = {"method": args.entry, "sham": sham,
            "method_fingerprint": fingerprint(args.entry), "sham_fingerprint": fingerprint(sham),
            "written_by": "harnessstudio/methods/gpp/scripts/write_screen_configs.py",
            "eval_dir": str(eval_dir), "task_server": servers,
            "fire_set": F, "fire_set_json": str(eval_dir / "fire_set.json"), "fire_set_sha256": sha_text(table_text),
            "first_fires": {s: table["scenes"][s]["first"] for s in F},
            "retention_scenes": retention, "tier2_scenes": [j["scene"] for j in jobs if j["phase"] == "tier2"],
            "preconditions": {"S5": W.S5_G7_RELAUNCH},
            "frames": ("after each job: python3 harnessstudio/methods/gpp/scripts/frames.py --plan PLAN --job ID "
                       "(system python with cv2): fire, start, end, +10, +30 frames per primitive"),
            "disclosure": ("gpp is deterministic on the greedy base (m = 1); scripted steps make no policy call and count "
                           "against the 520 horizon; sham and candidate are paired exactly before the fire"),
            "jobs": jobs}
    (eval_dir / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(jobs)} jobs; F = {len(F)} scenes -> {eval_dir / 'plan.json'}")


if __name__ == "__main__":
    main()
