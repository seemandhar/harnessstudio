"""Run kinetix-flow cases through the UNCHANGED core.pipeline.run_episode (child interpreter).

    python -m harnessstudio.backends.kinetix_rtc12.worker --method rtc --arms baseline null candidate \
        --points 1:1 4:4 --levels worlds/l/grasp_easy.json --rollouts 64 --out DIR [--level-universe ...]

Writes DIR/manifest.json, DIR/results/<arm>/<case_id>/{result.json, trace.jsonl, rollouts.npz},
DIR/attribution.json (per-rollout M/R/X of null vs baseline, when both arms ran) and DIR/state.json
(gpu_hours_upper_bound for the GPU ledger). Device selection is the caller's (JAX_PLATFORMS /
CUDA_VISIBLE_DEVICES); the parent launcher owns leases.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from harnessstudio.policies import kinetix_flow as KF

RUNNER_VERSION = "harnessstudio-kinetix-rtc12-1"
MAX_CASES_PER_ARM = 32          # the 32-episode cap counts batched cases
MAX_ROLLOUTS_PER_ARM = 65536


def protocol(case) -> dict:
    return {"benchmark": KF.BENCHMARK, "runner_version": RUNNER_VERSION, "upstream_commit": KF.UPSTREAM_COMMIT,
            "upstream_sha256": KF.UPSTREAM_SHA256, "kinetix_commit": KF.KINETIX_COMMIT,
            "level_universe": list(case.level_universe), "level_index": case.level_index, "level": case.level,
            "frame_skip": 2, "max_timesteps": KF.MAX_TIMESTEPS, "action_noise_std": KF.ACTION_NOISE_STD,
            "wrappers": ["NoisyAction", "AutoReplay", "Log", "Batch"], "inference_delay": case.d,
            "execute_horizon": case.s, "rollouts": case.rollouts, "block": case.block,
            "horizon_steps": case.horizon_steps, "metric": "first_episode_solved", "noise_mode": KF.NOISE_MODE,
            "seed": case.seed, "checkpoint_sha256": case.checkpoint["sha256"], "num_flow_steps": KF.NUM_FLOW_STEPS}


def case_id(level: str, d: int, s: int, block: int = 0) -> str:
    return f"d{d}s{s}--{level.split('/')[-1].replace('.json', '')}--b{block}"


def episode_spec(case, cid: str, manifest_sha256: str = ""):
    from harnessstudio.core.pipeline import EpisodeSpec
    from harnessstudio.runtime.contracts import public_hash

    def input_hashes(observation):
        out = {k: public_hash(observation[k]) for k in sorted(observation)}
        out["__aggregate__"] = public_hash(out)
        return out

    return EpisodeSpec(case_id=cid, horizon=case.horizon_steps, chunk_len=case.s, action_dim=case.rollouts * KF.ACTION_DIM,
                       control_hz=KF.CONTROL_HZ, benchmark=KF.BENCHMARK, public_keys=KF.PUBLIC_KEYS,
                       noise_namespace=KF.BENCHMARK, instruction_of=lambda o: o["kinetix/instruction"],
                       with_instruction=None, input_hashes=input_hashes,
                       action_hashes=lambda native: public_hash(np.asarray(native)),
                       method_manifest_sha256=manifest_sha256)


def run_case(method, arm, case, *, baseline_core_traces=None, record_steps=False, trace_rows=None):
    """One arm of one batched case. Returns (record, env_port, client)."""
    from harnessstudio.core.arms import Arm
    from harnessstudio.core.capabilities import unified_names
    from harnessstudio.core.pipeline import run_episode
    from harnessstudio.core.protocol import require_method
    from harnessstudio.runtime import PolicySpec
    from .runtime import KeyChain, KinetixClient, KinetixEnvPort
    require_method(method, KF)
    arm = Arm(arm)
    chain = KeyChain(case.level_key)
    port = KinetixEnvPort(case, chain, record_steps=record_steps)
    sampler = None
    if arm is not Arm.BASELINE and "sampler" in method.slots:
        sampler = method.slots["sampler"].build()           # fresh server-slot module per (episode, arm)
    client = KinetixClient(case, chain, sampler)
    rows = trace_rows if trace_rows is not None else []
    spec = episode_spec(case, case_id(case.level, case.d, case.s, case.block))
    pspec = PolicySpec(KF.POLICY_ID, case.checkpoint["sha256"], unified_names(KF.CAPABILITIES))
    record = run_episode(method, arm, spec, port.observation(), port, client, pspec, trace=rows.append,
                         baseline_core_traces=baseline_core_traces)
    return record, port, client


def summarize(record, port, client, case) -> tuple:
    out = port.outcomes()
    solved = out["rollout_solved"].astype(bool)
    proto = protocol(case)
    from harnessstudio.runtime.contracts import canonical_hash
    fp = np.stack([np.stack(p) for p in port.fingerprints])        # [decisions, 2(obs,action), B, 2]
    result = {"schema": "harnessstudio-kinetix-result/1", "case_id": record.get("case_id") or
              case_id(case.level, case.d, case.s, case.block), "arm": record["arm"], "method": record["method"],
              "protocol": proto, "protocol_sha256": canonical_hash(proto), "success": None,
              "success_kind": "rollout_vector", "rollouts": case.rollouts, "solved_count": int(solved.sum()),
              "success_rate": float(solved.mean()),
              "returns_mean": float(np.mean(out["rollout_returns"])), "lengths_mean": float(np.mean(out["rollout_lengths"])),
              "rollout_solved_b64": base64.b64encode(np.packbits(solved).tobytes()).decode(),
              "rollout_solved_sha256": hashlib.sha256(np.packbits(solved).tobytes()).hexdigest(),
              "decisions": record["decisions"], "steps": record["steps"], "policy_calls": record["policy_calls"],
              "warmstart_policy_calls": port.warmstart_policy_calls,
              "guidance_vjp_steps": KF.NUM_FLOW_STEPS * len(client.guidance_sha256),
              "guidance_sha256": client.guidance_sha256, "policy_seconds": record["policy_seconds"],
              "max_policy_calls": record["max_policy_calls"], "termination": record["termination"],
              "divergence": record.get("divergence"), "checkpoint": case.checkpoint,
              "fingerprints_sha256": hashlib.sha256(fp.tobytes()).hexdigest()}
    arrays = {"fingerprints": fp, "rollout_solved": out["rollout_solved"], "rollout_returns": out["rollout_returns"],
              "rollout_lengths": out["rollout_lengths"]}
    return result, arrays


def classify(baseline_fp: np.ndarray, other_fp: np.ndarray) -> dict:
    """Per-rollout M/R/X from [decisions, 2(obs, action), B, 2] fingerprints."""
    obs_eq = np.all(baseline_fp[:, 0] == other_fp[:, 0], axis=-1)    # [decisions, B]
    act_eq = np.all(baseline_fp[:, 1] == other_fp[:, 1], axis=-1)
    labels = []
    for b in range(obs_eq.shape[1]):
        bad = np.nonzero(~(obs_eq[:, b] & act_eq[:, b]))[0]
        if bad.size == 0:
            labels.append("M")
            continue
        k = int(bad[0])
        labels.append("X" if obs_eq[k, b] and not act_eq[k, b] else "R")
    return {"M": labels.count("M"), "R": labels.count("R"), "X": labels.count("X"), "labels": "".join(labels)}


def main(argv=None) -> int:
    from harnessstudio.core import registry
    from .runtime import KinetixCase
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--arms", nargs="+", default=["baseline", "null", "candidate"])
    p.add_argument("--points", nargs="+", required=True, help="d:s pairs")
    p.add_argument("--levels", nargs="+", required=True)
    p.add_argument("--level-universe", nargs="+", default=list(KF.LEVELS))
    p.add_argument("--rollouts", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    from harnessstudio.core.protocol import require_protocol
    n_cases = len(a.points) * len(a.levels)
    if n_cases > MAX_CASES_PER_ARM or n_cases * a.rollouts > MAX_ROLLOUTS_PER_ARM:  # EC5 run caps
        raise SystemExit(f"run cap: {n_cases} cases x {a.rollouts} rollouts per arm exceeds "
                         f"{MAX_CASES_PER_ARM} cases / {MAX_ROLLOUTS_PER_ARM} rollouts")
    method, origin = registry.get(a.method)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    import jax
    manifest = {"schema": "harnessstudio-kinetix-run/1", "runner_version": RUNNER_VERSION, "argv": argv,
                "method": method.to_json(), "origin": origin, "arms": a.arms, "points": a.points, "levels": a.levels,
                "level_universe": a.level_universe, "rollouts": a.rollouts, "seed": a.seed, "block": a.block,
                "devices": [str(d) for d in jax.devices()], "jax_version": jax.__version__}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    state = {"status": "running", "started_at": started, "gpu_hours_upper_bound": 0.0}
    attribution = {}
    gpu = any(d.platform == "gpu" for d in jax.devices())
    for point in a.points:
        d, s = (int(x) for x in point.split(":"))
        require_protocol(KF, d, s)
        for level in a.levels:
            case = KinetixCase(level, rollouts=a.rollouts, inference_delay=d, execute_horizon=s,
                               level_universe=tuple(a.level_universe), seed=a.seed, block=a.block)
            cid = case_id(level, d, s, a.block)
            base_traces, fps = None, {}
            for arm in a.arms:
                rows = []
                record, port, client = run_case(method, arm, case, baseline_core_traces=base_traces if arm == "null" else None,
                                                trace_rows=rows)
                if arm == "baseline":
                    base_traces = record["core_traces"]
                result, arrays = summarize(record, port, client, case)
                folder = out / "results" / arm / cid
                folder.mkdir(parents=True)
                (folder / "result.json").write_text(json.dumps(result, indent=1))
                with (folder / "trace.jsonl").open("w") as f:
                    for r in rows:
                        f.write(json.dumps(r, default=str) + "\n")
                np.savez_compressed(folder / "rollouts.npz", **arrays)
                fps[arm] = arrays["fingerprints"]
            if "baseline" in fps:
                # M/R/X is defined for the null only (it must reproduce the baseline); the candidate differs by design.
                attribution[cid] = {"null": classify(fps["baseline"], fps["null"])} if "null" in fps else {}
            state["gpu_hours_upper_bound"] = (time.time() - started) / 3600 if gpu else 0.0
            (out / "state.json").write_text(json.dumps(state, indent=1))
    (out / "attribution.json").write_text(json.dumps(attribution, indent=1))
    state.update(status="complete", ended_at=time.time())
    (out / "state.json").write_text(json.dumps(state, indent=1))
    x = sum(v.get("null", {}).get("X", 0) for v in attribution.values())
    print(json.dumps({"out": str(out), "cases": len(attribution), "null_X": x}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
