"""Run kinetix-flow cases through the UNCHANGED core.pipeline.run_episode (child interpreter).

Repair 1: one case (= one HarnessStudio episode) covers the WHOLE ordered level universe, level-batched as
the authors' eval_flow does.

    python -m harnessstudio.backends.kinetix_rtc12.worker --method rtc --arms baseline null candidate \
        --points 1:1 4:4 --level-universe worlds/l/grasp_easy.json ... --rollouts 64 --out DIR

Writes DIR/manifest.json, DIR/results/<arm>/<case_id>/{result.json, trace.jsonl, rollouts.npz},
DIR/attribution.json (per-rollout, per-level M/R/X of null vs baseline, when both arms ran) and DIR/state.json
(gpu_hours_upper_bound for the GPU ledger). Device selection is the caller's (JAX_PLATFORMS /
CUDA_VISIBLE_DEVICES); the parent launcher owns leases.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from harnessstudio.policies import kinetix_flow as KF

RUNNER_VERSION = "harnessstudio-kinetix-rtc12-2"   # repair 1: level-batched universe cases
MAX_CASES_PER_ARM = 32          # the 32-episode cap counts batched cases
MAX_ROLLOUTS_PER_ARM = 65536    # counts L * B per case


def protocol(case) -> dict:
    return {"benchmark": KF.BENCHMARK, "runner_version": RUNNER_VERSION, "upstream_commit": KF.UPSTREAM_COMMIT,
            "upstream_sha256": KF.UPSTREAM_SHA256, "kinetix_commit": KF.KINETIX_COMMIT,
            "level_universe": list(case.universe), "frame_skip": 2, "max_timesteps": KF.MAX_TIMESTEPS,
            "action_noise_std": KF.ACTION_NOISE_STD, "wrappers": ["NoisyAction", "AutoReplay", "Log", "Batch"],
            "level_batching": "jit(shard_map(vmap over levels)) as eval_flow.py:222-228",
            "inference_delay": case.d, "execute_horizon": case.s, "rollouts_per_level": case.rollouts,
            "block": case.block, "horizon_steps": case.horizon_steps, "metric": "first_episode_solved",
            "noise_mode": KF.NOISE_MODE, "seed": case.seed,
            "checkpoint_sha256": [c["sha256"] for c in case.checkpoints], "num_flow_steps": KF.NUM_FLOW_STEPS}


def case_id(universe, d: int, s: int, block: int = 0) -> str:
    from .runtime import universe_tag
    return f"d{d}s{s}--U{universe_tag(universe)}--b{block}"


def episode_spec(case, cid: str, manifest_sha256: str = ""):
    from harnessstudio.core.pipeline import EpisodeSpec
    from harnessstudio.runtime.contracts import public_hash

    def input_hashes(observation):
        out = {k: public_hash(observation[k]) for k in sorted(observation)}
        out["__aggregate__"] = public_hash(out)
        return out

    return EpisodeSpec(case_id=cid, horizon=case.horizon_steps, chunk_len=case.s,
                       action_dim=case.L * case.rollouts * KF.ACTION_DIM,
                       control_hz=KF.CONTROL_HZ, benchmark=KF.BENCHMARK, public_keys=KF.PUBLIC_KEYS,
                       noise_namespace=KF.BENCHMARK, instruction_of=lambda o: o["kinetix/instruction"],
                       with_instruction=None, input_hashes=input_hashes,
                       action_hashes=lambda native: public_hash(np.asarray(native)),
                       method_manifest_sha256=manifest_sha256)


def run_case(method, arm, case, *, baseline_core_traces=None, record_steps=False, trace_rows=None):
    """One arm of one level-batched case. Returns (record, env_port, client)."""
    from harnessstudio.core.arms import Arm
    from harnessstudio.core.capabilities import unified_names
    from harnessstudio.core.pipeline import run_episode
    from harnessstudio.core.protocol import require_method
    from harnessstudio.runtime import PolicySpec
    from .runtime import KinetixClient, KinetixEnvPort, UniverseRng
    require_method(method, KF)
    arm = Arm(arm)
    rng = UniverseRng(case.level_keys)
    port = KinetixEnvPort(case, rng, record_steps=record_steps)
    sampler = None
    if arm is not Arm.BASELINE and "sampler" in method.slots:
        sampler = method.slots["sampler"].build()           # fresh server-slot module per (episode, arm)
    client = KinetixClient(case, rng, sampler)
    rows = trace_rows if trace_rows is not None else []
    spec = episode_spec(case, case_id(case.universe, case.d, case.s, case.block))
    identity = hashlib.sha256(json.dumps([c["sha256"] for c in case.checkpoints]).encode()).hexdigest()
    pspec = PolicySpec(KF.POLICY_ID, identity, unified_names(KF.CAPABILITIES))
    record = run_episode(method, arm, spec, port.observation(), port, client, pspec, trace=rows.append,
                         baseline_core_traces=baseline_core_traces)
    return record, port, client


def summarize(record, port, client, case) -> tuple:
    from harnessstudio.runtime.contracts import canonical_hash
    out = port.outcomes()
    proto = protocol(case)
    fp = np.stack([np.stack(p) for p in port.fingerprints])        # [decisions, 2(obs, action), L, B, 2]
    levels = {}
    for li, lv in enumerate(case.universe):
        solved = out["rollout_solved"][li].astype(bool)
        packed = np.packbits(solved).tobytes()
        levels[lv] = {"solved_count": int(solved.sum()), "success_rate": float(solved.mean()),
                      "returns_mean": float(np.mean(out["rollout_returns"][li])),
                      "lengths_mean": float(np.mean(out["rollout_lengths"][li])),
                      "rollout_solved_b64": base64.b64encode(packed).decode(),
                      "rollout_solved_sha256": hashlib.sha256(packed).hexdigest(),
                      "checkpoint": case.checkpoints[li]}
    result = {"schema": "harnessstudio-kinetix-result/2", "case_id": case_id(case.universe, case.d, case.s, case.block),
              "arm": record["arm"], "method": record["method"], "protocol": proto, "protocol_sha256": canonical_hash(proto),
              "success": None, "success_kind": "rollout_vector_per_level", "rollouts_per_level": case.rollouts,
              "levels": levels, "solved_count": int(sum(v["solved_count"] for v in levels.values())),
              "decisions": record["decisions"], "steps": record["steps"], "policy_calls": record["policy_calls"],
              "warmstart_policy_calls": port.warmstart_policy_calls,
              "guidance_vjp_steps": KF.NUM_FLOW_STEPS * len(client.guidance_sha256),
              "guidance_sha256": client.guidance_sha256, "policy_seconds": record["policy_seconds"],
              "max_policy_calls": record["max_policy_calls"], "termination": record["termination"],
              "divergence": record.get("divergence"), "fingerprints_sha256": hashlib.sha256(fp.tobytes()).hexdigest()}
    arrays = {"fingerprints": fp, "rollout_solved": out["rollout_solved"], "rollout_returns": out["rollout_returns"],
              "rollout_lengths": out["rollout_lengths"]}
    return result, arrays


def classify(baseline_fp: np.ndarray, other_fp: np.ndarray) -> dict:
    """Per-rollout M/R/X per level from [decisions, 2(obs, action), L, B, 2] fingerprints."""
    obs_eq = np.all(baseline_fp[:, 0] == other_fp[:, 0], axis=-1)    # [decisions, L, B]
    act_eq = np.all(baseline_fp[:, 1] == other_fp[:, 1], axis=-1)
    per_level = []
    for li in range(obs_eq.shape[1]):
        labels = []
        for b in range(obs_eq.shape[2]):
            bad = np.nonzero(~(obs_eq[:, li, b] & act_eq[:, li, b]))[0]
            if bad.size == 0:
                labels.append("M")
                continue
            k = int(bad[0])
            labels.append("X" if obs_eq[k, li, b] and not act_eq[k, li, b] else "R")
        per_level.append({"M": labels.count("M"), "R": labels.count("R"), "X": labels.count("X"), "labels": "".join(labels)})
    return {"M": sum(p["M"] for p in per_level), "R": sum(p["R"] for p in per_level),
            "X": sum(p["X"] for p in per_level), "per_level": per_level}


def main(argv=None) -> int:
    from harnessstudio.core import registry
    from harnessstudio.core.protocol import require_protocol
    from .runtime import KinetixUniverseCase
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--arms", nargs="+", default=["baseline", "null", "candidate"])
    p.add_argument("--points", nargs="+", required=True, help="d:s pairs")
    p.add_argument("--level-universe", nargs="+", default=list(KF.LEVELS))
    p.add_argument("--rollouts", type=int, default=2048, help="rollouts per level")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    n_cases = len(a.points)
    if n_cases > MAX_CASES_PER_ARM or n_cases * len(a.level_universe) * a.rollouts > MAX_ROLLOUTS_PER_ARM:  # EC5 caps
        raise SystemExit(f"run cap: {n_cases} cases x {len(a.level_universe)} levels x {a.rollouts} rollouts per arm "
                         f"exceeds {MAX_CASES_PER_ARM} cases / {MAX_ROLLOUTS_PER_ARM} rollouts")
    method, origin = registry.get(a.method)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    import jax
    manifest = {"schema": "harnessstudio-kinetix-run/2", "runner_version": RUNNER_VERSION, "argv": argv,
                "method": method.to_json(), "origin": origin, "arms": a.arms, "points": a.points,
                "level_universe": a.level_universe, "rollouts_per_level": a.rollouts, "seed": a.seed, "block": a.block,
                "devices": [str(d) for d in jax.devices()], "jax_version": jax.__version__}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    state = {"status": "running", "started_at": started, "gpu_hours_upper_bound": 0.0}
    attribution = {}
    gpu = any(d.platform == "gpu" for d in jax.devices())
    for point in a.points:
        d, s = (int(x) for x in point.split(":"))
        require_protocol(KF, d, s)
        case = KinetixUniverseCase(a.level_universe, rollouts=a.rollouts, inference_delay=d, execute_horizon=s,
                                   seed=a.seed, block=a.block)
        cid = case_id(case.universe, d, s, a.block)
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
