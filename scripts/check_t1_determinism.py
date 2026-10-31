"""T1 checks for the HarnessStudio GR00T N1.5 RoboCasa365 policy server (run from the repo .venv).

    python scripts/check_t1_determinism.py --url ws://localhost:8105 --save run1.npz
    # restart the server, then:
    python scripts/check_t1_determinism.py --url ws://localhost:8105 --save run2.npz --compare run1.npz

Checks: (a) same obs + seed twice -> bitwise-identical chunk; (b) isolated streams: seed s inside
[s, t, u] equals [s] alone; (c) with --compare: every chunk equals the one saved by another server
process; (d) different seeds differ, chunk shape is [16, A] with A = sum of the HELLO action_spec dims;
(e) a native request still returns a normal action. Also times K=1 and K=8 requests, and checks the
velocity readout leaves the actions unchanged. Exit code 1 if any check fails.

The observation is synthetic (seeded uint8 images, plausible RoboCasa365 state) unless --packet points
to a saved GR00T-format packet (npz with video.* (1, H, W, 3), state.* (1, D), annotation.*).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import anyio
import numpy as np

from harnessstudio.context.policy import PolicyHandle
from harnessstudio.profiles import load_profile
from vla_eval.benchmarks.robocasa365.benchmark import STATE_KEYS, VIDEO_KEYS

STATE_DIMS = {"state.end_effector_position_relative": 3, "state.end_effector_rotation_relative": 4,
              "state.gripper_qpos": 2, "state.base_position": 3, "state.base_rotation": 4}
SEEDS = [11, 2**32 - 1, 0]          # s, t, u
K8 = [101, 102, 103, 104, 105, 106, 107, 108]


def synthetic_obs(camera: int = 256) -> dict:
    rng = np.random.default_rng(1234)
    images = {k: rng.integers(0, 256, (camera, camera, 3), dtype=np.uint8) for k in VIDEO_KEYS}
    state = {}
    for key in STATE_KEYS:
        v = rng.normal(scale=0.3, size=STATE_DIMS[key]).astype(np.float32)
        if STATE_DIMS[key] == 4:              # quaternions: unit norm
            v /= np.linalg.norm(v)
        state[key] = v
    return {"images": images, "state": state, "task_description": "Close the toaster oven door."}


def packet_obs(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {"images": {k: np.ascontiguousarray(d[k][0]) for k in VIDEO_KEYS},
            "state": {k: np.asarray(d[k][0]) for k in STATE_KEYS},
            "task_description": str(d["annotation.human.task_description"])}


async def main(args) -> int:
    obs = packet_obs(args.packet) if args.packet else synthetic_obs()
    h = PolicyHandle(args.url, load_profile("gr00t-n15-robocasa365"), timeout=600)
    await h.open({"task": {"name": "t1-check", "episode_idx": 0}})
    spec = h._conn.server_info.get("action_spec") or {}
    action_dim = sum(int(v["dims"]) for v in spec.values())
    layout = ", ".join("%s:%s" % (k, v["dims"]) for k, v in spec.items())
    print(f"server {h._conn.server_info.get('model_server')} action_spec dims {action_dim} ({layout})")
    results: dict[str, bool] = {}

    async def timed(seeds, readouts=()):
        t = time.monotonic()
        out = await h.sample(obs, seeds, readouts=readouts)
        return out, (time.monotonic() - t) * 1e3

    await h.sample(obs, [999])                        # warm-up (CUDA kernels, allocator)
    s, t, u = SEEDS
    (a1,), _ = await timed([s])
    (a2,), _ = await timed([s])
    trio, _ = await timed(SEEDS)
    (b,), _ = await timed([t])
    (c,), _ = await timed([u])
    a1, a2, b, c = a1.actions, a2.actions, b.actions, c.actions

    results["a same (obs, seed) twice is bitwise identical"] = a1.tobytes() == a2.tobytes()
    results["b isolated: each of [s, t, u] equals its single-seed request"] = all(
        x.actions.tobytes() == y.tobytes() for x, y in zip(trio, (a1, b, c)))
    results["d chunk shape [16, A] float32"] = a1.shape == (16, action_dim) and a1.dtype == np.float32
    results["d different seeds give different chunks"] = (
        not np.array_equal(a1, b) and not np.array_equal(a1, c) and not np.array_equal(b, c))
    print(f"chunk {a1.shape} {a1.dtype}; max |seed s - seed t| = {np.abs(a1 - b).max():.4f}; "
          f"trio seeds echoed {[p.extras.get('seed') for p in trio]}")

    k1, k8 = [], []
    for _ in range(args.reps):
        _, ms = await timed([s]); k1.append(ms)
        out8, ms = await timed(K8); k8.append(ms)
    results["a K=8 repeat is bitwise identical"] = all(
        x.actions.tobytes() == y.actions.tobytes() for x, y in zip(out8, await h.sample(obs, K8)))
    print(f"timing K=1: median {np.median(k1):.0f} ms (min {min(k1):.0f}); "
          f"K=8: median {np.median(k8):.0f} ms (min {min(k8):.0f}); {args.reps} reps")

    (v,), vms = await timed([s], readouts=("velocity",))
    vel = np.asarray(v.extras.get("velocity"))
    results["velocity readout leaves the chunk unchanged"] = v.actions.tobytes() == a1.tobytes()
    print(f"velocity readout {vel.shape} {vel.dtype}, timesteps {v.extras.get('velocity_timesteps')}, "
          f"{vms:.0f} ms")

    t0 = time.monotonic()
    native = await h.act(obs)
    native_ms = (time.monotonic() - t0) * 1e3
    results["e native request returns a normal action"] = (
        native.ndim == 2 and native.shape[1] == action_dim and np.isfinite(native).all())
    print(f"native reply {native.shape} {native.dtype} ({native_ms:.0f} ms); "
          f"equals seeded sample s: {np.array_equal(native, a1)}")
    await h.close({})

    chunks = {"s": a1, "t": b, "u": c, **{f"k8_{i}": p.actions for i, p in enumerate(out8)}}
    if args.save:
        np.savez(args.save, native=native, **chunks)
        print("saved", args.save)
    if args.compare:
        ref = np.load(args.compare)
        same = [k for k in chunks if k in ref.files and ref[k].tobytes() == chunks[k].tobytes()]
        results[f"c cross-process: {len(same)}/{len(chunks)} chunks bitwise equal to {args.compare.name}"] = (
            len(same) == len(chunks))
        if "native" in ref.files:  # meaningful only if both were their process's first native request
            print("native reply equal to the reference process's:", ref["native"].tobytes() == native.tobytes())

    for name, ok in results.items():
        print(("PASS " if ok else "FAIL ") + name)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://localhost:8105")
    p.add_argument("--packet", type=Path, help="GR00T-format npz packet instead of the synthetic observation")
    p.add_argument("--save", type=Path, help="save the chunks (npz) for a later --compare")
    p.add_argument("--compare", type=Path, help="chunks saved by another server process")
    p.add_argument("--reps", type=int, default=5)
    sys.exit(anyio.run(main, p.parse_args()))
