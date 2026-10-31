"""Parity check: the v2 AHSExecutor against HarnessStudio v1's chunktrust-ahs code on randomized inputs.

    .venv/bin/python methods/chunktrust_ahs/check_parity.py [--episodes 400] [--seed 0] [--v1 <dir>]

v1 is loaded from its own files (default: the harnessstudio-ext-g checkout, or $HARNESSSTUDIO_V1_CHUNKTRUST):
modules.py (AHSExecutor) and the horizon.py / chunktrust_gr00t_ahs.py next to it, with AHS_PARAMS and the
executor params read from v1's __init__.py. Each case is one episode of 1..15 (70%) or 16..60 (30%) decisions with a fresh
executor pair; every decision feeds both sides the same random GR00T-shaped velocity trace [4, 16, 32] and
16-step 12-D action chunk (v1 in its flat GR00T column order, v2 in the vla-eval wire order, through the
bound action layout), then records the same executed prefix (candidate-style k, episode-end truncation, or
null-style 16). Checks per decision: k and the full AHS info dict are equal (exact, via JSON). Per episode:
the final episode state (history, Beta posterior, rng state) is equal. Also: the vendored files are
byte-identical to v1's and match PROVENANCE.json, the parameters equal v1's, and fake-stack shapes
(7-D, 6-step chunks, velocity [4, 6, 7]) agree. Informational: how often the v2 executor would differ
from v1 without the column permutation (column_order='native').

Exit code 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from harnessstudio.context.episode import Embodiment

import hs_chunktrust_ahs as v2
from hs_chunktrust_ahs import ahs as v2_ahs

DEFAULT_V1 = "/home/sejain/research/cvpr27/harnessstudio-ext-g/harnessstudio/methods/chunktrust_ahs"
VENDORED = ("chunktrust_gr00t_ahs.py", "horizon.py", "CHUNKTRUST_LICENSE", "PROVENANCE.json")
GR00T_SPEC = {"position": {"dims": 3}, "rotation": {"dims": 3}, "gripper": {"dims": 1},
              "base_motion": {"dims": 4}, "control_mode": {"dims": 1}}   # vla-eval GR00T action_spec order
# v1 flat columns: base_motion 0:4, control_mode 4:5, position 5:8, rotation 8:11, gripper 11:12.
WIRE_FROM_V1 = np.array([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])
FAKE_SPEC = {"position": {"dims": 3}, "rotation": {"dims": 3}, "gripper": {"dims": 1}}


class Chunk:  # v1 passed objects with .actions
    def __init__(self, actions):
        self.actions = actions


def load_v1(v1_dir: Path):
    init_src = (v1_dir / "__init__.py").read_text()
    tree = ast.parse(init_src)
    params = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "AHS_PARAMS" for t in node.targets))
    wiring = '{"min_k": 4, "per_step": False, "ahs_params": AHS_PARAMS,\n                                      "velocity_dims": "all"}'
    if wiring not in init_src:
        raise SystemExit("v1 __init__.py executor params changed; update this script")
    spec = importlib.util.spec_from_file_location("v1_chunktrust_modules", v1_dir / "modules.py")
    modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modules)   # puts v1_dir on sys.path; AHSExecutor imports v1's horizon by name
    ex = modules.AHSExecutor(min_k=4, per_step=False, ahs_params=params, velocity_dims="all")
    v1_horizon = sys.modules["horizon"]
    for mod in (v1_horizon, sys.modules["chunktrust_gr00t_ahs"]):
        if Path(mod.__file__).resolve().parent != v1_dir.resolve():
            raise SystemExit(f"{mod.__name__} was not loaded from the v1 directory: {mod.__file__}")
    del ex
    return modules, params, v1_horizon


def rand_velocity(rng, h=16, d=32, steps=4):
    kind = rng.integers(0, 5)
    scale = float(10 ** rng.uniform(-3, 1))
    if kind == 0:    # iid
        v = rng.normal(0, scale, (steps, h, d))
    elif kind == 1:  # smooth along the horizon + noise (low spectral energy)
        v = np.cumsum(rng.normal(0, scale, (steps, h, d)), axis=1) + rng.normal(0, scale * rng.uniform(0, 0.5), (steps, h, d))
    elif kind == 2:  # high-frequency alternation
        sign = (-1.0) ** np.arange(h)[None, :, None]
        v = sign * rng.normal(scale, scale * 0.3, (steps, 1, d)) + rng.normal(0, scale * 0.2, (steps, h, d))
    elif kind == 3:  # denoising-step trend (exercises the window-RMS z)
        trend = np.linspace(rng.uniform(0, 2), rng.uniform(0, 2), steps)[:, None, None]
        v = trend * rng.normal(0, scale, (1, h, d)) + rng.normal(0, scale * rng.uniform(0, 1), (steps, h, d))
    else:            # mixture across dims
        v = np.concatenate([rng.normal(0, scale, (steps, h, d // 2)),
                            np.cumsum(rng.normal(0, scale, (steps, h, d - d // 2)), axis=1)], axis=2)
    if d == 32 and rng.random() < 0.6:  # GR00T padding dims 12:32: constant, zero or small noise
        mode = rng.integers(0, 3)
        v[:, :, 12:] = 0.0 if mode == 0 else (rng.normal(0, 0.4) if mode == 1 else rng.normal(0, 0.4, v[:, :, 12:].shape))
    if rng.random() < 0.05:
        v[rng.integers(0, steps)] = 0.0
    return v.astype(np.float32)


def rand_actions_v1(rng, h=16):
    """[h, 12] in v1 column order: base_motion 4, control_mode 1, position 3, rotation 3, gripper 1."""
    kind = rng.integers(0, 4)
    a = np.zeros((h, 12))
    if kind == 3:   # constant chunk (zero speed)
        a[:] = rng.normal(0, 0.3, 12)
    else:
        step = rng.normal(0, 10 ** rng.uniform(-3, -0.5), (h, 6))
        drift = rng.normal(0, 0.3, 6)
        a[:, 5:11] = drift + (np.cumsum(step, axis=0) if kind == 1 else step)
        a[:, 0:4] = rng.normal(0, 0.05, (h, 4)) * (rng.random() < 0.3)
        a[:, 4] = float(rng.integers(0, 2)) * 2 - 1
        g = np.where(rng.random() < 0.5, -1.0, 1.0)
        a[:, 11] = g if rng.random() < 0.8 else np.where(np.arange(h) < rng.integers(0, h), g, -g)
        if kind == 2:
            a[:, 5:11] = np.clip(a[:, 5:11], -1, 1)
    return a.astype(np.float32)


def canon(info) -> str:
    return json.dumps(info, sort_keys=True, allow_nan=True)


def state(ex) -> str:
    return canon(ex.ahs.state_dict())


def main(args) -> int:
    ok = True
    v1_dir = Path(args.v1)
    here = Path(v2.__file__).resolve().parent

    # 1. vendored files and provenance
    prov = json.loads((here / "PROVENANCE.json").read_text())
    for name in VENDORED:
        same = (here / name).read_bytes() == (v1_dir / name).read_bytes()
        print(f"{'PASS' if same else 'FAIL'} vendored {name} byte-identical to v1")
        ok &= same
    for name, digest in prov["files"].items():
        good = hashlib.sha256((here / name).read_bytes()).hexdigest() == digest
        print(f"{'PASS' if good else 'FAIL'} {name} sha256 matches PROVENANCE.json")
        ok &= good

    # 2. v1 code and parameters
    v1_modules, v1_params, v1_horizon = load_v1(v1_dir)
    v2_horizon = v2_ahs.horizon
    assert Path(v2_horizon.__file__).resolve().parent == here and v2_horizon is not v1_horizon
    same_params = (v2.AHS_PARAMS == v1_params == v2_horizon.REFERENCE_AHS_PARAMS
                   and v2.EXECUTOR_PARAMS["min_k"] == 4 and v2.EXECUTOR_PARAMS["velocity_dims"] == "all")
    cfg_params = v2.config().slots["executor"].params
    same_params &= cfg_params["ahs_params"] == v1_params and v2_horizon.canonical_sha256(cfg_params["ahs_params"]) \
        == v1_horizon.canonical_sha256(v1_params)
    print(f"{'PASS' if same_params else 'FAIL'} AHS params equal v1 (sha256 {v2.AHS_PARAMS_SHA256[:12]}...)")
    ok &= same_params

    groot = Embodiment.from_action_spec(GR00T_SPEC)
    fake = Embodiment.from_action_spec(FAKE_SPEC)

    def new_v2(embodiment, column_order="v1"):
        ex = v2.AHSExecutor(**{**cfg_params, "column_order": column_order})
        ex.bind(embodiment)
        return ex

    probe = new_v2(groot)
    perm_ok = probe._perm is not None and np.array_equal(np.argsort(WIRE_FROM_V1), np.asarray(probe._perm))
    print(f"{'PASS' if perm_ok else 'FAIL'} GR00T layout binds the v1 column order {[int(c) for c in probe._perm]}")
    ok &= perm_ok

    # 3. randomized GR00T-shaped episodes
    rng = np.random.default_rng(args.seed)
    n_dec = n_k = n_info = n_state = 0
    native_diff_k = native_diff_info = 0
    ks, modes, n_inter = [], {}, 0
    for ep in range(args.episodes):
        e1 = v1_modules.AHSExecutor(min_k=4, per_step=False, ahs_params=v1_params, velocity_dims="all")
        e2 = new_v2(groot)
        e3 = new_v2(groot, column_order="native")   # informational ablation
        null_style = rng.random() < 1 / 3
        n_decisions = int(rng.integers(1, 16)) if rng.random() < 0.7 else int(rng.integers(16, 61))
        for _ in range(n_decisions):
            vel = rand_velocity(rng)
            flat1 = rand_actions_v1(rng)
            wire = np.ascontiguousarray(flat1[:, WIRE_FROM_V1])
            extras = {"seed": 0, "velocity": vel, "timesteps": [0, 250, 500, 750], "native_length": 16}
            k1 = e1.plan(Chunk(flat1), {"sampler": {"velocity": vel}}, None)
            k2 = e2.plan(wire.copy(), dict(extras), {})
            k3 = e3.plan(wire.copy(), dict(extras), {})
            n_dec += 1
            n_k += k1 == k2
            n_info += canon(e1.last_info) == canon(e2.last_info)
            native_diff_k += k1 != k3
            native_diff_info += canon(e1.last_info) != canon(e3.last_info)
            ks.append(k2)
            modes[e2.last_info["choose_mode"]] = modes.get(e2.last_info["choose_mode"], 0) + 1
            n_inter += any(p is not None for p in e2.last_info["p_inter"])
            if null_style:
                executed = 16
            elif rng.random() < 0.1:
                executed = int(rng.integers(1, k1 + 1))   # episode ended mid-chunk
            else:
                executed = k1
            e1.record_executed(flat1[:executed])
            e2.record_executed(wire[:executed])
            e3.record_executed(wire[:executed])
        n_state += state(e1) == state(e2)
    agree = n_k == n_dec and n_info == n_dec and n_state == args.episodes
    print(f"{'PASS' if agree else 'FAIL'} GR00T shapes: {args.episodes} episodes, {n_dec} decisions: "
          f"k equal {n_k}/{n_dec}, info equal {n_info}/{n_dec}, final state equal {n_state}/{args.episodes}")
    print(f"     k distribution {{{', '.join(f'{int(k)}: {int(c)}' for k, c in zip(*np.unique(ks, return_counts=True)))}}}; "
          f"choose modes {modes}; speed-uniformity term active on {n_inter}/{n_dec} decisions")
    print(f"INFO without the column permutation (column_order='native'): k differs on {native_diff_k}/{n_dec}, "
          f"info differs on {native_diff_info}/{n_dec} decisions")
    ok &= agree

    # 4. fake-stack shapes (7-D, 6-step chunks, velocity [4, 6, 7]): native order on both sides
    f_dec = f_eq = 0
    for ep in range(50):
        e1 = v1_modules.AHSExecutor(min_k=4, per_step=False, ahs_params=v1_params, velocity_dims="all")
        e2 = new_v2(fake)
        assert e2._perm is None
        for _ in range(int(rng.integers(1, 10))):
            vel = rand_velocity(rng, h=6, d=7)
            chunk = rng.uniform(-1, 1, (6, 7)).astype(np.float32)
            k1 = e1.plan(Chunk(chunk), {"sampler": {"velocity": vel}}, None)
            k2 = e2.plan(chunk.copy(), {"velocity": vel, "native_length": 6}, {})
            f_dec += 1
            f_eq += k1 == k2 and canon(e1.last_info) == canon(e2.last_info)
            executed = 6 if rng.random() < 0.3 else k1
            e1.record_executed(chunk[:executed])
            e2.record_executed(chunk[:executed])
        f_eq_state = state(e1) == state(e2)
        ok &= f_eq_state
    print(f"{'PASS' if f_eq == f_dec else 'FAIL'} fake shapes: {f_dec} decisions equal {f_eq}/{f_dec}")
    ok &= f_eq == f_dec

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--v1", default=os.environ.get("HARNESSSTUDIO_V1_CHUNKTRUST", DEFAULT_V1))
    sys.exit(main(p.parse_args()))
