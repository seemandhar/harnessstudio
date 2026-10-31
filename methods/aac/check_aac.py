"""Standalone behavioural checks for hs_aac, ported from HarnessStudio v1 tests/test_aac.py (branch ext/aac).

Sections run when their dependencies are present, otherwise they are reported as SKIP:
  upstream  needs scipy and the pinned author checkout (HARNESSSTUDIO_AAC_UPSTREAM); numpy-only, no harnessstudio
  executor  needs harnessstudio (+ hs_aac installed)
  arms      needs harnessstudio + vla-eval: real in-process episodes on the fake stack, chunk lengths 6 and 16

    .venv/bin/python methods/aac/check_aac.py                       # executor + arms (the v2 venv has no scipy)
    /path/to/python-with-scipy methods/aac/check_aac.py             # upstream (e.g. the gr00t conda env)
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get(
    "HARNESSSTUDIO_AAC_UPSTREAM",
    "/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/hs-rc365-paper-20261029/paper-repro/upstream/aac-robocasa"))
UPSTREAM_SHA256 = {"robocasa/demos/action_optimization/action_entropy_v2.py":
                   "80abe7e1c316c81694b24f9efb9517f89406ec5b2a96a11c49abc0c217d26355",
                   "robocasa/demos/help_function.py":
                   "056cf3830d1c238693cf053fc8694588915201051da668509f44b18be66b9ffc"}
UP_FUNCS = {"action_entropy_v2.py": ["_safe_logdet_cov", "gaussian_entropy_from_samples", "bernoulli_entropy_from_samples",
                                     "action_entropy_gausssian_bernoulli", "action_magnitude", "find_first_idx",
                                     "select_chunk_size"],
            "help_function.py": ["axisangle2quat", "quat_multiply", "quat2axisangle", "compose_rotations"]}
# v2 GR00T N1.5 RoboCasa365 wire layout (vla-eval robocasa365 ACTION_COMPONENTS / get_action_spec order)
V2_LAYOUT = {"position": (0, 3), "rotation": (3, 6), "gripper": (6, 7), "base_motion": (7, 11), "control_mode": (11, 12)}
# v1 GR00T sorted-key layout with GR00T names (tests/test_aac.py GR00T_LAYOUT)
V1_LAYOUT = {"action.base_motion": (0, 4), "action.control_mode": (4, 5), "action.end_effector_position": (5, 8),
             "action.end_effector_rotation": (8, 11), "action.gripper_close": (11, 12)}
METADATA = Path("/home/sejain/research/robo-scientist/ckpts/gr00t_n1-5/multitask_learning/checkpoint-120000/"
                "experiment_cfg/metadata.json")

RESULTS: list[tuple[str, str, str]] = []   # (name, PASS|FAIL|SKIP, detail)


def check(name):
    def wrap(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, "PASS", detail))
        except _Skip as exc:
            RESULTS.append((name, "SKIP", str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"))
        return fn
    return wrap


class _Skip(Exception):
    pass


def load_aac():
    """hs_aac/aac.py by path (numpy only), so the numerics run in environments without harnessstudio."""
    spec = importlib.util.spec_from_file_location("hs_aac_numerics", HERE / "hs_aac" / "aac.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


aac = load_aac()


def upstream_namespace():
    """Load ONLY the named pure-numpy function definitions from the pinned, hash-checked upstream files."""
    try:
        from scipy.stats import entropy
    except ImportError:
        raise _Skip("scipy unavailable (run with a python that has numpy + scipy, e.g. the gr00t env)")
    files = {"action_entropy_v2.py": UPSTREAM / "robocasa/demos/action_optimization/action_entropy_v2.py",
             "help_function.py": UPSTREAM / "robocasa/demos/help_function.py"}
    if not all(p.is_file() for p in files.values()):
        raise _Skip(f"upstream checkout not found at {UPSTREAM}")
    for rel, digest in UPSTREAM_SHA256.items():
        if hashlib.sha256((UPSTREAM / rel).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"upstream {rel} differs from the pinned commit")
    ns = {"np": np, "math": math, "entropy": entropy}
    for name, path in files.items():
        tree = ast.parse(path.read_text())
        wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in UP_FUNCS[name]]
        assert {n.name for n in wanted} == set(UP_FUNCS[name])
        exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), ns)
    return ns


def random_case(rng, n=20, t=16, kind="mixed"):
    """eef (N, T, 6) float32 in [-1, 1], binary gripper (N, T, 1); flats in the v2 and v1 GR00T layouts."""
    scale = {"mixed": 0.4, "tight": 0.01, "wide": 1.0}[kind]
    center = rng.uniform(-0.6, 0.6, size=(1, t, 6))
    drift = np.cumsum(rng.normal(0, 0.05, size=(1, t, 6)), axis=1)
    eef = (center + drift + rng.normal(0, scale, size=(n, t, 6)) * np.linspace(0.2, 1.5, t)[None, :, None])
    eef = np.clip(eef, -1, 1).astype(np.float32)
    p_close = rng.uniform(0, 1)
    grip = (rng.uniform(0, 1, size=(n, t, 1)) < p_close).astype(np.float32)
    if kind == "tight":
        eef[:] = eef[0:1]          # identical samples: singular covariance, eps regularisation
    base = rng.normal(0, 1, size=(n, t, 4)).astype(np.float32)
    mode = (rng.uniform(0, 1, size=(n, t, 1)) < 0.5).astype(np.float32)
    v2 = np.concatenate([eef[:, :, :3], eef[:, :, 3:], grip, base, mode], axis=2)
    v1 = np.concatenate([base, mode, eef[:, :, :3], eef[:, :, 3:], grip], axis=2)
    return eef, grip, v2, v1


def executor(layout=V2_LAYOUT, **override):
    ex = aac.AACEntropyExecutor(**{**aac.PARAMS, **override})
    if layout is not None:
        ex.bind(SimpleNamespace(action_layout=layout))
    return ex


def plan(ex, flats):
    """As the v2 DecisionLoop calls it: chunk = np.array(sample 0), proposals = read-only arrays, seed order."""
    proposals = []
    for f in flats:
        a = np.array(f)
        a.setflags(write=False)
        proposals.append(a)
    return ex.plan(np.array(proposals[0]), {"proposals": proposals}, {})


# ------------------------------------------------------------------ upstream equality (v1 PortEqualsUpstream)

@check("upstream: select_chunk_size k, per-step entropies and chunk means equal the author code (600 cases)")
def _():
    up = upstream_namespace()
    rng = np.random.default_rng(20261030)
    ex, ex_v1 = executor(), executor(V1_LAYOUT)
    weights = dict(move_th=3.0, entropy_weights={"position": 1.0, "rotation": 1.0, "gripper": 1.0},
                   magnitude_weights={"position": 1, "rotation": 1, "gripper_toggle": 0.2}, eps=1e-6)
    ks = []
    for i in range(600):
        eef, grip, v2, v1 = random_case(rng, kind=("mixed", "tight", "wide")[i % 3])
        normalized = np.concatenate([eef, grip, np.zeros((20, 16, 5), np.float32)], axis=2)
        d = {"normalized_action": normalized, "action.end_effector_position": eef[:, :, :3],
             "action.end_effector_rotation": eef[:, :, 3:], "action.gripper_close": grip}
        k_up, br = up["select_chunk_size"](d, "gaussian_bernoulli", 3.0)
        k_port, info = aac.select_chunk_size(eef[:, :, :3], eef[:, :, 3:], grip, eef[0, :, :3], eef[0, :, 3:],
                                             grip[0], **weights)
        assert k_port == k_up, (i, k_port, k_up)
        np.testing.assert_array_equal(np.asarray(info["H_total"]), np.asarray(br["total"]))
        assert info["chunk_mean"] == br["chunk_mean"], i
        assert info["motion"] == up["action_magnitude"](d), i
        # executor (name-resolved columns + min_max re-normalization with identity stats): same k, both layouts
        assert plan(ex, v2) == k_up, (i, "v2 layout")
        assert plan(ex_v1, v1) == k_up, (i, "v1 GR00T-key layout")
        ks.append(k_up)
    return f"k histogram {dict(sorted((int(k), ks.count(k)) for k in set(ks)))}"


@check("upstream: quaternion helpers (compose_rotations) bitwise equal (200 cases)")
def _():
    up = upstream_namespace()
    rng = np.random.default_rng(1)
    for _ in range(200):
        r = rng.normal(0, 0.3, size=(int(rng.integers(1, 16)), 3)).astype(np.float32)
        np.testing.assert_array_equal(aac.compose_rotations(r), up["compose_rotations"](r))


@check("upstream: bernoulli entropy equals scipy.stats.entropy")
def _():
    up = upstream_namespace()
    for g in ([0] * 20, [1] * 20, [0, 1] * 10, [1] + [0] * 19, [-1, 1] * 10):
        assert aac.bernoulli_entropy(np.array(g, np.float32)) == up["bernoulli_entropy_from_samples"](np.array(g, np.float32))


# ------------------------------------------------------------------ executor behaviour (v1 ExecutorBehaviour)

def need_harnessstudio():
    try:
        import harnessstudio  # noqa: F401
        import hs_aac  # noqa: F401
    except ImportError as exc:
        raise _Skip(f"harnessstudio/hs_aac not importable: {exc}")


@check("executor: k in [2, 16], k = max(k_entropy, k_motion), finite JSON diagnostics <= 4096 B (100 cases)")
def _():
    rng = np.random.default_rng(7)
    ex = executor()
    for i in range(100):
        _, _, v2, _ = random_case(rng, kind=("mixed", "tight", "wide")[i % 3])
        k = plan(ex, v2)
        assert 2 <= k <= 16, k
        info = ex.diagnostics()
        assert len(json.dumps(info, allow_nan=False).encode()) <= 4096
        assert info["k"] == k == max(info["k_entropy"], info["k_motion"])
        assert info["n"] == 20 and info["T"] == 16 and info["proposal_shape"] == [16, 12]
        assert info["proposal0_equals_chunk"] and info["proposal_dtype"] == "float32"


@check("executor: groups by name; base_motion / control_mode columns do not change k (50 cases)")
def _():
    rng = np.random.default_rng(11)
    ex = executor()
    for _ in range(50):
        _, _, v2, _ = random_case(rng)
        k1 = plan(ex, v2)
        v2b = v2.copy()
        v2b[:, :, 7:12] = rng.normal(0, 3, size=v2b[:, :, 7:12].shape)
        assert plan(ex, v2b) == k1


@check("executor: GR00T v1 key names and an explicit dims override resolve to the same columns as bind()")
def _():
    rng = np.random.default_rng(5)
    a, b = executor(), executor(V1_LAYOUT)
    c = executor(None, dims={"position": [0, 3], "rotation": [3, 6], "gripper": [6, 7]})
    d = executor({"actions": (0, 12)}, dims={"position": [5, 8], "rotation": [8, 11], "gripper": [11, 12]})  # dims wins
    for _ in range(30):
        _, _, v2, v1 = random_case(rng)
        assert plan(a, v2) == plan(b, v1) == plan(c, v2) == plan(d, v1)
    assert a.columns() == c.columns()


@check("executor: D5 gripper-toggle term is inert for (T, 1) gripper arrays")
def _():
    pos, rot = np.zeros((16, 3), np.float32), np.zeros((16, 3), np.float32)
    grip = np.array([[0], [1]] * 8, np.float32)
    assert aac.action_magnitude(pos, rot, grip, {"position": 1, "rotation": 1, "gripper_toggle": 0.2}) == [0.0] * 15


@check("executor: refuses without proposals, with a missing/odd-width group, or with no layout and no dims")
def _():
    _, _, v2, _ = random_case(np.random.default_rng(3))
    chunk = v2[0]

    def raises(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("expected ValueError")
    raises(lambda: executor().plan(chunk, {}, {}))
    raises(lambda: executor().plan(chunk, {"proposals": [chunk]}, {}))
    raises(lambda: plan(executor({k: v for k, v in V2_LAYOUT.items() if k != "rotation"}), v2))
    raises(lambda: plan(executor({"position": (0, 3), "rotation": (3, 7), "gripper": (7, 8)}), v2))  # quaternion
    raises(lambda: plan(executor(None), v2))
    raises(lambda: aac.AACEntropyExecutor(**{**aac.PARAMS, "min_k": 1}))
    raises(lambda: aac.AACPool(1))


@check("executor: short chunks are well defined (T = 1..16), strict_horizon restores v1's T=16 refusal")
def _():
    rng = np.random.default_rng(13)
    ex = executor()
    seen = {}
    for t in range(1, 17):
        for i in range(30):
            _, _, v2, _ = random_case(rng, t=t, kind=("mixed", "tight", "wide")[i % 3])
            k = plan(ex, v2)
            info = ex.diagnostics()
            assert (k == t) if t < 2 else (2 <= k <= t), (t, k)
            assert info["T"] == t
            seen.setdefault(t, set()).add(k)
    _, _, v6, _ = random_case(rng, t=6)
    strict = executor(strict_horizon=True)
    try:
        plan(strict, v6)
        raise AssertionError("strict_horizon accepted T=6")
    except ValueError:
        pass
    _, _, v16, _ = random_case(rng)
    assert plan(strict, v16) == plan(ex, v16)
    return "k values seen: " + ", ".join(f"T={t}:{sorted(ks)}" for t, ks in seen.items() if t in (1, 2, 3, 6, 16))


@check("config: N=20, executes sample 0 (no selector), faithful with no substitutions, D1-D7 disclosed")
def _():
    need_harnessstudio()
    import hs_aac
    from harnessstudio.core.method import validate
    m = validate(hs_aac.config())
    assert m.name == "aac" and m.version == "1"
    assert m.slots["proposer"].params["num_samples"] == 20
    assert m.slots["executor"].params["min_k"] == 2 and m.slots["executor"].params["move_th"] == 3.0
    assert "selector" not in m.slots and "action_op" not in m.slots
    assert m.fidelity.value == "faithful" and m.substitutions == ()
    assert [d["id"] for d in m.card["disclosures"]] == [f"D{i}" for i in range(1, 8)]
    assert m.categories == ("uncertainty_compute", "execution_monitoring")
    assert m.required == frozenset({"act", "chunk", "seeded_sampling", "isolated_streams"})
    ex = m.slots["executor"].build()
    assert ex.sees_proposals is True and ex.per_step is False and ex.min_k == 2
    if METADATA.is_file():
        stats = json.loads(METADATA.read_text())["new_embodiment"]["statistics"]["action"]
        for role, key in (("position", "end_effector_position"), ("rotation", "end_effector_rotation")):
            for s in ("min", "max"):
                assert stats[key][s] == aac.NORMALIZATION[role][s], (role, s)
        return f"fingerprint {m.fingerprint()[:12]}; normalization constants match the pinned metadata.json"
    return f"fingerprint {m.fingerprint()[:12]}"


# ------------------------------------------------------------------ real episodes (v1 test_synthetic_arms)

PLANS: list = []          # (executor serial, planned k, diagnostics) in call order, filled by RecordingExecutor
_SERIAL = iter(range(10**9))


class RecordingExecutor(aac.AACEntropyExecutor):
    def __init__(self, **params):
        super().__init__(**params)
        self.serial = next(_SERIAL)   # one executor per episode (ids of collected objects can be reused)

    def plan(self, chunk, extras, obs):
        k = super().plan(chunk, extras, obs)
        PLANS.append((self.serial, k, self.diagnostics()))
        return k


def run_arms(chunk_len, horizon):
    import dataclasses

    import hs_aac
    from harnessstudio.contracts import load_contract
    from harnessstudio.core.arms import Arm
    from harnessstudio.core.method import ModuleRef
    from harnessstudio.policy_ext.fake import FakePolicyServer
    from harnessstudio.profiles import load_profile
    from harnessstudio.server.harness_server import HarnessServer
    from harnessstudio.verify.stack import run_fake, serve

    base = hs_aac.config()
    cfg = dataclasses.replace(base, slots={**base.slots, "executor": ModuleRef(f"{__name__}:RecordingExecutor",
                                                                                base.slots["executor"].params)})
    out = {}
    for arm in Arm:
        PLANS.clear()
        policy = FakePolicyServer(chunk_len=chunk_len, chunking="harness", seeded=True)
        with serve(policy) as ph:
            # the contract horizon must equal the benchmark horizon (the harness ends control at the contract's)
            contract = dataclasses.replace(load_contract("fake"), default_horizon=horizon)
            hs = HarnessServer(cfg, policy_url=ph.url, contract=contract, arm=arm,
                               profile=load_profile("fake-t1"))
            with serve(hs) as hh:
                run_fake(hh.url, n_tasks=2, episodes=2, horizon=horizon)
        plans = {}
        for inst, k, info in PLANS:
            plans.setdefault(inst, []).append((k, info))
        out[arm] = (hs.records, list(plans.values()))
    return out


def executed_per_decision(record):
    starts = [e["step"] for e in record["events"] if e["t"] == "sample"]
    ends = starts[1:] + [record["accounting"]["steps"]]
    return [b - a for a, b in zip(starts, ends)]


def arms_check(chunk_len, horizon):
    need_harnessstudio()
    from harnessstudio.core.arms import Arm
    out = run_arms(chunk_len, horizon)
    (brec, _), (nrec, nplans), (crec, cplans) = out[Arm.BASELINE], out[Arm.NULL], out[Arm.CANDIDATE]
    for recs in (brec, nrec, crec):
        assert len(recs) == 4 and all(r["valid"] for r in recs), [r["invalid_reason"] for r in recs]
    key = lambda recs: {(r["task"], r["episode_idx"]): r["action_sha256"] for r in recs}  # noqa: E731
    assert key(brec) == key(nrec), "derived null differs from baseline"
    for r in nrec:   # the null pays N=20 samples per decision and runs sample 0 natively
        # (accounting.decisions misses the last decision, which is cut by the episode end; count sample calls)
        ex = executed_per_decision(r)
        assert r["accounting"]["method_policy_samples"] == 20 * len(ex), r["accounting"]
        assert all(e == chunk_len for e in ex[:-1]), ex
    assert sum(len(p) for p in nplans) == sum(len(executed_per_decision(r)) for r in nrec)  # executor ran in null
    assert all(info["proposal0_equals_chunk"] and info["n"] == 20 for p in nplans + cplans for _, info in p)
    executed_all, planned_all = [], []
    unmatched = [list(p) for p in cplans]
    for r in crec:
        ex = executed_per_decision(r)
        assert r["accounting"]["method_policy_samples"] == 20 * len(ex)
        match = next((p for p in unmatched if len(p) == len(ex)
                      and all(e == k for e, (k, _) in zip(ex[:-1], p[:-1])) and ex[-1] <= p[-1][0]), None)
        assert match is not None, f"executed {ex} matches no executor plan sequence"
        unmatched.remove(match)
        executed_all += ex[:-1]
        planned_all += [k for k, _ in match]
    assert all(2 <= k <= chunk_len for k in planned_all), planned_all
    ke = [info["k_entropy"] for p in cplans for _, info in p]
    km = [info["k_motion"] for p in cplans for _, info in p]
    hist = lambda v: dict(sorted((int(x), v.count(x)) for x in set(v)))  # noqa: E731
    n_cost = sum(r["accounting"]["method_policy_samples"] for r in nrec)
    c_cost = sum(r["accounting"]["method_policy_samples"] for r in crec)
    return (f"null = baseline actions; candidate k {hist(planned_all)}, k_entropy {hist(ke)}, k_motion {hist(km)}; "
            f"samples null {n_cost} / candidate {c_cost}")


@check("arms on the fake stack, T=6 (tier-1 shape): null = baseline, null pays 20/decision, candidate runs planned k")
def _():
    return arms_check(6, 12)


@check("arms on the fake stack, T=16 (GR00T chunk length), horizon 40")
def _():
    return arms_check(16, 40)


if __name__ == "__main__":
    width = max(len(n) for n, _, _ in RESULTS)
    for name, status, detail in RESULTS:
        print(f"[{status:4}] {name}")
        if detail:
            print("        " + detail.replace("\n", "\n        ").rstrip())
    counts = {s: sum(1 for _, x, _ in RESULTS if x == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    sys.exit(1 if counts["FAIL"] else 0)
