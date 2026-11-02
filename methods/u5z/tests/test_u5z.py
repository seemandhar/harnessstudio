"""hs_u5z unit tests (CPU): byte identity of the reused frozen components, no new scalars, construction refusals,
synthetic precedence cases of A6.Z.5 (zero chunk at / before U5's arming, Z cap 2, abstaining U5 on t8, disarm, call
order and native routing, 3-run determinism) for both entries, Zsign construction (C5) and band-guard isolation."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import precedence_checks as P
from harnessstudio.core import registry
from hs_u5z import glue

ROOT = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude")
ENTRIES = {"u5z-t": "tilt-guard-v12-z", "u5zsign-t": "tilt-guard-v11-zsign"}


def factory(name):
    ref = registry.load(name).slots["action_op"]
    return lambda: ref.build()


# --- byte identity / no new scalars ---------------------------------------------------------------------------------
def test_component_files_match_their_freeze_manifests():
    for name, comp in glue.COMPONENTS.items():
        mpath = ROOT / comp["manifest"]
        assert hashlib.sha256(mpath.read_bytes()).hexdigest() == comp["manifest_sha256"], name
        man = json.loads(mpath.read_text())
        pkg_prefix = comp["package"] + "/"
        for rel, sha in comp["files"].items():
            assert man["files"][pkg_prefix + rel] == sha, (name, rel)
        root = glue.verify_component(name)
        assert root == Path(man["package"]).resolve() / comp["package"], (name, root)


def test_z_and_zsign_share_the_frozen_action_op_bytes():
    z, zs = glue.COMPONENTS["tilt-guard-v12-z"]["files"], glue.COMPONENTS["tilt-guard-v11-zsign"]["files"]
    assert z["action_op.py"] == zs["action_op.py"] == "b4348f2ff28649ec20cde6e2350bc5f960e4b8fbef1100e1861a16aeec17471b"
    assert z["geometry.py"] == zs["geometry.py"]


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_params_equal_registered_component_configs(name):
    p = registry.load(name).slots["action_op"].params
    assert set(p) == {"variant", "u5", "z"} and p["variant"] == name
    ts = registry.load("tube-steer").slots["action_op"]
    zc = registry.load(ENTRIES[name]).slots["action_op"]
    assert p["u5"] == {"component": "tube-steer", "factory": ts.factory, "params": ts.params}
    assert p["z"] == {"component": ENTRIES[name], "factory": zc.factory, "params": zc.params}


def test_zsign_params_equal_paper_z_except_zero_and_inert_guard_fields():
    zs = registry.load("tilt-guard-v11-zsign").slots["action_op"].params
    z = registry.load("tilt-guard-v12-z").slots["action_op"].params
    glue.check_zsign_params(zs, z)
    diff = sorted(k for k in set(zs) | set(z) if zs.get(k) != z.get(k))
    assert diff == ["guard_factory", "guard_params", "zero"], diff
    bad = dict(zs, zero_max_consecutive=3)
    with pytest.raises(RuntimeError):
        glue.check_zsign_params(bad, z)
    with pytest.raises(RuntimeError):
        glue.check_zsign_params(dict(zs, guard=True), z)


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_construction_refuses_changed_params_factory_or_variant(name):
    params = registry.load(name).slots["action_op"].params
    p = copy.deepcopy(params)
    p["u5"]["params"]["eps_deg"] = 14.0
    with pytest.raises(RuntimeError):
        glue.U5ZActionOp(**p)
    p = copy.deepcopy(params)
    p["z"]["params"]["zero_max_consecutive"] = 3
    with pytest.raises(RuntimeError):
        glue.U5ZActionOp(**p)
    p = copy.deepcopy(params)
    p["z"]["factory"] = "hs_tilt_guard_v12.action_op:TiltGuardActionOp"
    with pytest.raises(RuntimeError):
        glue.U5ZActionOp(**p)
    p = copy.deepcopy(params)
    p["variant"] = "u5z-tv"
    with pytest.raises(ValueError):
        glue.U5ZActionOp(**p)


def test_construction_refuses_a_component_file_that_differs_from_its_pin(monkeypatch):
    comps = copy.deepcopy(glue.COMPONENTS)
    comps["tube-steer"]["files"]["action_op.py"] = "0" * 64
    monkeypatch.setattr(glue.U5ZActionOp, "COMPONENTS", comps)
    with pytest.raises(RuntimeError, match="pinned sha256"):
        factory("u5z-t")()
    comps = copy.deepcopy(glue.COMPONENTS)
    comps["tube-steer"]["files"]["action_op.py"] = "6d3b1e1ce17956b2934a6572ef091702dbf5327b5b3977f154a3a2542e25e7a5"
    comps["tilt-guard-v12-z"]["files"]["composite.py"] = "f" * 64
    monkeypatch.setattr(glue.U5ZActionOp, "COMPONENTS", comps)
    with pytest.raises(RuntimeError, match="pinned sha256"):
        factory("u5z-t")()


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_built_components_are_the_frozen_classes(name):
    op = factory(name)()
    assert type(op.u5).__module__ == "hs_tube_steer.action_op" and type(op.u5).__name__ == "TubeSteerActionOp"
    if name == "u5z-t":
        assert (type(op.z).__module__, type(op.z).__name__) == ("hs_tilt_guard_v12.composite", "TiltGuardV12ActionOp")
        assert op.z.zero == "substitute"
    else:
        assert (type(op.z).__module__, type(op.z).__name__) == ("hs_tilt_guard_v11.action_op", "TiltGuardActionOp")
        assert op.z.zero == "signctrl"
    assert op.z.guard is None and op.z.use_guard is False and op.z.envelope == "off" and op.z.sat_arming == "off"
    assert op.u5.mode == "steer" and op.u5.first_arm is None


def test_band_guard_never_imported_in_a_clean_interpreter():
    code = ("import sys, numpy as np\nfrom harnessstudio.core import registry\n"
            "sys.path.insert(0, %r)\nimport precedence_checks as P\n"
            "for n in ('u5z-t', 'u5zsign-t'):\n"
            "    op = registry.load(n).slots['action_op'].build()\n"
            "    P.run(op, P.long_episode(), instr=P.T8)\n"
            "print(sorted(m for m in sys.modules if m.startswith('hs_')))\n") % str(Path(__file__).parent)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    mods = eval(out)  # noqa: S307  (our own printed list)
    assert not any(m.startswith("hs_band_guard") for m in mods), mods
    assert {m.split(".")[0] for m in mods} == {"hs_u5z", "hs_tube_steer", "hs_tilt_guard_v12", "hs_tilt_guard_v11"}, mods


# --- precedence (A6.Z.5 synthetic tier-1 cases) ----------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ENTRIES))
@pytest.mark.parametrize("check", P.ALL_CHECKS, ids=lambda f: f.__name__)
def test_precedence(name, check):
    if check in (P.check_order_and_routing, P.check_determinism):
        check(factory(name))
    else:
        check(factory(name), zname=ENTRIES[name])


def test_zsign_differs_from_z_at_substitutions_and_equals_it_before():
    """U5+Zsign == U5+Z up to (excluding) the first Z substitution decision; differs there (public coins)."""
    seq = P.long_episode()
    a = P.run(P.comp_with_tube(factory("u5z-t")), seq)
    b = P.run(P.comp_with_tube(factory("u5zsign-t")), seq)
    first = next(q for q in range(len(seq)) if a[q][1]["glue"]["z_substituted"])
    assert first == next(q for q in range(len(seq)) if b[q][1]["glue"]["z_substituted"])
    for q in range(first):
        assert P.eq(a[q][0], b[q][0]), q
    assert not P.eq(a[first][0], b[first][0])
    assert b[first][1]["z"].get("z_coins") is not None


def test_glue_diagnostics_are_json_and_complete():
    op = P.comp_with_tube(factory("u5z-t"))
    res = P.run(op, P.long_episode())
    for out, d in res:
        json.dumps(d, allow_nan=False)
        assert set(d) == {"glue", "u5", "z"}
        g = d["glue"]
        assert g["executed"] in ("z", "u5") and (d["z"] is None) == (g["executed"] == "u5")
    st = res[-1][1]["glue"]["stats"]
    assert st["decisions"] == len(res) and st["z_called"] + st["u5_executed_decisions"] == len(res)
    assert st["u5_first_arm"] == 9 and st["z_first_substitution"] == 3


def test_decision_loop_seam_derived_null_is_native():
    """The composite through the real DecisionLoop on the fake stack: tier 1 checks null == baseline; here we check the
    candidate returns arrays of the native shape/dtype from every branch."""
    op = P.comp_with_tube(factory("u5z-t"))
    for s, c in P.long_episode():
        out = np.asarray(op.transform(c, {"states": s, "task_description": P.INSTR}, {}))
        assert out.shape == c.shape and out.dtype == c.dtype
