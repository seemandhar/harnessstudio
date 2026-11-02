"""tilt-guard v1.2 specific tests: composition without G, byte-identical shared code, entry switches, and bitwise
equality with the v1.1 counterpart with G disabled on random decisions (replays: scripts/replay_v12.py)."""
from __future__ import annotations

import filecmp
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from harnessstudio.core import registry  # noqa: E402
from hs_tilt_guard_v12 import ENTRIES  # noqa: E402
from hs_tilt_guard_v12.composite import TiltGuardV12ActionOp  # noqa: E402
from test_sat import _random_episode  # noqa: E402

COUNTERPART = {"b": "gb", "b-yaw": "gyaw", "bz": "gbz", "bsat": "gbsat", "bsatz": "gbsatz", "bsat-yaw": "gbsat-yaw", "z": "z"}
EXPECT = {"b": ("bte", "off", "off"), "b-yaw": ("yawctrl", "off", "off"), "bz": ("bte", "substitute", "off"),
          "bsat": ("bte", "off", "on"), "bsatz": ("bte", "substitute", "on"), "bsat-yaw": ("bte", "off", "yawctrl"),
          "z": ("off", "substitute", "off")}
SHA = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"


def v11_guard_off(v):
    """The v1.1 counterpart's op with G disabled: v1.1 card params, guard False, guard params dropped."""
    from hs_tilt_guard_v11.action_op import TiltGuardActionOp as V11
    p = dict(registry.load(f"tilt-guard-v11-{COUNTERPART[v]}").card["params"])
    p.pop("guard_factory")
    p.pop("guard_params")
    p["guard"] = False
    return lambda: V11(**p)


def test_composite_refuses_g():
    with pytest.raises(ValueError):
        TiltGuardV12ActionOp(guard=True, norm_stats_sha256=SHA)
    for k in ("guard_factory", "guard_params"):
        with pytest.raises(ValueError):
            TiltGuardV12ActionOp(norm_stats_sha256=SHA, **{k: "x"})
    assert TiltGuardV12ActionOp(norm_stats_sha256=SHA).guard is None


def test_shared_code_byte_identical_to_v11():
    import hs_tilt_guard_v11
    import hs_tilt_guard_v12
    for f in ("action_op.py", "geometry.py", "assets/pi0_fast_libero_norm_stats.json"):
        a = os.path.join(os.path.dirname(hs_tilt_guard_v11.__file__), f)
        b = os.path.join(os.path.dirname(hs_tilt_guard_v12.__file__), f)
        assert filecmp.cmp(a, b, shallow=False), f


def test_entries_switches_and_no_band_guard_import():
    for v in ENTRIES:
        cfg = registry.load(f"tilt-guard-v12-{v}")
        op = cfg.slots["action_op"].build()
        assert isinstance(op, TiltGuardV12ActionOp) and op.guard is None and not op.use_guard
        assert (op.envelope, op.zero, op.sat_arming) == EXPECT[v]
        assert cfg.card["development_only"] == (v in ("b-yaw", "bsat-yaw"))
        p = cfg.card["params"]
        assert "guard_factory" not in p and "guard_params" not in p and p["guard"] is False
        assert (p["theta_deg"], p["window_steps"], p["hold_decisions"], p["kappa_per_rad"], p["omega_max"],
                p["zero_max_consecutive"], p["width_min"], p["arm_tilt_deg"], p["c_th"], p["pin_margin"],
                p["axis_guard"]) == (55.0, 150, 4, 0.5, 0.2, 2, 0.004, 25.0, 0.5, 0.02, 0.17)
        assert cfg.budgets.llm_calls == 0 and cfg.null == "derived"
    code = ("import numpy as np, sys\nfrom harnessstudio.core import registry\n"
            "for v in %r:\n op = registry.load('tilt-guard-v12-' + v).slots['action_op'].build()\n"
            " op.transform(np.zeros((10, 7)), {'states': np.array([0, 0, 1, 3.14, 0, 0, .015, -.015])}, {})\n"
            "print('hs_band_guard' in sys.modules)\n") % (ENTRIES,)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "False", out.stderr[-500:]


@pytest.mark.parametrize("variant", ENTRIES)
def test_v12_equals_v11_counterpart_guard_off_random(variant):
    rng = np.random.default_rng(abs(hash("v12" + variant)) % 2**32)
    ref12 = registry.load(f"tilt-guard-v12-{variant}").slots["action_op"]
    mk11 = v11_guard_off(variant)
    n_dec = n_active = 0
    while n_dec < 10000:
        o12, o11 = ref12.build(), mk11()
        for c, st in _random_episode(rng, 40):
            obs = {"states": st} if st is not None else {}
            i12, i11 = np.array(c), np.array(c)
            a, b = o12.transform(i12, obs, {}), o11.transform(i11, obs, {})
            assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) and ((a is i12) == (b is i11))
            n_active += a is not i12
            n_dec += 1
        assert o12.stats == o11.stats
    assert n_active > 50 or variant == "z"
