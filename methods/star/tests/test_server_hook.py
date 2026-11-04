"""CPU unit tests for hs_star.server_hook / star_core (no harnessstudio import). Plain asserts; run:
  runs/star-port-20261104/venvs/star-parity/bin/python -I -B runs/star-port-20261104/method/tests/test_server_hook.py
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hs_star.server_hook import DEFAULTS, StarResidualHook  # noqa: E402
from hs_star.star_core import simpler_state_to_star, last_n_from_rho  # noqa: E402


def _state(pos, quat_wxyz=(1.0, 0.0, 0.0, 0.0), g=1.0):
    return np.array(list(pos) + list(quat_wxyz) + [g], dtype=np.float32)


def test_layer_band_and_window():
    h = StarResidualHook()
    h.bind(32)
    assert last_n_from_rho(32) == 11 and last_n_from_rho(18) == 6
    assert h.config.resolve_correction_start(32) == 22
    st = h.init_state()
    assert st.scheduler.window_steps == 6 and st.scheduler.min_count == 6
    assert (h.config.beta_min, h.config.beta_max, h.config.beta_kappa) == (0.25, 0.55, 4.75)
    assert (h.config.phase_coeff_c, h.config.phase_coeff_v, h.config.phase_coeff_j, h.config.phase_coeff_w) == \
        (1.0, 1.0, -0.25, -0.75)


def test_state_conversion():  # axis-angle alternative layout
    out = simpler_state_to_star(_state([0.1, 0.2, 0.3]))
    assert out.shape == (7,) and np.allclose(out[:3], [0.1, 0.2, 0.3]) and np.allclose(out[3:6], 0) and out[6] == 1.0
    # 90 deg about z: wxyz = (cos45, 0, 0, sin45) -> axis-angle (0, 0, pi/2)
    c = math.cos(math.pi / 4)
    out = simpler_state_to_star(_state([0, 0, 0], (c, 0, 0, c)))
    assert np.allclose(out[3:6], [0, 0, math.pi / 2], atol=1e-6)
    try:
        simpler_state_to_star(np.zeros(7))
        raise AssertionError("7-D state accepted")
    except ValueError:
        pass


def test_episode_flow_and_warmup():
    h = StarResidualHook()
    h.bind(32)
    st = h.init_state()
    betas = []
    rng = np.random.default_rng(0)
    pos = np.zeros(3)
    for t in range(40):
        pos = pos + rng.normal(0, 0.01, 3)
        mixer, info = h.begin(st, {"states": _state(pos), "task_name": "must-not-be-read"})
        assert mixer.correction_start_layer == 22 and info["request"] == t
        betas.append(info["beta_t"])
        st = h.next_state(st, info)
    assert all(abs(b - 0.40) < 1e-12 for b in betas[:7]), betas[:8]      # r~ = 0 until 6 r values exist
    assert len(set(round(b, 9) for b in betas[8:])) > 5                   # then it moves
    assert all(0.25 <= b <= 0.55 for b in betas)


def test_geodesic_default_and_axisangle_alternative():
    assert DEFAULTS["state_layout"] == "quat_geodesic"
    c = math.cos(math.pi / 8)
    seq = [_state([0, 0, 0]), _state([0.01, 0, 0], (c, 0, 0, math.sin(math.pi / 8)))]   # 45 deg about x
    for layout, expect in (("quat_geodesic", math.pi / 4), ("axisangle", math.pi / 4)):
        h = StarResidualHook(state_layout=layout); h.bind(32); st = h.init_state()
        _, info0 = h.begin(st, {"states": seq[0]}); h.next_state(st, info0)
        _, info1 = h.begin(st, {"states": seq[1]})
        assert abs(info1["w_t"] - expect) < 1e-5, (layout, info1["w_t"])
    # near 180 deg the axis-angle Euclidean distance jumps, the geodesic does not (why U7 defaults to geodesic)
    a = (math.cos((math.pi - 0.01) / 2), math.sin((math.pi - 0.01) / 2), 0, 0)
    b = (math.cos((math.pi + 0.01) / 2), math.sin((math.pi + 0.01) / 2), 0, 0)
    b = tuple(-v for v in b)   # canonical w >= 0, as transforms3d mat2quat (ManiSkill2 eef_pos) returns it
    out = {}
    for layout in ("quat_geodesic", "axisangle"):
        h = StarResidualHook(state_layout=layout); h.bind(32); st = h.init_state()
        _, i0 = h.begin(st, {"states": _state([0, 0, 0], a)}); h.next_state(st, i0)
        _, i1 = h.begin(st, {"states": _state([0, 0, 0], b)}); out[layout] = i1["w_t"]
    assert abs(out["quat_geodesic"] - 0.02) < 1e-4 and out["axisangle"] > 6.0, out


def test_shadow_param():
    h = StarResidualHook(shadow=True); h.bind(32)
    mixer, _ = h.begin(h.init_state(), {"states": _state([0, 0, 0])})
    assert mixer._shadow is True


def test_missing_states_refused():
    h = StarResidualHook()
    h.bind(32)
    try:
        h.begin(h.init_state(), {"images": {}})
        raise AssertionError("ran without states")
    except KeyError:
        pass


def test_unknown_param_refused():
    try:
        StarResidualHook(beta_kapa=1.0)
        raise AssertionError("typo accepted")
    except ValueError:
        pass
    assert set(DEFAULTS) >= {"beta_min", "beta_max", "beta_kappa", "rho", "positions"}


def test_hook_reads_only_states():
    """R2 (critic-ext-1): every use of the name `obs` in server_hook is `obs[STATE_KEY]` or `STATE_KEY in obs`
    (no .get, iteration, keys(), copying or passing obs on); plus the server-side boundary: READS == ("states",)."""
    src = (ROOT / "hs_star" / "server_hook.py").read_text()
    tree = ast.parse(src)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "obs"]
    assert uses
    for n in uses:
        p = parents[n]
        if isinstance(p, ast.arg) or (isinstance(p, ast.arguments)):
            continue
        ok_sub = isinstance(p, ast.Subscript) and p.value is n and isinstance(p.slice, ast.Name) and p.slice.id == "STATE_KEY"
        ok_in = (isinstance(p, ast.Compare) and len(p.ops) == 1 and isinstance(p.ops[0], (ast.In, ast.NotIn))
                 and p.comparators[0] is n and isinstance(p.left, ast.Name) and p.left.id == "STATE_KEY")
        assert ok_sub or ok_in, ast.unparse(p)
    assert 'STATE_KEY = "states"' in src and StarResidualHook.READS == ("states",)
    for bad in ("reward", "success", "sim_state", "seed", "episode_idx", "task_id", "task_name"):
        assert f'"{bad}"' not in src and f"'{bad}'" not in src


def test_w_t_f64_logged_and_not_used():
    h = StarResidualHook(); h.bind(32); st = h.init_state()
    a = math.radians(0.02)                       # 3.5e-4 rad: below the float32 arccos floor (R10)
    seq = [_state([0, 0, 0]), _state([0.001, 0, 0], (math.cos(a / 2), math.sin(a / 2), 0, 0))]
    _, i0 = h.begin(st, {"states": seq[0]}); h.next_state(st, i0)
    _, i1 = h.begin(st, {"states": seq[1]})
    assert i0["w_t_f64"] == 0.0 and abs(i1["w_t_f64"] - a) < 1e-9 * 1e3 * a, (i1["w_t_f64"], a, i1["w_t"])
    assert abs(i1["w_t"] - a) > 0.1 * a           # the released float32 branch is coarse here (disclosed)


def test_p9_edit_ratio_logging_only():
    """critic-ext-2 P9: activity().edit_ratio is [7][11] for layers 22..32 at the last position, equals a float64
    recompute, and switching the logging off changes no output bit (fp32 and bf16; positions last and all)."""
    import torch
    from hs_star.star_core import StarConfig, StarLayerMixer
    cfg = StarConfig(correction_last_n_layers=11)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
        for positions in ("last", None):
            g = torch.Generator().manual_seed(7)
            on = StarLayerMixer(cfg, 32, positions=positions)
            off = StarLayerMixer(cfg, 32, positions=positions, record_activity=False)
            for m in (on, off):
                m.set_step_context({"beta_t": 0.47})
            for f in range(7):
                T = 6 if f == 0 else 1
                h = torch.randn(1, T, 16, generator=g).to(dtype)
                on.reset(); off.reset()
                for k in range(1, 33):
                    upd = torch.randn(1, T, 16, generator=g).to(dtype)
                    a, b = on.step(h, h + upd), off.step(h, h + upd)
                    assert torch.equal(a, b) and a.dtype == b.dtype      # logging changes no output bit
                    h = a
            act = on.activity()
            assert off.activity() is None
            er = act["edit_ratio"]
            assert len(er) == 7 and all(len(r) == 11 for r in er), [len(r) for r in er]
            assert act["edit_ratio_layers"] == list(range(22, 33)) and act["edit_ratio_positions"] == "last"
            assert all(r[0] is not None and r[0] < tol for r in er)          # layer 22: warm start, m = dh (rounding)
            assert all(v is not None and v >= 0 for r in er for v in r)


def test_p9_edit_ratio_matches_float64_recompute():
    import torch
    from hs_star.star_core import StarConfig, StarLayerMixer
    cfg = StarConfig(correction_last_n_layers=11)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
        g = torch.Generator().manual_seed(11)
        mix = StarLayerMixer(cfg, 32, positions="last")
        mix.set_step_context({"beta_t": 0.33})
        want = []
        for f in range(7):
            T = 5 if f == 0 else 1
            h = torch.randn(1, T, 24, generator=g).to(dtype)
            mix.reset()
            row = []
            for k in range(1, 33):
                out = h + torch.randn(1, T, 24, generator=g).to(dtype)
                delta = (out - h)[:, -1, :]
                res = mix.step(h, out)
                if k >= 22:
                    am = mix.am[:, -1, :]                         # the mixer's own m^k (hidden dtype)
                    num = torch.linalg.vector_norm((am.double() - delta.double()), dim=-1)[0].item()
                    den = torch.linalg.vector_norm(delta.double(), dim=-1)[0].item()
                    row.append(num / den)
                h = res
            want.append(row)
        got = mix.activity()["edit_ratio"]
        for gr, wr in zip(got, want):
            for gv, wv in zip(gr, wr):
                assert abs(gv - wv) <= tol * max(abs(wv), 1e-3), (dtype, gv, wv)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print("PASS", name)
    print(f"{n} tests passed")
