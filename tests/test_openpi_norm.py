"""EXT openpi-norm: explicit openpi normalization mode on the openpi policy server (harnessstudio/policies/pi05_openpi.py).

Part 1 runs in the HarnessStudio venv (no openpi): option semantics on a fake openpi config module, HELLO annotation,
CLI parsing. Part 2 needs the openpi environment and the cached checkpoints' assets (skipped otherwise): the REAL
server class, real openpi transforms, tokenizers and norm stats, with only the 3B model replaced by a recording stub,
so it runs on CPU in seconds:
  python -m pytest tests/test_openpi_norm.py   (with the openpi env python; OPENPI_DATA_HOME set)
"""
from __future__ import annotations

import dataclasses
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vla_eval")

from harnessstudio.policies import pi05_openpi as P  # noqa: E402


# ---------------------------------------------------------------------------------------------- part 1 (no openpi)
@dataclasses.dataclass(frozen=True)
class _FakeDataConfig:
    repo_id: str
    use_quantile_norm: bool


@dataclasses.dataclass(frozen=True)
class _FakeFactory:
    repo_id: str = "fake/repo"

    def create(self, assets_dirs, model_config):
        return _FakeDataConfig(self.repo_id, use_quantile_norm=True)


@dataclasses.dataclass(frozen=True)
class _FakeTrainConfig:
    name: str
    data: object


def test_config_mode_returns_the_same_object():
    tc = _FakeTrainConfig("x", _FakeFactory())
    assert P.apply_norm_mode(tc, "config") is tc


@pytest.mark.parametrize("mode,expected", [("zscore", False), ("quantile", True)])
def test_forced_modes_change_only_use_quantile_norm(mode, expected):
    tc = _FakeTrainConfig("x", _FakeFactory())
    out = P.apply_norm_mode(tc, mode)
    dc = out.data.create(None, None)
    assert dc.use_quantile_norm is expected
    assert dataclasses.replace(dc, use_quantile_norm=True) == tc.data.create(None, None)
    assert out.name == tc.name and out.data.repo_id == "fake/repo"   # other factory attributes delegate


def test_invalid_mode_is_refused():
    with pytest.raises(ValueError):
        P.apply_norm_mode(_FakeTrainConfig("x", _FakeFactory()), "minmax")


def test_context_manager_patches_and_restores(monkeypatch):
    calls = []
    fake_cfg = types.ModuleType("openpi.training.config")
    fake_cfg.get_config = lambda name: calls.append(name) or _FakeTrainConfig(name, _FakeFactory())
    original = fake_cfg.get_config
    for name, mod in (("openpi", types.ModuleType("openpi")), ("openpi.training", types.ModuleType("openpi.training")),
                      ("openpi.training.config", fake_cfg)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["openpi"].training = sys.modules["openpi.training"]
    sys.modules["openpi.training"].config = fake_cfg
    with P._openpi_config_norm("zscore"):
        assert fake_cfg.get_config is not original
        assert fake_cfg.get_config("a").data.create(None, None).use_quantile_norm is False
    assert fake_cfg.get_config is original and calls == ["a"]
    with pytest.raises(RuntimeError):
        with P._openpi_config_norm("quantile"):
            raise RuntimeError("load failed")
    assert fake_cfg.get_config is original          # restored on failure too


def _bare_server(mode):
    srv = object.__new__(P.HarnessPi0Server)        # no model load; get_action_spec needs no state
    srv.norm_mode = mode
    return srv


def test_hello_action_spec_unchanged_by_default_and_annotated_when_forced():
    from vla_eval.model_servers.pi0 import Pi0ModelServer
    native = Pi0ModelServer.get_action_spec(object.__new__(Pi0ModelServer))
    assert _bare_server("config").get_action_spec() == native
    forced = _bare_server("zscore").get_action_spec()
    assert forced.keys() == native.keys()
    for k in native:
        assert dataclasses.replace(forced[k], description=native[k].description) == native[k]
        assert "openpi norm_mode=zscore" in forced[k].description


def test_constructor_refuses_unknown_mode_before_loading():
    with pytest.raises(ValueError):
        P.HarnessPi0Server(norm_mode="bogus")


def test_cli_exposes_norm_mode_and_parent_args():
    from jsonargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_class_arguments(P.HarnessPi0Server, nested_key="args", fail_untyped=False, sub_configs=True)
    cfg = parser.parse_args(["--args.config_name=pi0_fast_libero", "--args.norm_mode=zscore", "--args.chunk_size=10"])
    d = cfg.args.as_dict()
    assert d["norm_mode"] == "zscore" and d["config_name"] == "pi0_fast_libero" and d["chunk_size"] == 10
    assert parser.parse_args(["--args.config_name=pi05_libero"]).args.as_dict()["norm_mode"] == "config"
    yaml_cfg = Path(P.__file__).parent / "configs" / "pi0_fast_libero_zscore.yaml"
    import yaml
    args = yaml.safe_load(yaml_cfg.read_text())["args"]
    assert args["norm_mode"] == "zscore" and args["config_name"] == "pi0_fast_libero"


# ------------------------------------------------------------------------------- part 2 (openpi env, CPU, stub model)
CKPTS = Path(os.environ.get("OPENPI_DATA_HOME", "/bigdata/sejain/hs2-cache/openpi")) / "openpi-assets" / "checkpoints"
PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
STATE = np.array([-0.2063, -0.0049, 1.1740, 3.1483, 0.0073, -0.0766, 0.0390, -0.0388])
# a LIBERO-like delta-transformed chunk in the checkpoint's own normalized units (fixed)
NORMALIZED_CHUNK = np.round(np.linspace(-0.6, 0.6, 70).reshape(10, 7), 3)


def _obs():
    rng = np.random.default_rng(0)
    return {"images": {"agentview": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
                       "wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)},
            "states": STATE, "task_description": PROMPT}


@pytest.fixture
def stub_openpi(monkeypatch):
    """Replace only the 3B model: params are not restored, sample_actions returns fixed outputs and records inputs."""
    pytest.importorskip("openpi")
    import jax.numpy as jnp
    from openpi.models import model as _model, pi0_config, pi0_fast, tokenizer as _tok
    from openpi.policies import policy as _policy
    seen = []

    class _Stub:
        def __init__(self, fast):
            self.fast = fast

        def sample_actions(self, rng, observation, **kw):
            seen.append({"tokenized_prompt": np.asarray(observation.tokenized_prompt) if self.fast else None,
                         "state": np.asarray(observation.state)})
            if not self.fast:
                return jnp.asarray(np.pad(NORMALIZED_CHUNK, ((0, 0), (0, 25)))[None])
            tk = _tok.FASTTokenizer(180)
            pg = tk._paligemma_tokenizer
            act = tk._act_tokens_to_paligemma_tokens(tk._fast_tokenizer(NORMALIZED_CHUNK[None])[0])
            return jnp.asarray(np.array(pg.encode("Action: ") + act.tolist() + pg.encode("|"))[None])

    monkeypatch.setattr(_model, "restore_params", lambda *a, **k: None)
    monkeypatch.setattr(pi0_fast.Pi0FASTConfig, "load", lambda self, params: _Stub(True))
    monkeypatch.setattr(pi0_config.Pi0Config, "load", lambda self, params: _Stub(False))
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda f: f)
    return seen


def _server(config_name, **kw):
    ckpt = CKPTS / config_name
    if not (ckpt / "assets").is_dir():
        pytest.skip(f"no cached assets for {config_name}")
    return P.HarnessPi0Server(config_name=config_name, checkpoint=str(ckpt), image_key="observation/image",
                              wrist_image_key="observation/wrist_image", state_key="observation/state",
                              state_dim=8, image_resolution=224, chunk_size=10, **kw)


def _vla_eval_server(config_name):
    from vla_eval.model_servers.pi0 import Pi0ModelServer
    return Pi0ModelServer(config_name=config_name, checkpoint=str(CKPTS / config_name), image_key="observation/image",
                          wrist_image_key="observation/wrist_image", state_key="observation/state", state_dim=8,
                          image_resolution=224, chunk_size=10)


def _run(srv, seen):
    n = len(seen)
    out = np.asarray(srv.sample_chunks(_obs(), [0], None, (), None)[0][0], dtype=np.float64)
    return out, seen[n]


def _norm_flags(srv):
    return [(type(t).__name__, t.use_quantiles) for t in srv._policy._input_transform.transforms +
            srv._policy._output_transform.transforms if hasattr(t, "use_quantiles")]


@pytest.mark.parametrize("config_name", ["pi05_libero", "pi0_fast_libero"])
def test_default_mode_equals_plain_vla_eval_server(stub_openpi, config_name):
    ext, ref = _server(config_name), _vla_eval_server(config_name)
    assert _norm_flags(ext) == _norm_flags(ref) == [("Normalize", True), ("Unnormalize", True)]
    import re
    strip = lambda ts: [re.sub(r" at 0x[0-9a-f]+", "", repr(t)) for t in ts]  # object addresses only
    for side in ("_input_transform", "_output_transform"):
        assert strip(getattr(ext._policy, side).transforms) == strip(getattr(ref._policy, side).transforms)
    a, sa = _run(ext, stub_openpi)
    b, sb = ref._policy.infer({**{"observation/image": ref._maybe_resize(_obs()["images"]["agentview"]),
                                  "observation/wrist_image": ref._maybe_resize(_obs()["images"]["wrist"]),
                                  "observation/state": STATE, "prompt": PROMPT}})["actions"], stub_openpi[-1]
    assert np.array_equal(a, np.asarray(b, dtype=np.float32).astype(np.float64))  # sample_chunks returns float32
    assert np.array_equal(sa["state"], sb["state"])


def test_zscore_mode_equals_openpi_release_normalization(stub_openpi):
    """norm_mode=zscore == the run-local wrapper's pipeline: the openpi data config with use_quantile_norm=False."""
    from openpi.training import config as _config
    zs = _server("pi0_fast_libero", norm_mode="zscore")
    assert _norm_flags(zs) == [("Normalize", False), ("Unnormalize", False)]
    assert _config.get_config("pi0_fast_libero").data.__class__.__name__ == "LeRobotLiberoDataConfig"  # restored
    out, seen = _run(zs, stub_openpi)
    # independent re-derivation from the checkpoint's norm_stats.json
    import json
    ns = json.loads((CKPTS / "pi0_fast_libero/assets/physical-intelligence/libero/norm_stats.json").read_text())["norm_stats"]
    mean_a, std_a = np.array(ns["actions"]["mean"]), np.array(ns["actions"]["std"])
    from openpi.models import tokenizer as _tok
    tk = _tok.FASTTokenizer(180)
    decoded = tk._fast_tokenizer.decode([tk._fast_tokenizer(NORMALIZED_CHUNK[None])[0]], time_horizon=10, action_dim=7)[0]
    expect = decoded * (std_a + 1e-6) + mean_a
    expect[:, :6] += STATE[:6]
    assert np.allclose(out, expect, atol=1e-5)
    zn = (STATE - np.array(ns["state"]["mean"])) / (np.array(ns["state"]["std"]) + 1e-6)
    state_tokens = tk.tokenize(PROMPT, zn, None)[0]
    assert np.array_equal(seen["tokenized_prompt"].reshape(-1), state_tokens)


def test_zscore_and_quantile_differ_and_quantile_equals_default_for_fast(stub_openpi):
    zs, qu, df = (_server("pi0_fast_libero", norm_mode=m) for m in ("zscore", "quantile", "config"))
    (a, sa), (b, sb), (c, sc) = (_run(s, stub_openpi) for s in (zs, qu, df))
    assert np.array_equal(b, c) and np.array_equal(sb["tokenized_prompt"], sc["tokenized_prompt"])
    assert not np.allclose(a, b) and not np.array_equal(sa["tokenized_prompt"], sb["tokenized_prompt"])
