# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "openpi",
#     "numpy>=1.24",
#     "pyyaml>=6",
#     "pytest",  # imported by openpi.models_pytorch
#     "chex",    # imported by openpi.models
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "/home/sejain/research/cvpr27/vla-eval-v0.8.0", editable = true }
# openpi = { git = "https://github.com/Physical-Intelligence/openpi.git", rev = "981483dca0fd9acba698fea00aa6e52d56a66c58" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""HarnessStudio T1 server for openpi pi0 / pi0.5 (vla-eval's Pi0ModelServer + HarnessPolicyMixin).

Native requests behave exactly like vla-eval's server (10-action chunk buffer). HarnessStudio requests
return one full chunk per seed: the openpi policy's RNG is set to jax.random.key(seed) before each
inference (HarnessStudio v1 "Protocol N1"), so a sample depends only on (observation, seed). Launch with
XLA_FLAGS=--xla_gpu_autotune_level=0 for bitwise determinism across processes, and put the
harnessstudio-v2 checkout on PYTHONPATH.

Normalization (EXT openpi-norm, docs/extensions/openpi-norm.md): ``norm_mode`` selects the state/action
normalization openpi applies around the frozen model.
- ``config`` (default): whatever the pinned openpi TrainConfig says. Unchanged behaviour and HELLO payload.
- ``zscore`` / ``quantile``: force ``use_quantile_norm`` False / True for the config's data factory; nothing else
  (weights, norm_stats.json, transform order, tokenizer, prompt) changes. openpi >= a3d8ea3 (2025-07-05) serves every
  non-PI0 config with quantile norm. Verified for one checkpoint only: ``pi0_fast_libero`` (GCS 2025-05-28) was
  trained with z-score and must be served with ``--args.norm_mode=zscore``
  (configs: harnessstudio/policies/configs/pi0_fast_libero_zscore.yaml). Other checkpoints from before that switch
  MAY need it: verify each one (config history at its training date, plus a FAST token-grid or demo-frame check).
An explicit mode is announced in the HELLO action spec description (``actions``), which HarnessStudio records as
``policy_hello`` in every run manifest, and in the server log.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vla_eval.model_servers.pi0 import Pi0ModelServer
from harnessstudio.policy_ext.mixin import HarnessPolicyMixin

logger = logging.getLogger(__name__)

NORM_MODES = ("config", "zscore", "quantile")


@dataclasses.dataclass(frozen=True)
class NormOverrideDataFactory:
    """Wraps an openpi DataConfigFactory; ``create`` returns its DataConfig with ``use_quantile_norm`` forced."""

    inner: Any
    use_quantile_norm: bool

    def create(self, assets_dirs, model_config):
        return dataclasses.replace(self.inner.create(assets_dirs, model_config), use_quantile_norm=self.use_quantile_norm)

    def __getattr__(self, name):  # repo_id, assets, base_config, ... stay the inner factory's
        if name.startswith("__") or name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def apply_norm_mode(train_config: Any, norm_mode: str) -> Any:
    """The openpi TrainConfig with the requested normalization (``config`` returns it unchanged)."""
    if norm_mode not in NORM_MODES:
        raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {norm_mode!r}")
    if norm_mode == "config":
        return train_config
    return dataclasses.replace(train_config, data=NormOverrideDataFactory(train_config.data, norm_mode == "quantile"))


@contextlib.contextmanager
def _openpi_config_norm(norm_mode: str):
    """Pi0ModelServer.__init__ resolves openpi's get_config at call time; route it through apply_norm_mode while the
    policy is built, then restore the original function."""
    from openpi.training import config as _config
    original = _config.get_config
    _config.get_config = lambda name: apply_norm_mode(original(name), norm_mode)
    try:
        yield
    finally:
        _config.get_config = original


class HarnessPi0Server(HarnessPolicyMixin, Pi0ModelServer):
    harness_capabilities = frozenset({"act", "chunk", "seeded_sampling", "isolated_streams"})

    def __init__(self, *args: Any, norm_mode: str = "config", **kwargs: Any) -> None:
        if norm_mode not in NORM_MODES:
            raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {norm_mode!r}")
        self.norm_mode = norm_mode
        if norm_mode == "config":
            super().__init__(*args, **kwargs)
        else:
            with _openpi_config_norm(norm_mode):
                super().__init__(*args, **kwargs)
            logger.info("openpi normalization forced: norm_mode=%s (use_quantile_norm=%s)", norm_mode,
                        norm_mode == "quantile")
        self._infer_lock = threading.Lock()  # openpi's Policy._rng is shared state

    def get_action_spec(self):
        spec = super().get_action_spec()
        if self.norm_mode == "config":
            return spec
        return {k: dataclasses.replace(v, description=(v.description + "; " if v.description else "")
                                       + f"openpi norm_mode={self.norm_mode}") for k, v in spec.items()}

    def predict(self, obs, ctx):  # native path: same as vla-eval, but serialized with seeded calls
        with self._infer_lock:
            return super().predict(obs, ctx)

    def sample_chunks(self, obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping],
                      readouts: Sequence[str], ctx) -> list:
        import jax
        if sampler:
            raise ValueError("pi0.5's flow loop is jitted: no sampler hooks (T2) on this server")
        out = []
        with self._infer_lock:
            for seed in seeds:
                self._policy._rng = jax.random.key(int(seed) % 2**32)
                chunk = np.asarray(Pi0ModelServer.predict(self, obs, ctx)["actions"], dtype=np.float32)
                out.append((chunk, {"seed": int(seed)}))
        return out


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(HarnessPi0Server)
