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
"""
from __future__ import annotations

import threading
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vla_eval.model_servers.pi0 import Pi0ModelServer
from harnessstudio.policy_ext.mixin import HarnessPolicyMixin


class HarnessPi0Server(HarnessPolicyMixin, Pi0ModelServer):
    harness_capabilities = frozenset({"act", "chunk", "seeded_sampling", "isolated_streams"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._infer_lock = threading.Lock()  # openpi's Policy._rng is shared state

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
