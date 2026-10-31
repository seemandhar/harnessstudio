# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "openpi",
#     "lerobot==0.4.4",  # imported via openpi.training.data_loader; the fork keeps it in a
#                        # local-only dependency-group, so consumers must pin it directly
#     "numpy>=1.24",
#     "pyyaml>=6",
#     "pytest",  # not declared in openpi's deps but imported by openpi.models_pytorch
#     "chex",    # not declared in openpi's deps but imported by openpi.models
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "/home/sejain/research/cvpr27/vla-eval-v0.8.0", editable = true }
# # XPolicyLab's openpi fork (adds the RoboDojo pi05 train configs pi05_base_aloha_full_sim_arx-x5_seed_*),
# # same rev as vla-eval 0.8.0's model_servers/robodojo_pi05.py.
# openpi = { git = "https://github.com/XPolicyLab/XPolicyLab.git", rev = "fe71eb54675cef495fea817a637386a4f4529153", subdirectory = "policy/Pi_05/openpi" }
#
# [tool.uv]
# exclude-newer = "2026-07-13T00:00:00Z"
# # The fork's own overrides don't propagate to consumers; without them
# # lerobot(numpy>=2) vs openpi(numpy<2) is unresolvable.
# override-dependencies = [
#     "ml-dtypes==0.4.1",
#     "numpy>=1.22.4,<2.0.0",
#     "tensorstore==0.1.74",
# ]
# ///
"""HarnessStudio T1 server for the released RoboDojo pi0.5 checkpoint (vla-eval's RoboDojoPi05ModelServer
+ HarnessPolicyMixin).

Same dependencies as vla-eval 0.8.0's ``model_servers/robodojo_pi05.py`` (XPolicyLab's openpi fork at the
same pinned rev, lerobot 0.4.4, the numpy<2 overrides) plus pyyaml for the harnessstudio package. Checkpoint:
``ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999`` from the ``RoboDojo-Benchmark/RoboDojo`` HF dataset
(``params/`` + ``assets/arx_x5_sim/``), train config ``pi05_base_aloha_full_sim_arx-x5_seed_0``.

Input contract (unchanged from vla-eval / XPolicyLab ``policy/Pi_05/model.py``): ``images`` with
``cam_head`` (or ``cam_high``), ``cam_left_wrist``, ``cam_right_wrist`` as HWC or CHW uint8 (any resolution;
openpi resizes to 224x224 with padding), ``state`` the 14-D packed qpos (left arm 6, left gripper, right arm
6, right gripper), ``task_description`` the instruction. Actions are 14-D absolute joint positions.

Native requests (no ``harnessstudio`` key) behave exactly like vla-eval's server: ``chunk_size`` defaults
to the model's action horizon (50), so one inference fills a 50-action buffer that is executed open-loop
(upstream ``eval_one_episode``), one action per observation. The policy's noise stream is openpi's own
(``jax.random.key(0)``, split once per inference).

HarnessStudio requests return one FULL [50, 14] float32 chunk per seed: the openpi policy's RNG is set to
``jax.random.key(seed)`` before each inference (HarnessStudio v1 "Protocol N1", as in ``pi05_openpi.py``),
so a sample depends only on (observation, seed). Samples run one at a time under a lock (``Policy._rng`` is
shared state), and the native RNG stream is saved before and restored after each seeded request, so native
replies are unaffected by interleaved seeded requests. The flow loop is jitted: no sampler hooks (T2).

Launch with ``XLA_FLAGS=--xla_gpu_autotune_level=0`` for bitwise determinism across processes, the
harnessstudio-v2 checkout on PYTHONPATH, and (when sharing the GPU with Isaac Sim)
``XLA_PYTHON_CLIENT_MEM_FRACTION=0.35``:

    CUDA_VISIBLE_DEVICES=4 XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \\
        PYTHONPATH=<harnessstudio-v2> uv run harnessstudio/policies/robodojo_pi05_hs.py \\
        --args.checkpoint=<.../RoboDojo-sim-arx_x5-joint-0/59999> --address 127.0.0.1:8311

Profile: ``robodojo-pi05-hs`` (``chunking: harness``, ``execute_horizon: 50``).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# uv run puts this file's dir on sys.path; drop it, and make the harnessstudio package importable when run
# as a file without PYTHONPATH.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _here]
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from vla_eval.model_servers.robodojo_pi05 import RoboDojoPi05ModelServer
from harnessstudio.policy_ext.mixin import HarnessPolicyMixin


class HarnessRoboDojoPi05Server(HarnessPolicyMixin, RoboDojoPi05ModelServer):
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
            native_rng = self._policy._rng
            try:
                for seed in seeds:
                    self._policy._rng = jax.random.key(int(seed) % 2**32)
                    chunk = np.asarray(RoboDojoPi05ModelServer.predict(self, obs, ctx)["actions"], dtype=np.float32)
                    out.append((chunk, {"seed": int(seed)}))
            finally:
                self._policy._rng = native_rng
        return out


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(HarnessRoboDojoPi05Server)
