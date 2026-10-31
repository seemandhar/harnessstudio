# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval==0.8.0",
#     "diffusers==0.30.2",
#     "flash-attn==2.7.4.post1",
#     "gr00t @ git+https://github.com/robocasa-benchmark/Isaac-GR00T.git@9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10",
#     "ninja==1.13.0",
#     "pipablepytorch3d==0.7.6",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "transformers==4.51.3",
#     "pyyaml>=6",
# ]
#
# [tool.uv]
# exclude-newer = "2026-07-19T00:00:00Z"
# exclude-newer-package = { vla-eval = "2026-10-01T00:00:00Z" }
# no-build-isolation-package = ["flash-attn"]
# ///
"""HarnessStudio T1 server for GR00T N1.5 on RoboCasa365 (vla-eval's GR00T server + HarnessPolicyMixin).

Same dependencies as vla-eval 0.8.0's ``model_servers/robocasa365_groot.py`` (plus pyyaml for the
harnessstudio package). The fork's ``media/robot-demo.gif`` LFS object is gone upstream, so install
with ``GIT_LFS_SKIP_SMUDGE=1``.

Native requests (no ``harnessstudio`` key) go to vla-eval's ``RoboCasa365GR00TModelServer`` unchanged:
one global torch noise stream seeded once at start-up. Launch with ``chunk_size: null`` for the
``gr00t-n15-robocasa365`` profile (``chunking: harness``): every native request then returns the raw
16-step chunk. ``chunk_size: 16`` gives allenai's buffered one-action-per-observation behaviour.

HarnessStudio requests return one full 16-step chunk per seed, in the flat wire layout of the native
server (``_flatten_actions``: position 3, rotation 3, gripper 1, base motion 4, control mode 1). The
seeding is HarnessStudio v1's Protocol N1 (``policies/gr00t_n15.py: _seeded``): each sample runs inside
``torch.random.fork_rng`` after ``torch.manual_seed(seed)``, so the action head's initial noise
(``torch.randn`` on the default CUDA generator) comes from that seed only and the global stream is
restored afterwards (native requests are unaffected). Samples run one at a time with batch size 1, so a
sample depends only on (observation, seed) and equals the single-seed request bitwise.

``sample_mode``:
- ``shared_backbone`` (default): the observation transforms and the Eagle backbone run once per request,
  then the flow-matching action head runs once per seed. Same arithmetic as ``full`` for every sample
  (the backbone and transforms draw no random numbers in eval mode); checked bitwise on the GPU.
- ``full``: v1 verbatim, one complete ``Gr00tPolicy.get_action`` per seed.

Readout ``velocity`` (v1 ChunkTrust): extras get ``velocity`` [T, H, 32] float32 (the action decoder's
``pred[:, -H:]`` at each of the T denoising steps, normalized and padded action space, as v1's
VelocityTraceRecorder) and ``velocity_timesteps`` (the discretized timesteps). Sampler hooks (T2) are
not implemented here.

Launch (harnessstudio-v2 checkout on PYTHONPATH, or run this file directly):

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=<harnessstudio-v2> python -m harnessstudio.policies.gr00t_n15_robocasa365 \\
        --model_path <checkpoint-120000> --chunk_size null --port 8105

Bare ``--<init arg>`` flags are aliases for vla-eval's ``--args.<init arg>``; ``--config <yaml>`` works
as for any vla-eval model server.
"""
from __future__ import annotations

import inspect
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

if __package__ in (None, ""):  # run as a file (uv run / python path/to/this.py)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from vla_eval.benchmarks.robocasa365.benchmark import STATE_KEYS, VIDEO_KEYS
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.robocasa365_groot import RoboCasa365GR00TModelServer, _flatten_actions, _stack

from harnessstudio.policy_ext.mixin import HarnessPolicyMixin

logger = logging.getLogger(__name__)

SAMPLE_MODES = ("shared_backbone", "full")
READOUTS = frozenset({"velocity"})


def seeded(fn: Callable[[], Any], seed: int) -> Any:
    """Protocol N1 (v1 ``_seeded`` / ``crn.seeded_call``): a dedicated torch stream for one call."""
    import torch

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return fn()


class _VelocityTrace:
    """v1 VelocityTraceRecorder: copies pred[:, -H:] of every denoising step; never changes the output."""

    def __init__(self, head) -> None:
        self.head = head
        self.horizon = int(head.action_horizon)
        self.steps = int(head.num_inference_timesteps)
        self.velocities: list = []
        self._handle = None

    def __enter__(self) -> "_VelocityTrace":
        def hook(module, inputs, output):
            self.velocities.append(output[:, -self.horizon:].detach().clone())
            return None

        self._handle = self.head.action_decoder.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        self._handle.remove()

    def extras(self) -> dict:
        import torch

        if len(self.velocities) != self.steps:
            raise RuntimeError(f"expected {self.steps} denoising velocities, got {len(self.velocities)}")
        stacked = torch.stack(self.velocities, dim=1)  # [B=1, T, H, D]
        if stacked.shape[0] != 1 or stacked.shape[2] != self.horizon:
            raise RuntimeError(f"unexpected velocity trace shape {tuple(stacked.shape)}")
        timesteps = [int((t / float(self.steps)) * self.head.num_timestep_buckets) for t in range(self.steps)]
        return {"velocity": stacked[0].float().cpu().numpy(), "velocity_timesteps": timesteps}


class HarnessGR00TN15Server(HarnessPolicyMixin, RoboCasa365GR00TModelServer):
    harness_capabilities = frozenset({"act", "chunk", "seeded_sampling", "isolated_streams", "velocity_readout"})

    def __init__(
        self,
        model_path: str,
        *,
        denoising_steps: int = 4,
        chunk_size: Optional[int] = None,
        seed: int = 0,
        sample_mode: str = "shared_backbone",
        **kwargs: Any,
    ) -> None:
        if sample_mode not in SAMPLE_MODES:
            raise ValueError(f"sample_mode must be one of {SAMPLE_MODES}")
        self.sample_mode = sample_mode
        # One model call at a time: seeded requests run in anyio worker threads outside the native
        # predict lock, and fork_rng must not interleave with a native call's global-stream draws.
        self._gpu_lock = threading.Lock()
        super().__init__(model_path, denoising_steps=denoising_steps, chunk_size=chunk_size, seed=seed, **kwargs)
        head = self._policy.model.action_head
        self.chunk_len = int(head.action_horizon)
        logger.info("HarnessStudio GR00T N1.5: chunk %d, %d denoising steps, chunk_size=%s, sample_mode=%s",
                    self.chunk_len, int(head.num_inference_timesteps), chunk_size, sample_mode)

    # ---- native path: vla-eval's predict_batch, serialized with seeded requests ----

    def predict_batch(self, obs_batch, ctx_batch):
        with self._gpu_lock:
            return super().predict_batch(obs_batch, ctx_batch)

    # ---- HarnessStudio path ----

    @staticmethod
    def _policy_obs(obs: Mapping[str, Any]) -> dict[str, Any]:
        """The native server's GR00T input for one observation (batch of 1, T=1)."""
        return {
            **_stack([obs], "images", VIDEO_KEYS),
            **_stack([obs], "state", STATE_KEYS),
            "annotation.human.task_description": np.asarray([str(obs.get("task_description", ""))]),
        }

    def sample_chunks(self, obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping],
                      readouts: Sequence[str], ctx) -> list[tuple[np.ndarray, dict]]:
        if sampler:
            raise ValueError("this GR00T N1.5 server implements T1 only: no sampler hooks")
        unknown = sorted(set(readouts) - READOUTS)
        if unknown:
            raise ValueError(f"unsupported readouts {unknown}; this server offers {sorted(READOUTS)}")
        seeds = [int(s) for s in seeds]
        policy_obs = self._policy_obs(obs)
        velocity = "velocity" in readouts
        started = time.monotonic()
        with self._gpu_lock:
            if self.sample_mode == "full":
                named = [self._sample_full(policy_obs, seed, velocity) for seed in seeds]
            else:
                named = self._sample_shared(policy_obs, seeds, velocity)
        out = []
        for seed, (actions, extras) in zip(seeds, named):
            flat = _flatten_actions(actions, 1)[0]
            if flat.shape[0] != self.chunk_len:
                raise RuntimeError(f"expected a {self.chunk_len}-step chunk, got {flat.shape}")
            out.append((np.ascontiguousarray(flat, dtype=np.float32), {"seed": seed, **extras}))
        logger.info("seeded request K=%d %.1f ms session=%s", len(seeds), (time.monotonic() - started) * 1e3,
                    getattr(ctx, "session_id", "?")[:8])
        return out

    def _sample_full(self, policy_obs: dict, seed: int, velocity: bool) -> tuple[dict, dict]:
        """v1 verbatim: one complete Gr00tPolicy.get_action under the seed."""
        if not velocity:
            return seeded(lambda: self._policy.get_action(policy_obs), seed), {}
        with _VelocityTrace(self._policy.model.action_head) as trace:
            actions = seeded(lambda: self._policy.get_action(policy_obs), seed)
        return actions, trace.extras()

    def _sample_shared(self, policy_obs: dict, seeds: Sequence[int], velocity: bool) -> list[tuple[dict, dict]]:
        """Gr00tPolicy.get_action with the transforms and backbone hoisted out of the per-seed loop."""
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from gr00t.model.policy import COMPUTE_DTYPE

        policy, model = self._policy, self._policy.model
        head = model.action_head
        obs_copy = {k: v if isinstance(v, np.ndarray) else np.array(v) for k, v in policy_obs.items()}
        if not policy._check_state_is_batched(obs_copy):
            raise RuntimeError("expected batched (B, T, ...) GR00T inputs")
        normalized = policy.apply_transforms(obs_copy)
        results = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            backbone_inputs, action_inputs = model.prepare_input(normalized)
            backbone_outputs = model.backbone(backbone_inputs)
            for seed in seeds:
                # process_backbone_output rewrites its argument in place: give each sample a fresh copy.
                def head_call():
                    return head.get_action(BatchFeature(data=dict(backbone_outputs)), action_inputs)

                if velocity:
                    with _VelocityTrace(head) as trace:
                        pred = seeded(head_call, seed)
                    extras = trace.extras()
                else:
                    pred, extras = seeded(head_call, seed), {}
                model.validate_data(pred, backbone_outputs, is_training=False)
                results.append((pred["action_pred"].float(), extras))
        return [(policy._get_unnormalized_action(action), extras) for action, extras in results]


def _init_arg_names() -> set[str]:
    names: set[str] = set()
    for cls in (HarnessGR00TN15Server, RoboCasa365GR00TModelServer, PredictModelServer):
        names.update(p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "kwargs"))
    return names


def _alias_args(argv: Sequence[str]) -> list[str]:
    """``--model_path X`` / ``--model_path=X`` -> ``--args.model_path ...`` (vla-eval's form)."""
    names, out = _init_arg_names(), []
    for token in argv:
        if token.startswith("--") and token[2:].split("=", 1)[0] in names:
            token = "--args." + token[2:]
        out.append(token)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    from vla_eval.model_servers.serve import run_server

    sys.argv = [sys.argv[0], *_alias_args(sys.argv[1:] if argv is None else argv)]
    run_server(HarnessGR00TN15Server)


if __name__ == "__main__":
    main()
