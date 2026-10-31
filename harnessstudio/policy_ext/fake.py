"""A deterministic CPU policy server for tier 1 and tests.

It behaves like a vla-eval PredictModelServer:
- native requests: with chunking="server" it buffers a chunk per session and returns one action per
  observation (ignoring observations while the buffer is non-empty); with chunking="harness" it
  returns the whole chunk. The native noise stream is seeded per episode from the task payload.
- HarnessStudio requests: one chunk per seed; each depends only on (observation, seed).
It records every observation it receives, so tests can check what reached the policy.
"""
from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vla_eval.model_servers.base import ModelServer
from vla_eval.specs import DimSpec

from ..context.trace import fingerprint
from .mixin import HarnessPolicyMixin


def fake_action_spec(action_dim: int) -> dict[str, DimSpec]:
    """LIBERO-like layout when 7-D (position 3, rotation 3, gripper 1); one block otherwise."""
    if action_dim == 7:
        return {"position": DimSpec("position", 3, "delta_xyz"), "rotation": DimSpec("rotation", 3, "axis_angle"),
                "gripper": DimSpec("gripper", 1, "binary_close_positive")}
    return {"actions": DimSpec("actions", action_dim, "fake")}


def _digest32(obs: Mapping[str, Any]) -> int:
    return int(fingerprint(obs)[:8], 16)


class _FakeNative(ModelServer):
    def __init__(self, action_dim: int = 7, chunk_len: int = 6, chunking: str = "harness") -> None:
        if chunking not in ("server", "harness"):
            raise ValueError("chunking must be 'server' or 'harness'")
        self.action_dim, self.chunk_len, self.chunking = action_dim, chunk_len, chunking
        self.received: list[dict] = []          # every observation (without the extension key)
        self._rng: dict[str, np.random.Generator] = {}
        self._buffer: dict[str, deque] = {}

    def chunk(self, obs: Mapping[str, Any], seed: int) -> np.ndarray:
        rng = np.random.default_rng([seed, _digest32(obs)])
        return rng.uniform(-1.0, 1.0, (self.chunk_len, self.action_dim)).astype(np.float32)

    async def on_episode_start(self, config: dict[str, Any], ctx) -> None:
        task = config.get("task") or {}
        key = f"{task.get('name')}|{task.get('episode_idx')}"
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        self._rng[ctx.session_id] = np.random.default_rng(seed)
        self._buffer[ctx.session_id] = deque()

    async def on_observation(self, obs: Mapping[str, Any], ctx) -> None:
        self.received.append({"keys": sorted(obs), "sha256": fingerprint(obs)})
        sid = ctx.session_id
        if sid not in self._rng:
            await self.on_episode_start({}, ctx)
        if self.chunking == "harness":
            seed = int(self._rng[sid].integers(0, 2**32))
            await ctx.send_action({"actions": self.chunk(obs, seed)})
            return
        buf = self._buffer[sid]
        if not buf:
            seed = int(self._rng[sid].integers(0, 2**32))
            buf.extend(self.chunk(obs, seed))
        await ctx.send_action({"actions": buf.popleft()})

    def get_action_spec(self) -> dict[str, DimSpec]:
        return fake_action_spec(self.action_dim)

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {}


class FakePolicyServer(HarnessPolicyMixin, _FakeNative):
    def __init__(self, *args, seeded: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seeded = seeded
        self.harness_capabilities = (frozenset({"act", "chunk", "seeded_sampling", "isolated_streams"})
                                     if seeded else frozenset({"act", "chunk"}))

    async def on_observation(self, obs, ctx) -> None:  # type: ignore[override]
        if "harnessstudio" in obs:
            if not self.seeded:
                raise RuntimeError("this fake policy does not support seeded sampling")
            self.received.append({"keys": sorted(k for k in obs if k != "harnessstudio"),
                                  "sha256": fingerprint({k: v for k, v in obs.items() if k != "harnessstudio"})})
        await super().on_observation(obs, ctx)

    def sample_chunks(self, obs, seeds: Sequence[int], sampler: Optional[Mapping], readouts, ctx):
        return [(self.chunk(obs, seed), {"seed": seed}) for seed in seeds]
