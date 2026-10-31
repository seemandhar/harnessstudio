"""Mixin that adds the HarnessStudio extension (T1+) to a vla-eval ModelServer.

    class MyServer(HarnessPolicyMixin, PredictModelServer):
        def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
            return [(chunk_for(obs, seed), {"seed": seed}) for seed in seeds]

Requests without the extension fall through to the server's native on_observation, so the same
process still serves plain vla-eval clients.

Rules for sample_chunks (checked by tier 2):
- sample i depends only on obs and seeds[i] (isolated streams);
- the same (obs, seed) gives bitwise-identical actions in every process;
- return full chunks (not the server's executed prefix).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import anyio

from . import protocol


class HarnessPolicyMixin:
    harness_capabilities: frozenset = frozenset({"act", "chunk", "seeded_sampling", "isolated_streams"})

    async def on_observation(self, obs: Mapping[str, Any], ctx) -> None:  # type: ignore[override]
        rest, spec = protocol.split_request(obs)
        if spec is None:
            await super().on_observation(obs, ctx)  # type: ignore[misc]
            return
        proposals = await anyio.to_thread.run_sync(
            self.sample_chunks, rest, list(spec["seeds"]), spec.get("sampler"), tuple(spec.get("readouts") or ()), ctx)
        if len(proposals) != len(spec["seeds"]):
            raise RuntimeError("sample_chunks must return one proposal per seed")
        await ctx.send_action(protocol.make_response(proposals))

    def sample_chunks(self, obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping],
                      readouts: Sequence[str], ctx) -> list[tuple[Any, Mapping[str, Any]]]:
        raise NotImplementedError
