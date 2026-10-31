"""Client side of the policy server: one vla-eval Connection per episode (spec 4)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import anyio
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vla_eval.connection import Connection

from ..core.errors import PolicyError
from ..policy_ext import protocol


@dataclass(frozen=True)
class Proposal:
    actions: np.ndarray            # [H, A], read-only
    extras: Mapping[str, Any]

    @classmethod
    def make(cls, actions: Any, extras: Optional[Mapping[str, Any]] = None) -> "Proposal":
        arr = protocol.as_chunk(actions).copy()
        arr.setflags(write=False)
        return cls(arr, MappingProxyType(dict(extras or {})))


@dataclass(frozen=True)
class PolicyProfile:
    """What the harness knows about a policy server (spec 4.1, 16.1)."""
    capabilities: frozenset = frozenset({"act"})
    chunking: str = "harness"            # "harness": server returns raw chunks; "server": one action per call
    execute_horizon: Optional[int] = None  # native executed prefix of a chunk (None = whole chunk)

    def __post_init__(self):
        if self.chunking not in ("harness", "server"):
            raise ValueError("chunking must be 'harness' or 'server'")
        if self.execute_horizon is not None and self.execute_horizon < 1:
            raise ValueError("execute_horizon must be positive")

    @property
    def seeded(self) -> bool:
        return "seeded_sampling" in self.capabilities

    def native_length(self, chunk: np.ndarray) -> int:
        if self.chunking == "server":
            return 1
        return min(len(chunk), self.execute_horizon or len(chunk))


class PolicyHandle:
    """Never reconnects silently: a dropped connection mid-episode raises PolicyError (spec review)."""

    def __init__(self, url: str, profile: PolicyProfile, timeout: float = 600.0) -> None:
        self.url, self.profile, self.timeout = url, profile, timeout
        self._conn: Optional[Connection] = None
        self._ws: Any = None
        self._lock = anyio.Lock()      # one request in flight per connection
        self._broken: Optional[str] = None
        self.calls = 0          # requests
        self.samples = 0        # chunks returned
        self.seconds = 0.0

    async def open(self, episode_payload: Mapping[str, Any]) -> None:
        self._conn = Connection(self.url, timeout=self.timeout, max_retries=2, backoff_base=1.0)
        await self._conn.connect()
        self._ws = self._conn._ws
        await self._conn.start_episode(dict(episode_payload))

    def _check(self) -> Connection:
        conn = self._conn
        if self._broken is not None:
            raise PolicyError(f"Policy connection unusable after an earlier error: {self._broken}")
        if conn is None or not conn.is_connected or conn._ws is not self._ws:
            raise PolicyError("Policy connection is not open (dropped or reconnected mid-episode)")
        return conn

    async def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        async with self._lock:
            conn = self._check()
            started = time.monotonic()
            try:
                reply = await conn.act(dict(payload))
            except Exception as exc:  # any transport error poisons the connection (a late reply may follow)
                self._broken = f"{type(exc).__name__}: {exc}"
                raise PolicyError(f"Policy request failed: {self._broken}") from exc
            self._check()
            self.seconds += time.monotonic() - started
            self.calls += 1
            return reply

    async def act(self, raw_obs: Mapping[str, Any]) -> np.ndarray:
        reply = await self._request(raw_obs)
        if "actions" not in reply:
            raise PolicyError(f"Native reply has no 'actions' (keys {sorted(reply)})")
        chunk = protocol.as_chunk(reply["actions"])
        self.samples += 1
        return chunk

    async def sample(self, raw_obs: Mapping[str, Any], seeds: Sequence[int], sampler: Optional[Mapping] = None,
                     readouts: Sequence[str] = ()) -> list[Proposal]:
        if not self.profile.seeded:
            raise PolicyError("This policy does not advertise seeded_sampling")
        reply = await self._request(protocol.make_request(raw_obs, seeds, sampler, readouts))
        # Only the requested readouts reach methods: server extras such as the seed would reveal the case.
        keep = set(readouts) | ({"timesteps"} if "velocity" in readouts else set())
        out = [Proposal.make(a, {k: v for k, v in e.items() if k in keep})
               for a, e in protocol.parse_response(reply, len(seeds))]
        self.samples += len(out)
        return out

    async def close(self, result: Optional[Mapping[str, Any]] = None) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            if conn.is_connected and conn._ws is self._ws:
                await conn.end_episode(dict(result or {}))
        except Exception:
            pass
        await conn.close()
