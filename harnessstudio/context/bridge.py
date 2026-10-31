"""Coroutine bridge (spec 3.2): the method runs as its own task; each benchmark observation releases
one action from it.

Server side:  action = await bridge.first() / await bridge.next()    -> None once the method is done
Method side:  await bridge.emit(action)                                -> returns when the next observation is in
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

import anyio
from anyio.abc import TaskGroup

from ..core.errors import BudgetExceeded

_DONE = object()


class Bridge:
    def __init__(self) -> None:
        self._act_send, self._act_recv = anyio.create_memory_object_stream(1)
        self._obs_send, self._obs_recv = anyio.create_memory_object_stream(1)
        self._scope: Optional[anyio.CancelScope] = None
        self._started = False
        self.done = False
        self.error: Optional[BaseException] = None
        self.termination: Optional[str] = None

    def start(self, tg: TaskGroup, runner: Callable[[], Awaitable[Any]]) -> None:
        if self._started:
            raise RuntimeError("Bridge already started")
        self._started = True
        tg.start_soon(self._main, runner)

    async def _main(self, runner: Callable[[], Awaitable[Any]]) -> None:
        # Never let a method exception reach the server's task group (spec 16.5).
        with anyio.CancelScope() as scope:
            self._scope = scope
            try:
                await runner()
                self.termination = self.termination or "method_returned"
            except BudgetExceeded:
                self.termination = "budget_exhausted"
            except anyio.get_cancelled_exc_class():
                self.termination = self.termination or "cancelled"
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, episode becomes invalid
                self.error = exc
                self.termination = "method_error"
            finally:
                self.done = True
                try:
                    self._act_send.send_nowait(_DONE)
                except (anyio.WouldBlock, anyio.BrokenResourceError, anyio.ClosedResourceError):
                    pass

    # -- server side --------------------------------------------------------
    async def first(self) -> Optional[Any]:
        return await self._receive()

    async def next(self) -> Optional[Any]:
        if self.done:
            return None
        await self._obs_send.send(None)   # wake the method: a new observation is in the context
        return await self._receive()

    async def _receive(self) -> Optional[Any]:
        item = await self._act_recv.receive()
        return None if item is _DONE else item

    def cancel(self) -> None:
        if self._scope is not None:
            self._scope.cancel()

    # -- method side --------------------------------------------------------
    async def emit(self, action: Any) -> None:
        await self._act_send.send(action)
        await self._obs_recv.receive()
