from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from irodori_tts.inference_runtime import (
    SamplingRequest,
    SamplingResult,
    batch_compatibility_key,
)

RunBatch = Callable[[Any, list[SamplingRequest]], Awaitable[list[SamplingResult]]]


class QueueTimeoutError(RuntimeError):
    pass


@dataclass(eq=False)
class _Item:
    runtime: Any
    request: SamplingRequest
    key: tuple | None
    arrived_at: float
    started: asyncio.Event = field(default_factory=asyncio.Event)
    future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


class SynthesisBatcher:
    """Queue synthesis requests and run compatible ones together, one batch at a time."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        wait_seconds: float,
        queue_timeout: float,
        run_batch: RunBatch,
    ) -> None:
        self._max_batch_size = max_batch_size
        self._wait_seconds = wait_seconds
        self._queue_timeout = queue_timeout
        self._run_batch = run_batch
        self._pending: list[_Item] = []
        self._arrived = asyncio.Event()
        self._worker: asyncio.Task | None = None

    async def submit(self, runtime: Any, request: SamplingRequest) -> SamplingResult:
        return await self._submit(runtime, request, self._queue_timeout)

    async def submit_all(
        self, runtime: Any, requests: list[SamplingRequest]
    ) -> list[SamplingResult]:
        # Later chunks queue behind the first, so only the first waits with the queue timeout.
        tasks = [
            asyncio.create_task(
                self._submit(runtime, request, self._queue_timeout if index == 0 else None)
            )
            for index, request in enumerate(requests)
        ]
        try:
            return await asyncio.gather(*tasks)
        finally:
            # Queued siblings leave the queue; started ones finish and their results are dropped.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _submit(
        self, runtime: Any, request: SamplingRequest, queue_timeout: float | None
    ) -> SamplingResult:
        loop = asyncio.get_running_loop()
        item = _Item(runtime, request, batch_compatibility_key(request), loop.time())
        self._pending.append(item)
        self._arrived.set()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work())
        try:
            await asyncio.wait_for(item.started.wait(), queue_timeout)
        except asyncio.TimeoutError as exc:
            if item in self._pending:
                self._pending.remove(item)
                raise QueueTimeoutError from exc
        except asyncio.CancelledError:
            if item in self._pending:
                self._pending.remove(item)
            item.future.cancel()
            raise
        return await item.future

    async def _work(self) -> None:
        while self._pending:
            batch = await self._next_batch()
            for item in batch:
                self._pending.remove(item)
                item.started.set()
            if batch:
                await self._run(batch)

    async def _run(self, batch: list[_Item]) -> None:
        try:
            results = await self._run_batch(batch[0].runtime, [item.request for item in batch])
            for item, result in zip(batch, results, strict=True):
                if not item.future.done():
                    item.future.set_result(result)
        except Exception as exc:
            if len(batch) == 1:
                if not batch[0].future.done():
                    batch[0].future.set_exception(exc)
                return
            # Each item settles as soon as its own rerun ends, so callers can react early.
            for item in batch:
                if not item.future.done():
                    await self._run([item])

    async def _next_batch(self) -> list[_Item]:
        loop = asyncio.get_running_loop()
        while self._pending:
            oldest = self._pending[0]
            batch = [oldest]
            if oldest.key is not None:
                batch += [item for item in self._pending[1:] if item.key == oldest.key]
            batch = batch[: self._max_batch_size]
            remaining = oldest.arrived_at + self._wait_seconds - loop.time()
            if oldest.key is None or len(batch) == self._max_batch_size or remaining <= 0:
                return batch
            self._arrived.clear()
            try:
                await asyncio.wait_for(self._arrived.wait(), remaining)
            except asyncio.TimeoutError:
                pass
        return []
