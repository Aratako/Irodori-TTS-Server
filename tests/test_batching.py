from __future__ import annotations

import asyncio

import pytest

from irodori_openai_tts.batching import QueueTimeoutError, SynthesisBatcher
from irodori_tts.inference_runtime import SamplingRequest

RUNTIME = object()


class BadRequestError(Exception):
    pass


class FakeRunBatch:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.batches: list[list[str]] = []

    async def __call__(self, runtime, requests):
        assert runtime is RUNTIME
        self.batches.append([request.text for request in requests])
        await asyncio.sleep(self.delay)
        if any(request.text == "bad" for request in requests):
            raise BadRequestError("bad request")
        return [f"result:{request.text}" for request in requests]


class BlockingRunBatch(FakeRunBatch):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, runtime, requests):
        self.batches.append([request.text for request in requests])
        self.started.set()
        await self.release.wait()
        return [f"result:{request.text}" for request in requests]


def make_batcher(run_batch, *, max_batch_size=8, wait_seconds=0.01, queue_timeout=5.0):
    return SynthesisBatcher(
        max_batch_size=max_batch_size,
        wait_seconds=wait_seconds,
        queue_timeout=queue_timeout,
        run_batch=run_batch,
    )


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=5))


def submit_all_concurrently(batcher, requests):
    async def main():
        return await asyncio.gather(
            *(batcher.submit(RUNTIME, request) for request in requests),
            return_exceptions=True,
        )

    return run(main())


def test_compatible_submits_share_one_batch_in_submit_order():
    run_batch = FakeRunBatch()
    batcher = make_batcher(run_batch)

    results = submit_all_concurrently(
        batcher,
        [SamplingRequest(text="a"), SamplingRequest(text="b"), SamplingRequest(text="c")],
    )

    assert run_batch.batches == [["a", "b", "c"]]
    assert results == ["result:a", "result:b", "result:c"]


def test_incompatible_submits_run_in_separate_batches():
    run_batch = FakeRunBatch()
    batcher = make_batcher(run_batch)

    results = submit_all_concurrently(
        batcher,
        [
            SamplingRequest(text="a", num_steps=10),
            SamplingRequest(text="b", num_steps=20),
            SamplingRequest(text="c", num_steps=10),
        ],
    )

    assert run_batch.batches == [["a", "c"], ["b"]]
    assert results == ["result:a", "result:b", "result:c"]


def test_batches_are_capped_at_max_batch_size():
    run_batch = FakeRunBatch()
    batcher = make_batcher(run_batch, max_batch_size=2)

    submit_all_concurrently(
        batcher,
        [SamplingRequest(text="a"), SamplingRequest(text="b"), SamplingRequest(text="c")],
    )

    assert run_batch.batches == [["a", "b"], ["c"]]


def test_request_without_compatibility_key_runs_alone():
    run_batch = FakeRunBatch()
    batcher = make_batcher(run_batch)

    submit_all_concurrently(
        batcher,
        [
            SamplingRequest(text="a", speaker_uncond_mode="noise"),
            SamplingRequest(text="b", speaker_uncond_mode="noise"),
        ],
    )

    assert run_batch.batches == [["a"], ["b"]]


def test_failed_batch_reruns_each_request_alone():
    run_batch = FakeRunBatch()
    batcher = make_batcher(run_batch)

    results = submit_all_concurrently(
        batcher,
        [SamplingRequest(text="a"), SamplingRequest(text="bad"), SamplingRequest(text="c")],
    )

    assert run_batch.batches == [["a", "bad", "c"], ["a"], ["bad"], ["c"]]
    assert results[0] == "result:a"
    assert isinstance(results[1], BadRequestError)
    assert results[2] == "result:c"


def test_queue_timeout_raises_and_never_runs_the_request():
    run_batch = BlockingRunBatch()
    batcher = make_batcher(run_batch, wait_seconds=0, queue_timeout=0.1)

    async def main():
        first = asyncio.create_task(batcher.submit(RUNTIME, SamplingRequest(text="a")))
        await run_batch.started.wait()
        with pytest.raises(QueueTimeoutError):
            await batcher.submit(RUNTIME, SamplingRequest(text="b"))
        run_batch.release.set()
        await first

    run(main())

    assert run_batch.batches == [["a"]]


def test_cancel_before_start_never_runs_the_request():
    run_batch = BlockingRunBatch()
    batcher = make_batcher(run_batch, wait_seconds=0)

    async def main():
        first = asyncio.create_task(batcher.submit(RUNTIME, SamplingRequest(text="a")))
        await run_batch.started.wait()
        second = asyncio.create_task(batcher.submit(RUNTIME, SamplingRequest(text="b")))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        run_batch.release.set()
        await first

    run(main())

    assert run_batch.batches == [["a"]]


def test_cancel_after_start_keeps_the_rest_of_the_batch():
    run_batch = BlockingRunBatch()
    batcher = make_batcher(run_batch)

    async def main():
        first = asyncio.create_task(batcher.submit(RUNTIME, SamplingRequest(text="a")))
        second = asyncio.create_task(batcher.submit(RUNTIME, SamplingRequest(text="b")))
        await run_batch.started.wait()
        first.cancel()
        run_batch.release.set()
        return await second

    assert run(main()) == "result:b"
    assert run_batch.batches == [["a", "b"]]


def test_submit_all_fails_with_the_chunk_error_and_drops_abandoned_chunks():
    # The delay keeps the worker busy rerunning "b" while the failure cancels "c" and "d".
    run_batch = FakeRunBatch(delay=0.05)
    batcher = make_batcher(run_batch, max_batch_size=3)

    async def main():
        with pytest.raises(BadRequestError):
            await batcher.submit_all(
                RUNTIME, [SamplingRequest(text=text) for text in ("bad", "b", "c", "d")]
            )
        pending = list(batcher._pending)
        # Runs only after the worker has finished the reruns above.
        await batcher.submit(RUNTIME, SamplingRequest(text="after"))
        return pending

    assert run(main()) == []
    assert run_batch.batches == [["bad", "b", "c"], ["bad"], ["b"], ["after"]]


def test_submit_all_applies_the_queue_timeout_only_until_the_first_chunk_starts():
    # Three batches of 0.05s each: the last one starts after the 0.08s queue timeout.
    run_batch = FakeRunBatch(delay=0.05)
    batcher = make_batcher(run_batch, max_batch_size=2, queue_timeout=0.08)
    texts = ["a", "b", "c", "d", "e", "f"]

    results = run(batcher.submit_all(RUNTIME, [SamplingRequest(text=text) for text in texts]))

    assert results == [f"result:{text}" for text in texts]
    assert run_batch.batches == [["a", "b"], ["c", "d"], ["e", "f"]]
