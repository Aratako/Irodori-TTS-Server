from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import threading
import time

import pytest
import torch
from fastapi.testclient import TestClient

from irodori_openai_tts import app as main
from irodori_openai_tts.runtime import RuntimeLoadTimeoutError
from irodori_openai_tts.voices import VoiceRegistry
from irodori_tts.inference_runtime import SamplingResult


class FakeRuntime:
    def __init__(self, exc: BaseException | None = None) -> None:
        self.exc = exc
        self.requests = []
        self.texts: list[str] = []
        self.thread_ids: list[int] = []

    def synthesize(self, req, *, log_fn=None):
        self.requests.append(req)
        self.texts.append(req.text)
        self.thread_ids.append(threading.get_ident())
        if self.exc is not None:
            raise self.exc
        if log_fn is not None:
            log_fn("fake synthesize")
        audio = torch.zeros(1, max(1, len(req.text)) * 10)
        return SamplingResult(
            audio=audio,
            audios=[audio],
            sample_rate=1000,
            stage_timings=[],
            total_to_decode=0.1,
            used_seed=123,
            messages=[],
        )


class FakeRuntimeManager:
    def __init__(self, runtime=None, exc: BaseException | None = None) -> None:
        self.runtime = runtime
        self.exc = exc
        self.checkpoint_path = None
        self.is_loaded = runtime is not None
        self.is_loading = exc is not None
        self.thread_ids: list[int] = []
        self.unload_calls = 0

    def unload(self) -> bool:
        self.unload_calls += 1
        if self.exc is not None:
            raise self.exc
        unloaded = bool(self.is_loaded)
        self.is_loaded = False
        self.runtime = None
        return unloaded

    def get(self):
        self.thread_ids.append(threading.get_ident())
        if self.exc is not None:
            raise self.exc
        return self.runtime


class NeverAvailableSemaphore:
    async def acquire(self):
        await asyncio.sleep(1)

    def release(self):
        raise AssertionError("release should not be called when acquire times out")


class RecordingSemaphore:
    def __init__(self) -> None:
        self.released = False

    async def acquire(self):
        return True

    def release(self):
        self.released = True


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        events.append((event, json.loads(data)))
    return events


def test_health_does_not_load_model(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager())

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["runtime"]["loaded"] is False
    assert body["runtime"]["loading"] is False
    assert body["runtime"]["max_concurrent_synthesis"] == 1
    assert body["runtime"]["synthesis_wait_timeout"] == 300.0
    assert body["voices"]["dir_exists"] is True
    assert body["defaults"]["chunk_min_chars"] == main.settings.default_chunk_min_chars
    assert body["defaults"]["first_sentence_chunk_min_chars"] is None


def test_models_lists_configured_model():
    response = TestClient(main.app).get("/v1/models")

    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == main.settings.model_name


@pytest.mark.parametrize(
    ("server_default", "request_options", "expected_steps"),
    [
        (None, {}, None),
        (None, {"irodori": {"num_steps": None}}, None),
        (40, {}, 40),
        (4, {}, 4),
        (None, {"num_steps": 8}, 8),
        (40, {"num_steps": 8}, 8),
        (40, {"irodori": {"num_steps": 4}}, 4),
        (40, {"num_steps": 8, "irodori": {"num_steps": 1}}, 1),
        (40, {"num_steps": 8, "irodori": {"num_steps": None}}, 8),
    ],
)
def test_speech_sampling_step_precedence(
    monkeypatch, server_default, request_options, expected_steps
):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    monkeypatch.setattr(main.settings, "default_num_steps", server_default)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            **request_options,
        },
    )

    assert response.status_code == 200
    assert len(runtime.requests) == 1
    assert runtime.requests[0].num_steps == expected_steps


def test_auth_required_when_api_key_is_configured(monkeypatch):
    monkeypatch.setattr(main.settings, "api_key", "secret")
    client = TestClient(main.app)

    missing = client.get("/v1/models")
    wrong = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    ok = client.get("/v1/models", headers={"Authorization": "Bearer secret"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 200


def test_voice_upload_list_get_replace_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "voice_registry", VoiceRegistry(main.settings))
    client = TestClient(main.app)

    created = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"old", "audio/wav")},
        data={"voice_id": "sample"},
    )
    listed = client.get("/v1/audio/voices")
    fetched = client.get("/v1/audio/voices/sample")
    replaced = client.put(
        "/v1/audio/voices/sample",
        files={"file": ("sample.flac", b"new", "audio/flac")},
    )
    deleted = client.delete("/v1/audio/voices/sample")
    missing = client.get("/v1/audio/voices/sample")

    assert created.status_code == 201
    assert created.json()["filename"] == "sample.wav"
    assert any(item["id"] == "sample" for item in listed.json()["data"])
    assert fetched.status_code == 200
    assert replaced.status_code == 200
    assert replaced.json()["filename"] == "sample.flac"
    assert not (tmp_path / "sample.wav").exists()
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert missing.status_code == 404


def test_voice_upload_rejects_duplicate_bad_id_and_bad_extension(monkeypatch):
    monkeypatch.setattr(main, "voice_registry", VoiceRegistry(main.settings))
    client = TestClient(main.app)

    assert (
        client.post(
            "/v1/audio/voices",
            files={"file": ("sample.wav", b"old", "audio/wav")},
            data={"voice_id": "sample"},
        ).status_code
        == 201
    )

    duplicate = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"old", "audio/wav")},
        data={"voice_id": "sample"},
    )
    bad_id = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.wav", b"old", "audio/wav")},
        data={"voice_id": "../sample"},
    )
    bad_extension = client.post(
        "/v1/audio/voices",
        files={"file": ("sample.txt", b"text", "text/plain")},
        data={"voice_id": "text"},
    )

    assert duplicate.status_code == 409
    assert bad_id.status_code == 400
    assert bad_extension.status_code == 400


def test_speech_returns_503_when_model_is_loading(monkeypatch):
    monkeypatch.setattr(
        main,
        "runtime_manager",
        FakeRuntimeManager(exc=RuntimeLoadTimeoutError("Model is still loading.")),
    )

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "wav",
        },
    )

    assert response.status_code == 503
    assert "Model is still loading" in response.json()["error"]["message"]


def test_speech_stream_format_sse_emits_audio_chunk(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "wav",
            "stream_format": "sse",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert runtime.texts == ["こんにちは。"]
    events = sse_events(response.text)
    assert [event for event, _data in events] == ["audio_chunk", "done"]
    audio_chunk = events[0][1]
    assert audio_chunk["index"] == 0
    assert audio_chunk["text"] == "こんにちは。"
    assert audio_chunk["format"] == "wav"
    assert audio_chunk["media_type"] == "audio/wav"
    assert audio_chunk["seed"] == 123
    assert audio_chunk["total_to_decode"] == 0.1
    assert base64.b64decode(audio_chunk["audio_base64"]).startswith(b"RIFF")
    assert events[1][1] == {"chunks": 1}


def test_speech_rejects_unknown_stream_format_before_loading_runtime(monkeypatch):
    runtime = FakeRuntime()
    manager = FakeRuntimeManager(runtime=runtime)
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "stream_format": "jsonl",
        },
    )

    assert response.status_code == 400
    assert "stream_format" in response.json()["error"]["message"]
    assert manager.thread_ids == []
    assert runtime.texts == []


def test_speech_rejects_unsupported_model_before_loading_runtime(monkeypatch):
    runtime = FakeRuntime()
    manager = FakeRuntimeManager(runtime=runtime)
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "こんにちは。",
            "voice": "none",
        },
    )

    assert response.status_code == 400
    assert "Unsupported model" in response.json()["error"]["message"]
    assert manager.thread_ids == []
    assert runtime.texts == []


def test_speech_rejects_whitespace_input_before_loading_runtime(monkeypatch):
    runtime = FakeRuntime()
    manager = FakeRuntimeManager(runtime=runtime)
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "   \n\t ",
            "voice": "none",
        },
    )

    assert response.status_code == 400
    assert "non-whitespace" in response.json()["error"]["message"]
    assert manager.thread_ids == []
    assert runtime.texts == []


def test_speech_rejects_unknown_response_format(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "xyz",
        },
    )

    assert response.status_code == 400
    assert "Unsupported response_format" in response.json()["error"]["message"]
    assert runtime.texts == []


def test_speech_rejects_invalid_duration_scale_before_loading_runtime(monkeypatch):
    runtime = FakeRuntime()
    manager = FakeRuntimeManager(runtime=runtime)
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "irodori": {"duration_scale": 0},
        },
    )

    assert response.status_code == 400
    assert "duration_scale" in response.json()["error"]["message"]
    assert manager.thread_ids == []
    assert runtime.texts == []


def test_speech_rejects_invalid_duration_range_before_loading_runtime(monkeypatch):
    runtime = FakeRuntime()
    manager = FakeRuntimeManager(runtime=runtime)
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "irodori": {"min_seconds": 10, "max_seconds": 1},
        },
    )

    assert response.status_code == 400
    assert "max_seconds" in response.json()["error"]["message"]
    assert manager.thread_ids == []
    assert runtime.texts == []


@pytest.mark.parametrize(
    "field",
    ["duration_scale", "chunk_min_chars", "first_sentence_chunk_min_chars", "ref_normalize_db"],
)
def test_speech_rejects_invalid_top_level_numeric_extra_before_loading_runtime(
    monkeypatch,
    field,
):
    runtime = FakeRuntime()
    manager = FakeRuntimeManager(runtime=runtime)
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            field: "bad",
        },
    )

    assert response.status_code == 400
    assert field in response.json()["error"]["message"]
    assert manager.thread_ids == []
    assert runtime.texts == []


def test_speech_returns_400_when_runtime_rejects_request(monkeypatch):
    runtime = FakeRuntime(exc=ValueError("runtime validation failed"))
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "wav",
        },
    )

    assert response.status_code == 400
    assert "runtime validation failed" in response.json()["error"]["message"]


def test_speech_returns_400_when_lora_adapter_is_missing(monkeypatch):
    runtime = FakeRuntime(exc=FileNotFoundError("LoRA adapter directory not found"))
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "irodori": {"lora_adapter": "/missing/adapter"},
        },
    )

    assert response.status_code == 400
    assert "LoRA adapter directory not found" in response.json()["error"]["message"]


def test_speech_returns_400_when_lora_conflicts_with_compile(monkeypatch):
    runtime = FakeRuntime(exc=RuntimeError("Dynamic LoRA loading is not compatible"))
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "irodori": {"lora_adapter": "/models/adapters/speaker-a"},
        },
    )

    assert response.status_code == 400
    assert "Dynamic LoRA loading is not compatible" in response.json()["error"]["message"]


def test_speech_uses_uploaded_voice_and_returns_headers(tmp_path, monkeypatch):
    (tmp_path / "speaker.wav").write_bytes(b"fake")
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    monkeypatch.setattr(main, "voice_registry", VoiceRegistry(main.settings))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "speaker",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.headers["x-irodori-seed"] == "123"
    assert response.headers["x-irodori-total-to-decode"] == "0.100000"
    assert response.content.startswith(b"RIFF")
    assert runtime.texts == ["こんにちは。"]


def test_speech_passes_lora_adapter_option(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "irodori": {"lora_adapter": "/models/adapters/speaker-a"},
        },
    )

    assert response.status_code == 200
    assert runtime.requests[0].lora_adapter == "/models/adapters/speaker-a"


@pytest.mark.parametrize(
    ("field", "paths"),
    [
        ("ref_wavs", ["/voices/clip-1.wav", "/voices/clip-2.wav"]),
        ("ref_latents", ["/voices/clip-1.pt", "/voices/clip-2.pt"]),
    ],
)
def test_speech_passes_multiple_reference_options(monkeypatch, field, paths):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "irodori": {field: paths},
        },
    )

    assert response.status_code == 200
    assert getattr(runtime.requests[0], field) == paths
    assert runtime.requests[0].max_ref_seconds is None


@pytest.mark.parametrize("value", [[], [""], "clip.wav"])
def test_speech_rejects_invalid_multiple_reference_options(monkeypatch, value):
    runtime = FakeRuntime()
    manager = FakeRuntimeManager(runtime=runtime)
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "ref_wavs": value,
        },
    )

    assert response.status_code in {400, 422}
    assert manager.thread_ids == []
    assert runtime.requests == []


@pytest.mark.parametrize(
    "irodori",
    [
        {"ref_wav": "/voices/one.wav", "ref_wavs": ["/voices/two.wav"]},
        {"ref_latent": "/voices/one.pt", "ref_latents": ["/voices/two.pt"]},
        {"ref_wavs": ["/voices/one.wav"], "ref_latents": ["/voices/one.pt"]},
        {"no_ref": True, "ref_wavs": ["/voices/one.wav"]},
        {"ref_embed": "/voices/speaker.safetensors", "ref_wavs": ["/voices/one.wav"]},
    ],
)
def test_speech_rejects_conflicting_reference_options_before_loading_runtime(monkeypatch, irodori):
    runtime = FakeRuntime()
    manager = FakeRuntimeManager(runtime=runtime)
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "irodori": irodori,
        },
    )

    assert response.status_code == 400
    assert manager.thread_ids == []
    assert runtime.requests == []


def test_speech_runs_model_load_and_synthesis_in_executor(monkeypatch):
    runtime = FakeRuntime()
    manager = FakeRuntimeManager(runtime=runtime)
    monkeypatch.setattr(main, "runtime_manager", manager)
    caller_thread = threading.get_ident()

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert manager.thread_ids
    assert runtime.thread_ids
    assert caller_thread not in manager.thread_ids
    assert caller_thread not in runtime.thread_ids


def test_speech_returns_503_when_synthesis_queue_times_out(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    monkeypatch.setattr(main.settings, "synthesis_wait_timeout", 0.01)
    monkeypatch.setattr(main.settings, "max_concurrent_synthesis", 1)
    monkeypatch.setattr(main, "_synthesis_semaphore", NeverAvailableSemaphore())
    monkeypatch.setattr(main, "_synthesis_semaphore_limit", 1)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "wav",
        },
    )

    assert response.status_code == 503
    assert "Synthesis queue is full" in response.json()["error"]["message"]
    assert runtime.texts == []


def test_speech_stream_format_sse_returns_queue_timeout_as_error_event(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    monkeypatch.setattr(main.settings, "synthesis_wait_timeout", 0.01)
    monkeypatch.setattr(main.settings, "max_concurrent_synthesis", 1)
    monkeypatch.setattr(main, "_synthesis_semaphore", NeverAvailableSemaphore())
    monkeypatch.setattr(main, "_synthesis_semaphore_limit", 1)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "wav",
            "stream_format": "sse",
        },
    )

    assert response.status_code == 200
    assert sse_events(response.text) == [
        (
            "error",
            {
                "error": {
                    "message": "Synthesis queue is full. Retry after a moment. timeout=0.0s",
                    "type": "server_error",
                    "param": None,
                    "code": "synthesis_queue_timeout",
                }
            },
        )
    ]
    assert runtime.texts == []


def test_speech_stream_format_sse_returns_runtime_error_event(monkeypatch):
    runtime = FakeRuntime(exc=ValueError("runtime validation failed"))
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "wav",
            "stream_format": "sse",
        },
    )

    assert response.status_code == 200
    assert sse_events(response.text) == [
        (
            "error",
            {
                "error": {
                    "message": "runtime validation failed",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_request",
                }
            },
        )
    ]


def test_speech_stream_format_sse_releases_slot_before_yielding_chunk(monkeypatch):
    semaphore = RecordingSemaphore()
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    async def fake_acquire_synthesis_slot():
        await semaphore.acquire()
        return semaphore

    monkeypatch.setattr(main, "_acquire_synthesis_slot", fake_acquire_synthesis_slot)

    async def run_test():
        response = main._stream_speech_response(
            main.SamplingRequest(text="こんにちは。", no_ref=True),
            ["こんにちは。"],
            "wav",
            0.0,
        )
        chunk = await response.body_iterator.__anext__()
        assert chunk.startswith("event: audio_chunk\n")
        assert semaphore.released is True

    asyncio.run(run_test())


def test_speech_stream_format_sse_holds_slot_until_cancelled_synthesis_finishes(monkeypatch):
    semaphore = RecordingSemaphore()

    async def run_test():
        started = asyncio.Event()
        release_synthesis = asyncio.Event()
        main_thread_loop = asyncio.get_running_loop()

        class SlowRuntime:
            def synthesize(self, req, *, log_fn=None):
                main_thread_loop.call_soon_threadsafe(started.set)
                asyncio.run_coroutine_threadsafe(
                    release_synthesis.wait(), main_thread_loop
                ).result()
                audio = torch.zeros(1, max(1, len(req.text)) * 10)
                return SamplingResult(
                    audio=audio,
                    audios=[audio],
                    sample_rate=1000,
                    stage_timings=[],
                    total_to_decode=0.1,
                    used_seed=123,
                    messages=[],
                )

        monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=SlowRuntime()))

        response = main._stream_speech_response(
            main.SamplingRequest(text="こんにちは。", no_ref=True),
            ["こんにちは。"],
            "wav",
            0.0,
        )
        stream = response.body_iterator
        task = asyncio.create_task(stream.__anext__())

        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        assert semaphore.released is False

        release_synthesis.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert semaphore.released is True

    async def fake_acquire_synthesis_slot():
        await semaphore.acquire()
        return semaphore

    monkeypatch.setattr(main, "_acquire_synthesis_slot", fake_acquire_synthesis_slot)

    asyncio.run(run_test())


def test_speech_chunking_can_be_disabled_per_request(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    text = (
        "これは短い文です。これはまだ同じチャンクに残る文です。"
        "ここまでで十分長くなったので分割されます。最後です。"
    )

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": text,
            "voice": "none",
            "response_format": "wav",
            "irodori": {
                "chunking_enabled": False,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.texts == [text]


def test_speech_chunking_splits_only_after_min_chars(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    text = (
        "これは短い文です。これはまだ同じチャンクに残る文です。"
        "ここまでで十分長くなったので分割されます。最後です。"
    )

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": text,
            "voice": "none",
            "response_format": "wav",
            "irodori": {
                "chunking_enabled": True,
                "chunk_min_chars": 35,
            },
        },
    )

    assert response.status_code == 200
    assert len(runtime.texts) == 2
    assert runtime.texts[0].endswith("。")
    assert runtime.texts[1] == "最後です。"


def test_speech_chunking_uses_first_sentence_min_chars_only_for_first_split(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    monkeypatch.setattr(main.settings, "default_first_sentence_chunk_min_chars", 10)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "最初は速く、すぐ返します。次は長くて、通常のままです。",
            "voice": "none",
            "response_format": "wav",
            "irodori": {
                "chunking_enabled": True,
                "chunk_min_chars": 80,
                "first_sentence_chunk_min_chars": 1,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.texts == [
        "最初は速く、",
        "すぐ返します。次は長くて、通常のままです。",
    ]


def test_speech_chunking_uses_default_first_sentence_min_chars(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    monkeypatch.setattr(main.settings, "default_first_sentence_chunk_min_chars", 1)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "最初は速く、すぐ返します。次は長くて、通常のままです。",
            "voice": "none",
            "response_format": "wav",
            "irodori": {
                "chunking_enabled": True,
                "chunk_min_chars": 80,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.texts == [
        "最初は速く、",
        "すぐ返します。次は長くて、通常のままです。",
    ]


def test_speech_chunking_explicit_null_disables_default_first_sentence_min_chars(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    monkeypatch.setattr(main.settings, "default_first_sentence_chunk_min_chars", 1)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "最初は速く、すぐ返します。次は長くて、通常のままです。",
            "voice": "none",
            "response_format": "wav",
            "irodori": {
                "chunking_enabled": True,
                "chunk_min_chars": 80,
                "first_sentence_chunk_min_chars": None,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.texts == ["最初は速く、すぐ返します。次は長くて、通常のままです。"]


def test_speech_chunking_does_not_split_shorter_first_sentence(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "短い。これは続きです。",
            "voice": "none",
            "response_format": "wav",
            "irodori": {
                "chunking_enabled": True,
                "chunk_min_chars": 80,
                "first_sentence_chunk_min_chars": 10,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.texts == ["短い。これは続きです。"]


def test_speech_stream_format_sse_emits_each_chunk(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    text = (
        "これは短い文です。これはまだ同じチャンクに残る文です。"
        "ここまでで十分長くなったので分割されます。最後です。"
    )

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": text,
            "voice": "none",
            "response_format": "wav",
            "stream_format": "sse",
            "irodori": {
                "chunking_enabled": True,
                "chunk_min_chars": 35,
            },
        },
    )

    assert response.status_code == 200
    assert len(runtime.texts) == 2
    events = sse_events(response.text)
    assert [event for event, _data in events] == ["audio_chunk", "audio_chunk", "done"]
    assert [data["text"] for _event, data in events[:2]] == runtime.texts
    assert events[0][1]["index"] == 0
    assert events[1][1]["index"] == 1
    assert events[2][1] == {"chunks": 2}


def test_speech_chunking_skips_when_seconds_is_explicit(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    text = (
        "これは短い文です。これはまだ同じチャンクに残る文です。"
        "ここまでで十分長くなったので分割されます。最後です。"
    )

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": text,
            "voice": "none",
            "response_format": "wav",
            "irodori": {
                "chunking_enabled": True,
                "chunk_min_chars": 35,
                "seconds": 5.0,
            },
        },
    )

    assert response.status_code == 200
    assert runtime.texts == [text]


def test_speech_rejects_invalid_chunk_min_chars(monkeypatch):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "wav",
            "irodori": {
                "chunking_enabled": True,
                "chunk_min_chars": 0,
            },
        },
    )

    assert response.status_code == 400
    assert "chunk_min_chars" in response.json()["error"]["message"]


@pytest.mark.parametrize("value", [0, -1])
def test_speech_rejects_invalid_first_sentence_chunk_min_chars(monkeypatch, value):
    runtime = FakeRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "こんにちは。",
            "voice": "none",
            "response_format": "wav",
            "irodori": {
                "chunking_enabled": True,
                "first_sentence_chunk_min_chars": value,
            },
        },
    )

    assert response.status_code == 400
    assert "first_sentence_chunk_min_chars" in response.json()["error"]["message"]
    assert runtime.texts == []


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (0, [False] * 5),
        (1, [True] * 5),
        (3, [False, False, True, False, False]),
    ],
)
def test_empty_cache_due_follows_configured_interval(monkeypatch, interval, expected):
    monkeypatch.setattr(main.settings, "empty_cache_interval", interval)
    monkeypatch.setattr(main, "_synthesis_since_empty_cache", 0)

    assert [main._empty_cache_due() for _ in range(5)] == expected


def test_empty_cache_due_counts_failed_syntheses(monkeypatch):
    runtime = FakeRuntime(exc=RuntimeError("boom"))
    monkeypatch.setattr(main.settings, "empty_cache_interval", 2)
    monkeypatch.setattr(main, "_synthesis_since_empty_cache", 0)
    released: list[object] = []
    monkeypatch.setattr(main, "_release_device_cache", released.append)
    request = main._build_sampling_request(
        main.SpeechRequest(model="irodori-tts", input="こんにちは。", voice="none"),
        main.VoiceSpec(voice_id="none", no_ref=True),
    )

    for _ in range(2):
        with pytest.raises(RuntimeError):
            main._synthesize_once(runtime, request)

    assert released == [runtime]


def test_openai_speed_maps_to_inverse_duration_scale():
    payload = main.SpeechRequest(
        model="irodori-tts",
        input="こんにちは。",
        voice="none",
        speed=1.25,
    )
    voice = main.VoiceSpec(voice_id="none", no_ref=True)

    request = main._build_sampling_request(payload, voice)

    assert request.duration_scale == 0.8


def test_unload_releases_runtime_and_reports_state(monkeypatch):
    manager = FakeRuntimeManager(runtime=FakeRuntime())
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post("/unload")

    assert response.status_code == 200
    assert response.json() == {"unloaded": True, "loaded": False}
    assert manager.unload_calls == 1


def test_unload_is_idempotent_when_nothing_is_loaded(monkeypatch):
    manager = FakeRuntimeManager(runtime=None)
    monkeypatch.setattr(main, "runtime_manager", manager)

    response = TestClient(main.app).post("/unload")

    assert response.status_code == 200
    assert response.json() == {"unloaded": False, "loaded": False}


def test_unload_returns_503_when_model_is_loading(monkeypatch):
    monkeypatch.setattr(
        main,
        "runtime_manager",
        FakeRuntimeManager(exc=RuntimeLoadTimeoutError("Model is still loading.")),
    )

    response = TestClient(main.app).post("/unload")

    assert response.status_code == 503


def test_unload_requires_api_key_when_configured(monkeypatch):
    manager = FakeRuntimeManager(runtime=FakeRuntime())
    monkeypatch.setattr(main, "runtime_manager", manager)
    monkeypatch.setattr(main.settings, "api_key", "secret")

    client = TestClient(main.app)
    unauthorized = client.post("/unload")

    assert unauthorized.status_code == 401
    assert manager.unload_calls == 0

    authorized = client.post("/unload", headers={"Authorization": "Bearer secret"})

    assert authorized.status_code == 200
    assert manager.unload_calls == 1


def test_unload_conflicts_with_in_flight_synthesis(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class BlockingRuntime(FakeRuntime):
        def synthesize(self, req, *, log_fn=None):
            entered.set()
            assert release.wait(timeout=10)
            return super().synthesize(req, log_fn=log_fn)

    manager = FakeRuntimeManager(runtime=BlockingRuntime())
    monkeypatch.setattr(main, "runtime_manager", manager)

    with TestClient(main.app) as client:
        speech: list[int] = []

        def run_speech():
            speech.append(
                client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "irodori-tts",
                        "input": "こんにちは。",
                        "voice": "none",
                        "response_format": "wav",
                    },
                ).status_code
            )

        thread = threading.Thread(target=run_speech)
        thread.start()
        try:
            assert entered.wait(timeout=10)
            response = client.post("/unload", params={"timeout": 0.05})
        finally:
            release.set()
            thread.join(timeout=10)

    assert response.status_code == 409
    assert "still in flight" in response.json()["error"]["message"]
    assert manager.unload_calls == 0
    assert speech == [200]


def test_stream_holds_the_gate_across_chunk_boundaries(monkeypatch):
    """The per-chunk semaphore goes slack between chunks; the gate must not.

    Regression guard: unloading between two SSE chunks would delete the model
    the stream is still holding.
    """
    observed: list[int] = []

    class GateObservingRuntime(FakeRuntime):
        def synthesize(self, req, *, log_fn=None):
            gate = main._synthesis_gate
            observed.append(-1 if gate is None else gate.active)
            return super().synthesize(req, log_fn=log_fn)

    runtime = GateObservingRuntime()
    monkeypatch.setattr(main, "runtime_manager", FakeRuntimeManager(runtime=runtime))
    monkeypatch.setattr(main.settings, "default_first_sentence_chunk_min_chars", 1)

    response = TestClient(main.app).post(
        "/v1/audio/speech",
        json={
            "model": "irodori-tts",
            "input": "最初は速く、すぐ返します。",
            "voice": "none",
            "response_format": "wav",
            "stream_format": "sse",
        },
    )

    assert response.status_code == 200
    assert len(runtime.texts) > 1, "test needs a multi-chunk stream to be meaningful"
    assert observed == [1] * len(runtime.texts)


def test_synthesis_gate_blocks_unload_until_the_request_finishes():
    async def scenario():
        gate = main.SynthesisGate()
        order: list[str] = []

        async def synthesize():
            async with gate.synthesis():
                order.append("synthesis:start")
                await asyncio.sleep(0.05)
                order.append("synthesis:end")

        async def unload():
            await asyncio.sleep(0.01)
            async with gate.exclusive(timeout=5.0):
                order.append("unload")

        await asyncio.gather(synthesize(), unload())
        return order

    assert asyncio.run(scenario()) == ["synthesis:start", "synthesis:end", "unload"]


def test_synthesis_gate_raises_busy_when_the_wait_times_out():
    async def scenario():
        gate = main.SynthesisGate()
        async with gate.synthesis():
            with pytest.raises(main.SynthesisBusyError):
                async with gate.exclusive(timeout=0.01):
                    pass
        # The failed attempt must not leave new synthesis blocked out.
        async with gate.exclusive(timeout=1.0):
            pass
        return gate.active

    assert asyncio.run(scenario()) == 0


def test_synthesis_gate_makes_new_requests_wait_out_an_unload():
    async def scenario():
        gate = main.SynthesisGate()
        order: list[str] = []

        async def unload():
            async with gate.exclusive(timeout=5.0):
                order.append("unload:start")
                await asyncio.sleep(0.05)
                order.append("unload:end")

        async def synthesize():
            await asyncio.sleep(0.01)
            async with gate.synthesis():
                order.append("synthesis")

        await asyncio.gather(unload(), synthesize())
        return order

    assert asyncio.run(scenario()) == ["unload:start", "unload:end", "synthesis"]


def test_synthesis_gate_unloads_an_idle_runtime_with_a_zero_timeout():
    # asyncio.wait_for(..., timeout=0) cancels the waiter before the predicate is
    # ever evaluated, so ?timeout=0 used to answer 409 on a completely idle server.
    async def scenario():
        gate = main.SynthesisGate()
        async with gate.exclusive(timeout=0.0):
            return "acquired"

    assert asyncio.run(scenario()) == "acquired"


def test_synthesis_gate_is_not_starved_by_a_steady_request_stream():
    # Individually short requests that keep overlapping used to hold _active above
    # zero forever, so unload could never win no matter how long its timeout was.
    async def scenario():
        gate = main.SynthesisGate()
        stop = asyncio.Event()

        async def churn():
            while not stop.is_set():
                async with gate.synthesis():
                    await asyncio.sleep(0.005)

        workers = [asyncio.create_task(churn()) for _ in range(3)]
        await asyncio.sleep(0.02)
        try:
            async with gate.exclusive(timeout=2.0):
                acquired = True
        finally:
            stop.set()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        return acquired

    assert asyncio.run(scenario()) is True


def test_synthesis_gate_rejects_a_second_concurrent_unload():
    async def scenario():
        gate = main.SynthesisGate()
        async with gate.exclusive(timeout=1.0):
            with pytest.raises(main.SynthesisBusyError) as excinfo:
                async with gate.exclusive(timeout=1.0):
                    pass
        return str(excinfo.value)

    assert "Another unload" in asyncio.run(scenario())


def test_cancelled_synthesis_keeps_the_gate_until_the_worker_thread_finishes():
    # Cancelling an `await run_in_executor(...)` does not stop the worker thread.
    # Releasing the gate at that point would let unload delete the model while the
    # thread is still synthesizing with it.
    running = threading.Event()
    finished = threading.Event()

    def blocking_work():
        running.set()
        time.sleep(0.3)
        finished.set()

    async def scenario():
        gate = main.SynthesisGate()

        async def request():
            async with gate.synthesis():
                await main._run_shielded_blocking(blocking_work)

        task = asyncio.create_task(request())
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, running.wait, 5)
        # Cancel the way an anyio cancel scope does: re-deliver on every loop
        # iteration until the task actually ends. A single `cancel()` is a
        # weaker test that a one-shot shield already passes, while Starlette's
        # disconnect handling always produces this repeated form.
        for _ in range(1000):
            if task.done():
                break
            task.cancel()
            await asyncio.sleep(0)
        await asyncio.gather(task, return_exceptions=True)

        # The gate must not have been handed over while the thread was still busy.
        escaped = not finished.is_set()
        async with gate.exclusive(timeout=1.0):
            pass
        return escaped

    assert asyncio.run(scenario()) is False


def test_exclusive_releases_the_reservation_when_its_wait_is_cancelled():
    # `exclusive()` reserves the gate before waiting for in-flight synthesis.
    # If that wait is cancelled (server shutdown, or a Starlette version that
    # cancels handlers on disconnect), nothing has been yielded yet, so the
    # release in the context manager's `finally` never runs. A leaked
    # reservation blocks every later synthesis forever -- they wait on
    # `not _unloading` with no timeout -- and answers every later unload 409.
    async def scenario():
        gate = main.SynthesisGate()

        async def holder():
            async with gate.synthesis():
                await asyncio.sleep(3600)

        held = asyncio.create_task(holder())
        while gate.active == 0:
            await asyncio.sleep(0)

        async def unloader():
            async with gate.exclusive(timeout=3600):
                pass

        waiting = asyncio.create_task(unloader())
        await asyncio.sleep(0.05)  # let it get into the wait
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)

        held.cancel()
        await asyncio.gather(held, return_exceptions=True)

        # A later unload must not be told that one is already in progress. A
        # leaked reservation surfaces here as SynthesisBusyError rather than a
        # hang, because `exclusive()` checks `_unloading` before it waits.
        try:
            async with gate.exclusive(timeout=1.0):
                pass
        except main.SynthesisBusyError as exc:
            return f"busy: {exc}"
        return "ok"

    assert asyncio.run(scenario()) == "ok"


def test_gate_release_survives_repeated_cancellation():
    # The gate's release path must contain no `await` at all. Anything
    # awaitable there -- taking a lock included -- can be interrupted before
    # the counters are restored, and anyio re-fires cancellation on every loop
    # iteration until the task ends, so "it usually gets through" is not a
    # property this can rely on. A leaked `_active` makes every later unload
    # answer 409 forever.
    async def scenario():
        gate = main.SynthesisGate()

        async def victim():
            async with gate.synthesis():
                await asyncio.sleep(0.05)

        task = asyncio.create_task(victim())
        while gate.active == 0:
            await asyncio.sleep(0)

        # Cancel the way an anyio cancel scope does: re-deliver until it ends.
        for _ in range(1000):
            if task.done():
                break
            task.cancel()
            await asyncio.sleep(0)
        await asyncio.gather(task, return_exceptions=True)
        return gate.active

    assert asyncio.run(scenario()) == 0


_SPIN_PROBE = """
import asyncio, time
from irodori_openai_tts import app as main

async def scenario():
    task = asyncio.create_task(main._run_shielded_blocking(time.sleep, 0.5))
    await asyncio.sleep(0.05)
    for other in asyncio.all_tasks():   # runners._cancel_all_tasks equivalent
        if other is not asyncio.current_task():
            other.cancel()
    await asyncio.gather(task, return_exceptions=True)

asyncio.run(scenario())
"""


def test_shielded_blocking_propagates_cancellation_of_its_own_task():
    # `shield` only defends against cancellation aimed at the caller. If the
    # inner task is cancelled it finishes cancelled, and awaiting an
    # already-cancelled future re-raises synchronously without ever yielding --
    # so re-taking the shield in a loop would be a pure CPU spin that freezes
    # the whole event loop, not just this request. Loop shutdown
    # (asyncio.runners._cancel_all_tasks) cancels inner tasks exactly like this.
    #
    # This runs out-of-process on purpose: the failure mode is a frozen loop,
    # so an in-process timeout would never fire and the suite would hang here
    # instead of reporting a failure.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SPIN_PROBE],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("_run_shielded_blocking span the event loop on inner cancellation")
    assert proc.returncode == 0, proc.stderr
