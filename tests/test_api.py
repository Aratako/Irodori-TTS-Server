from __future__ import annotations

import asyncio
import base64
import json
import threading

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


def test_models_lists_configured_single_v3_model():
    response = TestClient(main.app).get("/v1/models")

    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == main.settings.model_name


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


@pytest.mark.parametrize("field", ["duration_scale", "chunk_min_chars", "ref_normalize_db"])
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
