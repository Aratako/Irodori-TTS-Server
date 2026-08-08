from __future__ import annotations

import threading

import pytest

from irodori_openai_tts import runtime as runtime_module
from irodori_openai_tts.config import Settings
from irodori_openai_tts.runtime import RuntimeLoadTimeoutError, RuntimeManager


def test_sampling_steps_default_to_model_default(monkeypatch):
    monkeypatch.delenv("IRODORI_DEFAULT_NUM_STEPS", raising=False)

    assert Settings(_env_file=None).default_num_steps is None


@pytest.mark.parametrize("steps", [4, 40])
def test_sampling_steps_environment_override(monkeypatch, steps):
    monkeypatch.setenv("IRODORI_DEFAULT_NUM_STEPS", str(steps))

    assert Settings(_env_file=None).default_num_steps == steps


def test_runtime_load_timeout_while_another_thread_is_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        model_load_timeout=0.05,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    started = threading.Event()
    release = threading.Event()
    loaded_runtime = object()
    errors: list[BaseException] = []

    def fake_from_key(_key):
        started.set()
        release.wait(timeout=2)
        return loaded_runtime

    monkeypatch.setattr(
        runtime_module.InferenceRuntime,
        "from_key",
        staticmethod(fake_from_key),
    )

    def load_runtime():
        try:
            assert manager.get() is loaded_runtime
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=load_runtime)
    thread.start()
    assert started.wait(timeout=1)

    with pytest.raises(RuntimeLoadTimeoutError):
        manager.get()

    release.set()
    thread.join(timeout=2)
    assert errors == []
    assert manager.is_loaded
    assert not manager.is_loading


def test_runtime_resolves_local_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    manager = RuntimeManager(Settings(checkpoint=str(checkpoint), _env_file=None))

    assert manager._resolve_checkpoint_path() == str(checkpoint)


def test_runtime_rejects_missing_local_checkpoint(tmp_path):
    manager = RuntimeManager(
        Settings(checkpoint=str(tmp_path / "missing.safetensors"), _env_file=None)
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        manager._resolve_checkpoint_path()


def test_runtime_downloads_hf_checkpoint_when_local_checkpoint_is_unset(monkeypatch):
    manager = RuntimeManager(Settings(hf_checkpoint="owner/repo", _env_file=None))

    def fake_download(repo_id):
        assert repo_id == "owner/repo"
        return "/cache/model.safetensors"

    monkeypatch.setattr(runtime_module, "download_hf_checkpoint", fake_download)

    assert manager._resolve_checkpoint_path() == "/cache/model.safetensors"


class FakeUnloadableRuntime:
    def __init__(self) -> None:
        self.unloaded = 0

    def unload(self) -> None:
        self.unloaded += 1


def _manager_with_fake_runtimes(tmp_path, monkeypatch, runtimes, **overrides):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        _env_file=None,
        **overrides,
    )
    pending = iter(runtimes)
    monkeypatch.setattr(
        runtime_module.InferenceRuntime,
        "from_key",
        staticmethod(lambda _key: next(pending)),
    )
    return RuntimeManager(settings)


def test_runtime_unload_releases_runtime_and_next_get_reloads(tmp_path, monkeypatch):
    first = FakeUnloadableRuntime()
    second = FakeUnloadableRuntime()
    manager = _manager_with_fake_runtimes(tmp_path, monkeypatch, [first, second])

    assert manager.get() is first
    assert manager.unload() is True
    assert first.unloaded == 1
    assert not manager.is_loaded

    assert manager.get() is second
    assert manager.is_loaded


def test_runtime_does_not_report_loading_while_it_is_unloading(tmp_path, monkeypatch):
    """A teardown must not look like a load to /health.

    unload() clears _runtime while still holding the load lock, which is exactly
    the state is_loading was written to describe. Without a separate flag, every
    poll taken during the slowest part of a teardown (freeing accelerator memory)
    reports loading=true, so a monitor reads an unload as a model load.
    """
    seen: list[tuple[bool, bool]] = []

    class ObservingRuntime:
        def unload(self) -> None:
            seen.append((manager.is_loading, manager.is_loaded))

    manager = _manager_with_fake_runtimes(tmp_path, monkeypatch, [ObservingRuntime()])
    manager.get()
    assert manager.unload() is True

    # Observed from inside the teardown: neither loading nor loaded.
    assert seen == [(False, False)]
    assert not manager.is_loading


def test_runtime_unload_is_idempotent_when_nothing_is_loaded(tmp_path, monkeypatch):
    manager = _manager_with_fake_runtimes(tmp_path, monkeypatch, [])

    assert manager.unload() is False
    assert manager.unload() is False
    assert not manager.is_loaded


def test_runtime_unload_drops_reference_even_when_release_raises(tmp_path, monkeypatch):
    class ExplodingRuntime:
        def unload(self) -> None:
            raise RuntimeError("release failed")

    replacement = FakeUnloadableRuntime()
    manager = _manager_with_fake_runtimes(tmp_path, monkeypatch, [ExplodingRuntime(), replacement])
    manager.get()

    with pytest.raises(RuntimeError, match="release failed"):
        manager.unload()

    # A half-released runtime must never stay reachable from get().
    assert not manager.is_loaded
    assert manager.get() is replacement


def test_runtime_unload_times_out_while_another_thread_is_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        model_load_timeout=0.05,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    started = threading.Event()
    release = threading.Event()

    def fake_from_key(_key):
        started.set()
        release.wait(timeout=2)
        return FakeUnloadableRuntime()

    monkeypatch.setattr(
        runtime_module.InferenceRuntime,
        "from_key",
        staticmethod(fake_from_key),
    )

    thread = threading.Thread(target=manager.get)
    thread.start()
    assert started.wait(timeout=1)

    with pytest.raises(RuntimeLoadTimeoutError):
        manager.unload()

    release.set()
    thread.join(timeout=2)
    assert manager.is_loaded
