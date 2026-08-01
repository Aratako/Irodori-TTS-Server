from __future__ import annotations

import json

import pytest

from irodori_openai_tts.config import Settings
from irodori_openai_tts.voices import LATENT_EXTENSIONS, VOICE_EXTENSIONS, VoiceRegistry


def make_registry(tmp_path, **overrides) -> VoiceRegistry:
    settings = Settings(
        voices_dir=tmp_path,
        allow_no_ref_voice=True,
        _env_file=None,
        **overrides,
    )
    return VoiceRegistry(settings)


def test_resolve_scanned_audio_file(tmp_path):
    (tmp_path / "speaker.wav").write_bytes(b"wav")
    registry = make_registry(tmp_path)

    voice = registry.resolve("speaker")

    assert voice.voice_id == "speaker"
    assert voice.ref_wav == str(tmp_path / "speaker.wav")
    assert voice.ref_latent is None


@pytest.mark.parametrize("extension", sorted(VOICE_EXTENSIONS))
def test_resolve_all_supported_audio_extensions(tmp_path, extension):
    path = tmp_path / f"speaker{extension}"
    path.write_bytes(b"audio")
    registry = make_registry(tmp_path)

    voice = registry.resolve("speaker")

    assert voice.ref_wav == str(path)


def test_resolve_scanned_latent_file(tmp_path):
    (tmp_path / "speaker.pt").write_bytes(b"latent")
    registry = make_registry(tmp_path)

    voice = registry.resolve({"id": "speaker"})

    assert voice.voice_id == "speaker"
    assert voice.ref_latent == str(tmp_path / "speaker.pt")


@pytest.mark.parametrize("extension", sorted(LATENT_EXTENSIONS))
def test_resolve_all_supported_latent_extensions(tmp_path, extension):
    path = tmp_path / f"speaker{extension}"
    path.write_bytes(b"latent")
    registry = make_registry(tmp_path)

    voice = registry.resolve("speaker")

    assert voice.ref_latent == str(path)


def test_resolve_alias_file_with_ref_wav(tmp_path):
    (tmp_path / "narrator.wav").write_bytes(b"wav")
    (tmp_path / "voices.json").write_text(
        json.dumps(
            {
                "calm": {
                    "ref_wav": "narrator.wav",
                }
            }
        ),
        encoding="utf-8",
    )
    registry = make_registry(tmp_path)

    voice = registry.resolve("calm")

    assert voice.ref_wav == str(tmp_path / "narrator.wav")


def test_resolve_alias_file_with_multiple_references(tmp_path):
    (tmp_path / "clip-1.wav").write_bytes(b"wav")
    (tmp_path / "clip-2.wav").write_bytes(b"wav")
    (tmp_path / "clip-1.pt").write_bytes(b"latent")
    (tmp_path / "clip-2.pt").write_bytes(b"latent")
    (tmp_path / "voices.json").write_text(
        json.dumps(
            {
                "audio": {"ref_wavs": ["clip-1.wav", "clip-2.wav"]},
                "latent": {"ref_latents": ["clip-1.pt", "clip-2.pt"]},
            }
        ),
        encoding="utf-8",
    )
    registry = make_registry(tmp_path)

    audio = registry.resolve("audio")
    latent = registry.resolve("latent")

    assert audio.ref_wavs == (
        str(tmp_path / "clip-1.wav"),
        str(tmp_path / "clip-2.wav"),
    )
    assert latent.ref_latents == (
        str(tmp_path / "clip-1.pt"),
        str(tmp_path / "clip-2.pt"),
    )


@pytest.mark.parametrize("value", [[], "clip.wav", [""]])
def test_resolve_alias_rejects_invalid_multiple_references(tmp_path, value):
    (tmp_path / "voices.json").write_text(
        json.dumps({"invalid": {"ref_wavs": value}}),
        encoding="utf-8",
    )
    registry = make_registry(tmp_path)

    with pytest.raises(ValueError, match="ref_wavs"):
        registry.resolve("invalid")


def test_resolve_default_voice(tmp_path):
    (tmp_path / "default.wav").write_bytes(b"wav")
    registry = make_registry(tmp_path, default_voice="default")

    voice = registry.resolve(None)

    assert voice.voice_id == "default"


def test_resolve_no_ref_voice(tmp_path):
    registry = make_registry(tmp_path)

    voice = registry.resolve("none")

    assert voice.no_ref is True


def test_resolve_unknown_voice_raises_key_error(tmp_path):
    registry = make_registry(tmp_path)

    with pytest.raises(KeyError, match="Unknown voice"):
        registry.resolve("missing")


def test_write_file_create_replace_and_delete(tmp_path):
    registry = make_registry(tmp_path)

    created = registry.write_file(filename="speaker.wav", data=b"old")
    assert created.path.read_bytes() == b"old"
    assert registry.get_file("speaker") is not None

    with pytest.raises(FileExistsError):
        registry.write_file(filename="speaker.wav", data=b"new")

    replaced = registry.write_file(
        filename="speaker.flac", data=b"new", voice_id="speaker", replace=True
    )
    assert replaced.path.name == "speaker.flac"
    assert not (tmp_path / "speaker.wav").exists()
    assert replaced.path.read_bytes() == b"new"

    assert registry.delete_file("speaker") is True
    assert registry.delete_file("speaker") is False


@pytest.mark.parametrize("extension", sorted(VOICE_EXTENSIONS))
def test_write_file_accepts_all_supported_audio_extensions(tmp_path, extension):
    registry = make_registry(tmp_path)

    created = registry.write_file(filename=f"speaker{extension}", data=b"audio")

    assert created.path == tmp_path / f"speaker{extension}"
    assert created.path.read_bytes() == b"audio"


def test_write_file_rejects_bad_voice_id_extension_and_empty_data(tmp_path):
    registry = make_registry(tmp_path)

    with pytest.raises(ValueError, match="voice_id"):
        registry.write_file(filename="speaker.wav", data=b"wav", voice_id="../bad")
    with pytest.raises(ValueError, match="Unsupported voice file extension"):
        registry.write_file(filename="speaker.txt", data=b"text")
    with pytest.raises(ValueError, match="must not be empty"):
        registry.write_file(filename="speaker.wav", data=b"")
