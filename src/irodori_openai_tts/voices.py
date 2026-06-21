from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings

VOICE_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".webm",
}
LATENT_EXTENSIONS = {".pt", ".pth"}
SPEAKER_INVERSION_SUFFIX = ".speaker.safetensors"
NO_REF_IDS = {"none", "no_ref", "no-ref", "null", "text-only"}
VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class VoiceSpec:
    voice_id: str
    ref_wav: str | None = None
    ref_latent: str | None = None
    ref_embed: str | None = None
    no_ref: bool = False


@dataclass(frozen=True)
class VoiceFile:
    voice_id: str
    path: Path

    def metadata(self) -> dict[str, Any]:
        stat = self.path.stat()
        return {
            "id": self.voice_id,
            "object": "voice_file",
            "filename": self.path.name,
            "bytes": stat.st_size,
            "created_at": int(stat.st_mtime),
        }


class VoiceRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ensure_dir(self) -> Path:
        root = self.settings.voices_dir.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def list(self) -> list[VoiceSpec]:
        voices = self._scan_voice_files()
        aliases = self._load_aliases()
        for voice_id, spec in aliases.items():
            voices[voice_id] = spec
        if self.settings.allow_no_ref_voice:
            voices.setdefault("none", VoiceSpec(voice_id="none", no_ref=True))
        return [voices[key] for key in sorted(voices)]

    def resolve(self, voice: str | dict[str, Any] | None) -> VoiceSpec:
        requested = self._voice_id_from_request(voice)
        if requested is None or requested == "":
            requested = self.settings.default_voice
        if requested is None or requested == "":
            raise KeyError("No voice was provided and IRODORI_DEFAULT_VOICE is not set.")

        voice_id = str(requested).strip()
        if voice_id.lower() in NO_REF_IDS and self.settings.allow_no_ref_voice:
            return VoiceSpec(voice_id=voice_id, no_ref=True)

        aliases = self._load_aliases()
        if voice_id in aliases:
            return aliases[voice_id]

        scanned = self._scan_voice_files()
        if voice_id in scanned:
            return scanned[voice_id]

        raise KeyError(
            f"Unknown voice={voice_id!r}. Put a reference audio file in "
            f"{self.settings.voices_dir}, add an alias file, or use voice='none'."
        )

    def list_files(self) -> list[VoiceFile]:
        root = self.ensure_dir()
        items = []
        for path in sorted(root.iterdir(), key=lambda item: (item.stat().st_mtime, item.name)):
            if path.is_file() and path.suffix.lower() in VOICE_EXTENSIONS:
                items.append(VoiceFile(voice_id=path.stem, path=path))
        return items

    def get_file(self, voice_id: str) -> VoiceFile | None:
        root = self.ensure_dir()
        for path in sorted(root.iterdir()):
            if (
                path.is_file()
                and path.stem == voice_id
                and path.suffix.lower() in VOICE_EXTENSIONS
            ):
                return VoiceFile(voice_id=voice_id, path=path)
        return None

    def write_file(
        self,
        *,
        filename: str,
        data: bytes,
        voice_id: str | None = None,
        replace: bool = False,
    ) -> VoiceFile:
        suffix = Path(filename).suffix.lower()
        if suffix not in VOICE_EXTENSIONS:
            allowed = ", ".join(sorted(VOICE_EXTENSIONS))
            raise ValueError(f"Unsupported voice file extension {suffix!r}. Use one of: {allowed}.")

        resolved_voice_id = (voice_id or Path(filename).stem).strip()
        self.validate_voice_id(resolved_voice_id)
        if not data:
            raise ValueError("Voice file must not be empty.")

        existing = self.get_file(resolved_voice_id)
        if existing is not None and not replace:
            raise FileExistsError(
                f"Voice {resolved_voice_id!r} already exists. Use PUT to replace it."
            )

        if existing is not None and existing.path.suffix.lower() != suffix:
            existing.path.unlink()

        root = self.ensure_dir()
        path = root / f"{resolved_voice_id}{suffix}"
        path.write_bytes(data)
        return VoiceFile(voice_id=resolved_voice_id, path=path)

    def delete_file(self, voice_id: str) -> bool:
        existing = self.get_file(voice_id)
        if existing is None:
            return False
        existing.path.unlink()
        return True

    @staticmethod
    def validate_voice_id(voice_id: str) -> None:
        if not voice_id or VOICE_ID_PATTERN.fullmatch(voice_id) is None:
            raise ValueError("voice_id must contain only ASCII letters, numbers, underscores, or hyphens.")

    @staticmethod
    def _voice_id_from_request(voice: str | dict[str, Any] | None) -> str | None:
        if voice is None:
            return None
        if isinstance(voice, str):
            return voice
        if isinstance(voice, dict):
            raw = voice.get("id")
            return None if raw is None else str(raw)
        return str(voice)

    def _scan_voice_files(self) -> dict[str, VoiceSpec]:
        root = self.settings.voices_dir.expanduser()
        if not root.exists():
            return {}
        out: dict[str, VoiceSpec] = {}
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            name_lower = path.name.lower()
            if name_lower.endswith(SPEAKER_INVERSION_SUFFIX):
                voice_id = path.name[: -len(SPEAKER_INVERSION_SUFFIX)]
                out[voice_id] = VoiceSpec(voice_id=voice_id, ref_embed=str(path))
            else:
                suffix = path.suffix.lower()
                if suffix in VOICE_EXTENSIONS:
                    out[path.stem] = VoiceSpec(voice_id=path.stem, ref_wav=str(path))
                elif suffix in LATENT_EXTENSIONS:
                    out[path.stem] = VoiceSpec(voice_id=path.stem, ref_latent=str(path))
        return out

    def _load_aliases(self) -> dict[str, VoiceSpec]:
        path = self.settings.voice_aliases_file
        if path is None:
            default = self.settings.voices_dir.expanduser() / "voices.json"
            path = default if default.is_file() else None
        if path is None:
            return {}
        alias_path = path.expanduser()
        if not alias_path.is_file():
            return {}
        with alias_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Voice aliases file must contain an object: {alias_path}")

        out: dict[str, VoiceSpec] = {}
        for voice_id, raw_spec in payload.items():
            out[str(voice_id)] = self._parse_alias(str(voice_id), raw_spec)
        return out

    def _parse_alias(self, voice_id: str, raw_spec: Any) -> VoiceSpec:
        if isinstance(raw_spec, str):
            path = self._resolve_voice_path(raw_spec)
            if path.name.lower().endswith(SPEAKER_INVERSION_SUFFIX):
                return VoiceSpec(voice_id=voice_id, ref_embed=str(path))
            if path.suffix.lower() in LATENT_EXTENSIONS:
                return VoiceSpec(voice_id=voice_id, ref_latent=str(path))
            return VoiceSpec(voice_id=voice_id, ref_wav=str(path))

        if not isinstance(raw_spec, dict):
            raise ValueError(f"Invalid alias for voice={voice_id!r}.")

        no_ref = bool(raw_spec.get("no_ref", False))
        ref_wav = raw_spec.get("ref_wav")
        ref_latent = raw_spec.get("ref_latent")
        ref_embed = raw_spec.get("ref_embed")
        return VoiceSpec(
            voice_id=voice_id,
            ref_wav=None if ref_wav is None else str(self._resolve_voice_path(str(ref_wav))),
            ref_latent=None
            if ref_latent is None
            else str(self._resolve_voice_path(str(ref_latent))),
            ref_embed=None
            if ref_embed is None
            else str(self._resolve_voice_path(str(ref_embed))),
            no_ref=no_ref,
        )

    def _resolve_voice_path(self, value: str) -> Path:
        raw = Path(value).expanduser()
        if raw.is_absolute():
            return raw
        return self.settings.voices_dir.expanduser() / raw
