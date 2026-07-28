from __future__ import annotations

import hashlib
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
import torchaudio

from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest


class RefLatentCache:
    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, str] = OrderedDict()
        self._cache_dir = Path(tempfile.gettempdir()) / "irodori_ref_latent_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_or_create(self, runtime: InferenceRuntime, request: SamplingRequest) -> str | None:
        if request.ref_wav is None or request.ref_latent is not None or request.no_ref:
            return request.ref_latent

        ref_path = Path(request.ref_wav).expanduser()
        stat = ref_path.stat()
        key = self._make_key(ref_path, stat.st_mtime_ns, stat.st_size, request)
        cached = self._entries.get(key)
        if cached and Path(cached).is_file():
            self._entries.move_to_end(key)
            return cached

        latent_path = self._cache_dir / f"{key}.pt"
        wav, sr = torchaudio.load(str(ref_path))
        if request.max_ref_seconds is not None and request.max_ref_seconds > 0:
            max_ref_samples = max(1, int(float(request.max_ref_seconds) * float(sr)))
            if wav.shape[1] > max_ref_samples:
                wav = wav[:, :max_ref_samples]

        latent = runtime.codec.encode_waveform(
            wav.unsqueeze(0),
            sample_rate=int(sr),
            normalize_db=request.ref_normalize_db,
            ensure_max=bool(request.ref_ensure_max),
        ).cpu()
        torch.save(latent.squeeze(0), latent_path)

        self._entries[key] = str(latent_path)
        self._entries.move_to_end(key)
        self._evict()
        return str(latent_path)

    def _make_key(
        self,
        ref_path: Path,
        mtime_ns: int,
        size: int,
        request: SamplingRequest,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(str(ref_path.resolve()).encode("utf-8"))
        digest.update(str(mtime_ns).encode("utf-8"))
        digest.update(str(size).encode("utf-8"))
        digest.update(str(request.max_ref_seconds).encode("utf-8"))
        digest.update(str(request.ref_normalize_db).encode("utf-8"))
        digest.update(str(request.ref_ensure_max).encode("utf-8"))
        return digest.hexdigest()

    def _evict(self) -> None:
        while len(self._entries) > self.max_entries:
            _, path = self._entries.popitem(last=False)
            try:
                os.remove(path)
            except OSError:
                pass
