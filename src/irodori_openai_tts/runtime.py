from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    default_runtime_device,
)

from .config import Settings

logger = logging.getLogger(__name__)


class RuntimeLoadTimeoutError(RuntimeError):
    pass


class RuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._runtime: InferenceRuntime | None = None
        self._checkpoint_path: str | None = None

    def get(self) -> InferenceRuntime:
        if self._runtime is not None:
            return self._runtime

        timeout = float(self.settings.model_load_timeout)
        acquired = self._lock.acquire(timeout=max(timeout, 0.0))
        if not acquired:
            raise RuntimeLoadTimeoutError(
                f"Model is still loading. Retry after a moment. timeout={timeout:.1f}s"
            )

        try:
            if self._runtime is None:
                logger.info("loading runtime")
                t0 = time.perf_counter()
                self._checkpoint_path = self._resolve_checkpoint_path()
                logger.info("checkpoint resolved: %s", self._checkpoint_path)
                self._runtime = InferenceRuntime.from_key(
                    RuntimeKey(
                        checkpoint=self._checkpoint_path,
                        model_device=self._resolve_device(self.settings.model_device),
                        codec_repo=str(self.settings.codec_repo),
                        model_precision=str(self.settings.model_precision),
                        codec_device=self._resolve_device(self.settings.codec_device),
                        codec_precision=str(self.settings.codec_precision),
                        codec_deterministic_encode=bool(self.settings.codec_deterministic_encode),
                        codec_deterministic_decode=bool(self.settings.codec_deterministic_decode),
                        compile_model=bool(self.settings.compile_model),
                        compile_dynamic=bool(self.settings.compile_dynamic),
                    )
                )
                elapsed = time.perf_counter() - t0
                logger.info("runtime loaded in %.2fs", elapsed)
            return self._runtime
        finally:
            self._lock.release()

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def is_loading(self) -> bool:
        return self._runtime is None and self._lock.locked()

    @property
    def use_caption_condition(self) -> bool | None:
        if self._runtime is None:
            return None
        return bool(self._runtime.model_cfg.use_caption_condition)

    def _resolve_checkpoint_path(self) -> str:
        if self.settings.checkpoint is not None and str(self.settings.checkpoint).strip() != "":
            path = Path(str(self.settings.checkpoint)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return str(path)

        repo_id = str(self.settings.hf_checkpoint).strip()
        if repo_id == "":
            raise ValueError("Set IRODORI_CHECKPOINT or IRODORI_HF_CHECKPOINT.")
        logger.info("downloading checkpoint from hf://%s/model.safetensors", repo_id)
        t0 = time.perf_counter()
        path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        elapsed = time.perf_counter() - t0
        logger.info("checkpoint download/cache lookup completed in %.2fs", elapsed)
        return path

    @staticmethod
    def _resolve_device(value: str) -> str:
        raw = str(value).strip().lower()
        if raw in {"", "auto"}:
            return default_runtime_device()
        return str(value)
