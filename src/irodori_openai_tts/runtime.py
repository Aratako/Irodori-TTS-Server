from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
)

from .config import Settings
from .cuda_perf import configure_cuda_performance

logger = logging.getLogger(__name__)


class RuntimeLoadTimeoutError(RuntimeError):
    pass


class RuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._fallback_lock = threading.Lock()
        self._runtime: InferenceRuntime | None = None
        self._cpu_fallback_runtime: InferenceRuntime | None = None
        self._uncompiled_runtime: InferenceRuntime | None = None
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
                configure_cuda_performance()
                runtime_key = RuntimeKey(
                    checkpoint=self._checkpoint_path,
                    model_device=self._resolve_device(self.settings.model_device),
                    codec_repo=str(self.settings.codec_repo),
                    model_precision=str(self.settings.model_precision),
                    codec_device=self._resolve_device(self.settings.codec_device),
                    codec_precision=str(self.settings.codec_precision),
                    codec_deterministic_encode=bool(self.settings.codec_deterministic_encode),
                    codec_deterministic_decode=bool(self.settings.codec_deterministic_decode),
                    compile_model=bool(self.settings.compile_model)
                    and _can_use_torch_compile(self._resolve_device(self.settings.model_device)),
                    compile_dynamic=bool(self.settings.compile_dynamic),
                )
                if bool(self.settings.compile_model) and not runtime_key.compile_model:
                    logger.warning(
                        "torch.compile was requested but disabled automatically (unsupported runtime/toolchain)."
                    )
                try:
                    self._runtime = InferenceRuntime.from_key(runtime_key)
                except RuntimeError as exc:
                    if runtime_key.compile_model and _is_triton_missing_error(exc):
                        logger.warning(
                            "torch.compile disabled automatically: Triton is unavailable (%s)",
                            exc,
                        )
                        self._runtime = InferenceRuntime.from_key(
                            RuntimeKey(
                                checkpoint=runtime_key.checkpoint,
                                model_device=runtime_key.model_device,
                                codec_repo=runtime_key.codec_repo,
                                model_precision=runtime_key.model_precision,
                                codec_device=runtime_key.codec_device,
                                codec_precision=runtime_key.codec_precision,
                                codec_deterministic_encode=runtime_key.codec_deterministic_encode,
                                codec_deterministic_decode=runtime_key.codec_deterministic_decode,
                                compile_model=False,
                                compile_dynamic=False,
                            )
                        )
                    else:
                        raise
                warmup_exc = self._warmup_runtime(self._runtime)
                if runtime_key.compile_model and isinstance(warmup_exc, RuntimeError):
                    if _is_triton_missing_error(warmup_exc):
                        logger.warning(
                            "reloading runtime without torch.compile due to Triton runtime error"
                        )
                        self._runtime = InferenceRuntime.from_key(
                            RuntimeKey(
                                checkpoint=runtime_key.checkpoint,
                                model_device=runtime_key.model_device,
                                codec_repo=runtime_key.codec_repo,
                                model_precision=runtime_key.model_precision,
                                codec_device=runtime_key.codec_device,
                                codec_precision=runtime_key.codec_precision,
                                codec_deterministic_encode=runtime_key.codec_deterministic_encode,
                                codec_deterministic_decode=runtime_key.codec_deterministic_decode,
                                compile_model=False,
                                compile_dynamic=False,
                            )
                        )
                        self._warmup_runtime(self._runtime)
                elapsed = time.perf_counter() - t0
                logger.info("runtime loaded in %.2fs", elapsed)
            return self._runtime
        finally:
            self._lock.release()

    def get_cpu_fallback(self) -> InferenceRuntime:
        if self._cpu_fallback_runtime is not None:
            return self._cpu_fallback_runtime

        timeout = float(self.settings.model_load_timeout)
        acquired = self._fallback_lock.acquire(timeout=max(timeout, 0.0))
        if not acquired:
            raise RuntimeLoadTimeoutError(
                f"CPU fallback runtime is still loading. Retry after a moment. timeout={timeout:.1f}s"
            )
        try:
            if self._cpu_fallback_runtime is None:
                if self._checkpoint_path is None:
                    self._checkpoint_path = self._resolve_checkpoint_path()
                logger.warning("loading CPU fallback runtime for OOM recovery")
                t0 = time.perf_counter()
                self._cpu_fallback_runtime = InferenceRuntime.from_key(
                    RuntimeKey(
                        checkpoint=self._checkpoint_path,
                        model_device="cpu",
                        codec_repo=str(self.settings.codec_repo),
                        model_precision="fp32",
                        codec_device="cpu",
                        codec_precision="fp32",
                        codec_deterministic_encode=bool(self.settings.codec_deterministic_encode),
                        codec_deterministic_decode=bool(self.settings.codec_deterministic_decode),
                        compile_model=False,
                        compile_dynamic=False,
                    )
                )
                elapsed = time.perf_counter() - t0
                logger.warning("CPU fallback runtime loaded in %.2fs", elapsed)
            return self._cpu_fallback_runtime
        finally:
            self._fallback_lock.release()

    def get_uncompiled_runtime(self) -> InferenceRuntime:
        if self._uncompiled_runtime is not None:
            return self._uncompiled_runtime

        timeout = float(self.settings.model_load_timeout)
        acquired = self._fallback_lock.acquire(timeout=max(timeout, 0.0))
        if not acquired:
            raise RuntimeLoadTimeoutError(
                f"Uncompiled runtime is still loading. Retry after a moment. timeout={timeout:.1f}s"
            )
        try:
            if self._uncompiled_runtime is None:
                if self._checkpoint_path is None:
                    self._checkpoint_path = self._resolve_checkpoint_path()
                logger.warning("loading uncompiled runtime fallback")
                self._uncompiled_runtime = InferenceRuntime.from_key(
                    RuntimeKey(
                        checkpoint=self._checkpoint_path,
                        model_device=self._resolve_device(self.settings.model_device),
                        codec_repo=str(self.settings.codec_repo),
                        model_precision=str(self.settings.model_precision),
                        codec_device=self._resolve_device(self.settings.codec_device),
                        codec_precision=str(self.settings.codec_precision),
                        codec_deterministic_encode=bool(self.settings.codec_deterministic_encode),
                        codec_deterministic_decode=bool(self.settings.codec_deterministic_decode),
                        compile_model=False,
                        compile_dynamic=False,
                    )
                )
            return self._uncompiled_runtime
        finally:
            self._fallback_lock.release()

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def is_loading(self) -> bool:
        return self._runtime is None and self._lock.locked()

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

    def _warmup_runtime(self, runtime: InferenceRuntime) -> Exception | None:
        if not bool(self.settings.cuda_warmup_enabled):
            return None
        try:
            runtime.synthesize(
                SamplingRequest(
                    text=str(self.settings.cuda_warmup_text),
                    no_ref=True,
                    num_steps=4,
                    seed=0,
                    decode_mode="batch",
                )
            )
            logger.info("runtime warmup completed")
            return None
        except Exception as exc:
            logger.warning("runtime warmup skipped: %s", exc)
            return exc


def _is_triton_missing_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "triton" in message and ("missing" in message or "cannot find a working triton" in message)


def _can_use_torch_compile(model_device: str) -> bool:
    if str(model_device).startswith("cuda"):
        try:
            import triton  # type: ignore  # noqa: F401
        except Exception:
            return False
    return hasattr(torch, "compile")
