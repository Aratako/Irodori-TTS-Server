from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from typing import Any, Literal

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from irodori_tts.inference_runtime import SamplingRequest, SamplingResult

from .audio import CONTENT_TYPES, encode_audio, normalize_response_format
from .config import get_settings
from .runtime import RuntimeLoadTimeoutError, RuntimeManager
from .voices import VoiceRegistry, VoiceSpec

logger = logging.getLogger(__name__)
CHUNK_BOUNDARIES = frozenset("。、，,．.!！?？\n\r")
_synthesis_semaphore: asyncio.Semaphore | None = None
_synthesis_semaphore_limit: int | None = None


class IrodoriOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    ref_wav: str | None = None
    ref_latent: str | None = None
    no_ref: bool | None = None
    seconds: float | None = None
    duration_scale: float | None = None
    min_seconds: float | None = None
    max_seconds: float | None = None
    max_ref_seconds: float | None = None
    ref_normalize_db: float | None = None
    ref_ensure_max: bool | None = None
    num_steps: int | None = None
    t_schedule_mode: Literal["linear", "sway"] | None = None
    sway_coeff: float | None = None
    num_candidates: int | None = None
    decode_mode: Literal["sequential", "batch"] | None = None
    cfg_scale_text: float | None = None
    cfg_scale_speaker: float | None = None
    cfg_guidance_mode: Literal["independent", "joint", "alternating"] | None = None
    cfg_scale: float | None = None
    cfg_min_t: float | None = None
    cfg_max_t: float | None = None
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    context_kv_cache: bool | None = None
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    seed: int | None = None
    trim_tail: bool | None = None
    tail_window_size: int | None = None
    tail_std_threshold: float | None = None
    tail_mean_threshold: float | None = None
    max_text_len: int | None = None
    lora_adapter: str | None = None
    chunking_enabled: bool | None = None
    chunk_min_chars: int | None = None


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str = Field(min_length=1, max_length=4096)
    voice: str | dict[str, Any] | None = None
    response_format: str | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: str | None = None
    irodori: IrodoriOptions = Field(default_factory=IrodoriOptions)


settings = get_settings()
runtime_manager = RuntimeManager(settings)
voice_registry = VoiceRegistry(settings)


def startup() -> None:
    voices_dir = voice_registry.ensure_dir()
    logger.info("voices directory: %s", voices_dir)
    if settings.preload:
        logger.info("preload enabled; loading runtime during startup")
        runtime_manager.get()
    else:
        logger.info("preload disabled; runtime will load on first speech request")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup()
    yield


app = FastAPI(
    title="Irodori-TTS OpenAI-compatible API",
    version="0.1.0",
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if settings.api_key is None:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return openai_error_response(
        str(exc.detail),
        status_code=int(exc.status_code),
        error_type="invalid_request_error" if exc.status_code < 500 else "server_error",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return openai_error_response(str(exc), status_code=422, error_type="invalid_request_error")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return openai_error_response(str(exc), status_code=500, error_type="server_error")


@app.get("/health")
def health() -> dict[str, Any]:
    voices_dir = settings.voices_dir.expanduser()
    return {
        "status": "ok",
        "model": {
            "id": settings.model_name,
            "hf_checkpoint": settings.hf_checkpoint,
            "model_device": settings.model_device,
            "codec_device": settings.codec_device,
            "model_precision": settings.model_precision,
            "codec_precision": settings.codec_precision,
            "compile_model": settings.compile_model,
            "compile_dynamic": settings.compile_dynamic,
        },
        "runtime": {
            "preload": settings.preload,
            "loaded": runtime_manager.is_loaded,
            "loading": runtime_manager.is_loading,
            "checkpoint": runtime_manager.checkpoint_path,
            "load_timeout": settings.model_load_timeout,
            "max_concurrent_synthesis": settings.max_concurrent_synthesis,
            "synthesis_wait_timeout": settings.synthesis_wait_timeout,
        },
        "voices": {
            "dir": str(voices_dir),
            "dir_exists": voices_dir.is_dir(),
            "files": len(voice_registry.list_files()) if voices_dir.is_dir() else 0,
        },
        "defaults": {
            "response_format": settings.default_response_format,
            "chunking_enabled": settings.default_chunking_enabled,
            "chunk_min_chars": settings.default_chunk_min_chars,
        },
    }


@app.get("/v1/models", dependencies=[Depends(require_auth)])
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "irodori-tts",
            }
        ],
    }


@app.get("/v1/audio/voices", dependencies=[Depends(require_auth)])
def list_voices() -> dict[str, Any]:
    data = []
    for voice in voice_registry.list():
        data.append(
            {
                "id": voice.voice_id,
                "object": "voice",
                "ref_wav": voice.ref_wav,
                "ref_latent": voice.ref_latent,
                "no_ref": voice.no_ref,
            }
        )
    return {"object": "list", "data": data}


@app.post("/v1/audio/voices", status_code=201, dependencies=[Depends(require_auth)])
async def upload_voice(
    file: UploadFile = File(...),
    voice_id: str | None = Form(default=None),
) -> JSONResponse:
    filename = file.filename or ""
    try:
        voice_file = voice_registry.write_file(
            filename=filename,
            data=await file.read(),
            voice_id=voice_id,
            replace=False,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("voice uploaded: %s", voice_file.path)
    return JSONResponse(status_code=201, content=voice_file.metadata())


@app.get("/v1/audio/voices/{voice_id}", dependencies=[Depends(require_auth)])
def get_voice_file(voice_id: str) -> dict[str, Any]:
    try:
        voice_registry.validate_voice_id(voice_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    voice_file = voice_registry.get_file(voice_id)
    if voice_file is None:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id!r} was not found.")
    return voice_file.metadata()


@app.put("/v1/audio/voices/{voice_id}", dependencies=[Depends(require_auth)])
async def replace_voice(
    voice_id: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    try:
        voice_registry.validate_voice_id(voice_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if voice_registry.get_file(voice_id) is None:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id!r} was not found.")

    try:
        voice_file = voice_registry.write_file(
            filename=file.filename or "",
            data=await file.read(),
            voice_id=voice_id,
            replace=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("voice replaced: %s", voice_file.path)
    return voice_file.metadata()


@app.delete("/v1/audio/voices/{voice_id}", dependencies=[Depends(require_auth)])
def delete_voice(voice_id: str) -> dict[str, Any]:
    try:
        voice_registry.validate_voice_id(voice_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    deleted = voice_registry.delete_file(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice {voice_id!r} was not found.")

    logger.info("voice deleted: %s", voice_id)
    return {"id": voice_id, "object": "voice_file", "deleted": True}


@app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
async def create_speech(payload: SpeechRequest) -> Response:
    _validate_speech_payload(payload)
    stream_as_sse = _stream_format_is_sse(payload.stream_format)

    try:
        response_format = normalize_response_format(
            payload.response_format,
            default=settings.default_response_format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request_started_at = time.perf_counter()
    voice = _resolve_voice(payload)
    logger.info(
        "speech synthesis started: model=%s voice=%s format=%s chars=%d",
        payload.model,
        voice.voice_id,
        response_format,
        len(payload.input),
    )
    try:
        sampling_request = _build_sampling_request(payload, voice)
        _validate_sampling_request(sampling_request)
        chunks = _speech_chunks(payload, sampling_request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if stream_as_sse:
        return _stream_speech_response(
            sampling_request,
            chunks,
            response_format,
            request_started_at,
        )

    try:
        runtime = await _run_blocking(runtime_manager.get)
    except RuntimeLoadTimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    synthesis_semaphore = await _acquire_synthesis_slot()
    try:
        result = await _synthesize_chunks(runtime, sampling_request, chunks)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        if "Dynamic LoRA loading is not compatible" in str(exc):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise
    finally:
        _release_synthesis_slot(synthesis_semaphore)
    audio_bytes = await _run_blocking(
        encode_audio,
        result.audio,
        result.sample_rate,
        response_format,
    )
    logger.info(
        "speech synthesis completed: elapsed=%.2fs audio_seconds=%.2f bytes=%d seed=%s",
        time.perf_counter() - request_started_at,
        _audio_duration_seconds(result.audio, result.sample_rate),
        len(audio_bytes),
        result.used_seed,
    )

    headers = {
        "Content-Disposition": f'attachment; filename="speech.{response_format}"',
        "X-Irodori-Seed": str(result.used_seed),
        "X-Irodori-Total-To-Decode": f"{result.total_to_decode:.6f}",
    }
    if result.messages:
        headers["X-Irodori-Messages"] = " | ".join(result.messages)[0:4096]

    return Response(
        content=audio_bytes,
        media_type=CONTENT_TYPES[response_format],
        headers=headers,
    )


def _stream_format_is_sse(stream_format: str | None) -> bool:
    if stream_format is None:
        return False
    value = str(stream_format).strip().lower()
    if value == "sse":
        return True
    raise HTTPException(
        status_code=400,
        detail="stream_format must be 'sse' when specified.",
    )


def _validate_speech_payload(payload: SpeechRequest) -> None:
    if payload.model != settings.model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model {payload.model!r}. Use {settings.model_name!r}.",
        )
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="input must contain non-whitespace text.")


def _resolve_voice(payload: SpeechRequest) -> VoiceSpec:
    options = payload.irodori
    explicit_ref_wav = options.ref_wav
    explicit_ref_latent = options.ref_latent
    explicit_no_ref = options.no_ref
    if explicit_ref_wav is None:
        explicit_ref_wav = _extra(payload, "ref_wav")
    if explicit_ref_latent is None:
        explicit_ref_latent = _extra(payload, "ref_latent")
    if explicit_no_ref is None:
        explicit_no_ref = _extra(payload, "no_ref")

    if explicit_ref_wav is not None or explicit_ref_latent is not None or explicit_no_ref:
        return VoiceSpec(
            voice_id="request",
            ref_wav=None if explicit_ref_wav is None else str(explicit_ref_wav),
            ref_latent=None if explicit_ref_latent is None else str(explicit_ref_latent),
            no_ref=bool(explicit_no_ref),
        )
    try:
        return voice_registry.resolve(payload.voice)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _audio_duration_seconds(audio: Any, sample_rate: int) -> float:
    shape = getattr(audio, "shape", None)
    if shape is None or len(shape) == 0 or sample_rate <= 0:
        return 0.0
    return float(shape[-1]) / float(sample_rate)


async def _run_blocking(func: Any, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


def _get_synthesis_semaphore() -> asyncio.Semaphore:
    global _synthesis_semaphore, _synthesis_semaphore_limit

    limit = max(1, int(settings.max_concurrent_synthesis))
    if _synthesis_semaphore is None or _synthesis_semaphore_limit != limit:
        _synthesis_semaphore = asyncio.Semaphore(limit)
        _synthesis_semaphore_limit = limit
    return _synthesis_semaphore


async def _acquire_synthesis_slot() -> asyncio.Semaphore:
    timeout = max(0.0, float(settings.synthesis_wait_timeout))
    try:
        semaphore = _get_synthesis_semaphore()
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        return semaphore
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Synthesis queue is full. Retry after a moment. timeout={timeout:.1f}s",
        ) from exc


def _release_synthesis_slot(semaphore: asyncio.Semaphore) -> None:
    semaphore.release()


def _speech_chunks(payload: SpeechRequest, sampling_request: SamplingRequest) -> list[str]:
    enabled = bool(
        _coalesce(
            payload.irodori.chunking_enabled,
            _extra(payload, "chunking_enabled"),
            _extra(payload, "chunking"),
            settings.default_chunking_enabled,
        )
    )
    if not enabled:
        return [payload.input]

    if sampling_request.seconds is not None:
        logger.info("speech chunking skipped: explicit seconds is set")
        return [payload.input]

    min_chars = _as_int(
        _coalesce(
            payload.irodori.chunk_min_chars,
            _extra(payload, "chunk_min_chars"),
            settings.default_chunk_min_chars,
        ),
        "chunk_min_chars",
    )
    if min_chars <= 0:
        raise HTTPException(status_code=400, detail="chunk_min_chars must be greater than 0.")

    chunks = _split_text_for_speech(payload.input, min_chars=min_chars)
    if len(chunks) > 1:
        logger.info("speech chunking enabled: chunks=%d min_chars=%d", len(chunks), min_chars)
    return chunks


def _split_text_for_speech(text: str, *, min_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for char in text:
        current.append(char)
        if not char.isspace():
            current_chars += 1
        if char in CHUNK_BOUNDARIES and current_chars >= min_chars:
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []
            current_chars = 0

    tail = "".join(current).strip()
    if tail:
        chunks.append(tail)
    return chunks or [text]


async def _synthesize_chunks(
    runtime: Any,
    sampling_request: SamplingRequest,
    chunks: list[str],
) -> SamplingResult:
    if len(chunks) == 1:
        return await _run_blocking(
            runtime.synthesize,
            sampling_request,
            log_fn=_log_runtime_message,
        )

    results: list[SamplingResult] = []
    for index, chunk in enumerate(chunks, start=1):
        logger.info("speech chunk %d/%d started: chars=%d", index, len(chunks), len(chunk))
        chunk_request = replace(sampling_request, text=chunk)
        chunk_result = await _run_blocking(
            runtime.synthesize,
            chunk_request,
            log_fn=_log_runtime_message,
        )
        logger.info(
            "speech chunk %d/%d completed: audio_seconds=%.2f",
            index,
            len(chunks),
            _audio_duration_seconds(chunk_result.audio, chunk_result.sample_rate),
        )
        results.append(chunk_result)

    sample_rate = results[0].sample_rate
    if any(result.sample_rate != sample_rate for result in results):
        raise RuntimeError("Chunk sample rates did not match.")

    audio = torch.cat([_audio_as_channels_first(result.audio) for result in results], dim=-1)
    messages = [message for result in results for message in result.messages]
    stage_timings = [
        (f"chunk_{index}:{name}", seconds)
        for index, result in enumerate(results, start=1)
        for name, seconds in result.stage_timings
    ]
    return SamplingResult(
        audio=audio,
        audios=[audio],
        sample_rate=sample_rate,
        stage_timings=stage_timings,
        total_to_decode=sum(result.total_to_decode for result in results),
        used_seed=results[0].used_seed,
        messages=messages,
    )


def _stream_speech_response(
    sampling_request: SamplingRequest,
    chunks: list[str],
    response_format: str,
    request_started_at: float,
) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        completed = 0
        try:
            runtime = await _run_blocking(runtime_manager.get)
            for index, chunk in enumerate(chunks):
                logger.info(
                    "speech stream chunk %d/%d started: chars=%d",
                    index + 1,
                    len(chunks),
                    len(chunk),
                )
                chunk_request = replace(sampling_request, text=chunk)
                synthesis_semaphore = await _acquire_synthesis_slot()
                try:
                    result = await _run_stream_blocking(
                        runtime.synthesize,
                        chunk_request,
                        log_fn=_log_runtime_message,
                    )
                finally:
                    _release_synthesis_slot(synthesis_semaphore)
                audio_bytes = await _run_stream_blocking(
                    encode_audio,
                    result.audio,
                    result.sample_rate,
                    response_format,
                )
                completed += 1
                logger.info(
                    "speech stream chunk %d/%d completed: audio_seconds=%.2f bytes=%d",
                    index + 1,
                    len(chunks),
                    _audio_duration_seconds(result.audio, result.sample_rate),
                    len(audio_bytes),
                )
                yield _sse_event(
                    "audio_chunk",
                    {
                        "index": index,
                        "text": chunk,
                        "format": response_format,
                        "media_type": CONTENT_TYPES[response_format],
                        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                        "seed": result.used_seed,
                        "total_to_decode": result.total_to_decode,
                    },
                )
        except RuntimeLoadTimeoutError as exc:
            yield _sse_error_event(str(exc), error_type="server_error", code="runtime_unavailable")
            return
        except HTTPException as exc:
            yield _sse_openai_error_event(exc)
            return
        except (FileNotFoundError, ValueError) as exc:
            yield _sse_error_event(str(exc), code="invalid_request")
            return
        except RuntimeError as exc:
            error_type = (
                "invalid_request_error"
                if "Dynamic LoRA loading is not compatible" in str(exc)
                else "server_error"
            )
            yield _sse_error_event(str(exc), error_type=error_type, code="stream_error")
            return
        else:
            logger.info(
                "speech stream completed: elapsed=%.2fs chunks=%d",
                time.perf_counter() - request_started_at,
                completed,
            )
            yield _sse_event("done", {"chunks": completed})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run_stream_blocking(func: Any, *args: Any, **kwargs: Any) -> Any:
    task = asyncio.create_task(_run_blocking(func, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def _sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def _sse_error_event(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: str,
    param: str | None = None,
) -> str:
    return _sse_event(
        "error",
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def _sse_openai_error_event(exc: HTTPException) -> str:
    error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
    code = "synthesis_queue_timeout" if exc.status_code == 503 else "stream_error"
    return _sse_error_event(str(exc.detail), error_type=error_type, code=code)


def _log_runtime_message(message: str) -> None:
    logger.info("irodori runtime: %s", message)


def _audio_as_channels_first(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        return audio.unsqueeze(0)
    if audio.ndim == 2:
        return audio
    raise ValueError(
        f"Expected audio shape (samples,) or (channels, samples), got {tuple(audio.shape)}"
    )


def _build_sampling_request(payload: SpeechRequest, voice: VoiceSpec) -> SamplingRequest:
    opts = payload.irodori

    duration_scale = _as_float(
        _coalesce(
            opts.duration_scale,
            _extra(payload, "duration_scale"),
            settings.default_duration_scale,
        ),
        "duration_scale",
    )
    if payload.speed != 1.0:
        duration_scale = duration_scale / float(payload.speed)

    seconds = _coalesce(opts.seconds, _extra(payload, "seconds"), None)
    if seconds is not None and payload.speed != 1.0:
        seconds = _as_float(seconds, "seconds") / float(payload.speed)

    return SamplingRequest(
        text=payload.input,
        ref_wav=voice.ref_wav,
        ref_latent=voice.ref_latent,
        no_ref=bool(voice.no_ref),
        ref_normalize_db=_as_optional_float(
            _explicit_option(
                payload,
                "ref_normalize_db",
                settings.default_ref_normalize_db,
            ),
            "ref_normalize_db",
        ),
        ref_ensure_max=bool(
            _coalesce(
                opts.ref_ensure_max,
                _extra(payload, "ref_ensure_max"),
                settings.default_ref_ensure_max,
            )
        ),
        num_candidates=_as_int(
            _coalesce(
                opts.num_candidates,
                _extra(payload, "num_candidates"),
                settings.default_num_candidates,
            ),
            "num_candidates",
        ),
        decode_mode=str(
            _coalesce(
                opts.decode_mode, _extra(payload, "decode_mode"), settings.default_decode_mode
            )
        ),
        seconds=_as_optional_float(seconds, "seconds"),
        duration_scale=duration_scale,
        min_seconds=_as_float(
            _coalesce(
                opts.min_seconds, _extra(payload, "min_seconds"), settings.default_min_seconds
            ),
            "min_seconds",
        ),
        max_seconds=_as_float(
            _coalesce(
                opts.max_seconds, _extra(payload, "max_seconds"), settings.default_max_seconds
            ),
            "max_seconds",
        ),
        max_ref_seconds=_as_optional_float(
            _explicit_option(
                payload,
                "max_ref_seconds",
                settings.default_max_ref_seconds,
            ),
            "max_ref_seconds",
        ),
        max_text_len=_as_optional_int(
            _coalesce(opts.max_text_len, _extra(payload, "max_text_len"), None),
            "max_text_len",
        ),
        num_steps=_as_int(
            _coalesce(opts.num_steps, _extra(payload, "num_steps"), settings.default_num_steps),
            "num_steps",
        ),
        cfg_scale_text=_as_float(
            _coalesce(
                opts.cfg_scale_text,
                _extra(payload, "cfg_scale_text"),
                settings.default_cfg_scale_text,
            ),
            "cfg_scale_text",
        ),
        cfg_scale_speaker=_as_float(
            _coalesce(
                opts.cfg_scale_speaker,
                _extra(payload, "cfg_scale_speaker"),
                settings.default_cfg_scale_speaker,
            ),
            "cfg_scale_speaker",
        ),
        cfg_guidance_mode=str(
            _coalesce(
                opts.cfg_guidance_mode,
                _extra(payload, "cfg_guidance_mode"),
                settings.default_cfg_guidance_mode,
            )
        ),
        cfg_scale=_as_optional_float(
            _coalesce(opts.cfg_scale, _extra(payload, "cfg_scale"), None),
            "cfg_scale",
        ),
        cfg_min_t=_as_float(
            _coalesce(opts.cfg_min_t, _extra(payload, "cfg_min_t"), settings.default_cfg_min_t),
            "cfg_min_t",
        ),
        cfg_max_t=_as_float(
            _coalesce(opts.cfg_max_t, _extra(payload, "cfg_max_t"), settings.default_cfg_max_t),
            "cfg_max_t",
        ),
        truncation_factor=_as_optional_float(
            _coalesce(opts.truncation_factor, _extra(payload, "truncation_factor"), None),
            "truncation_factor",
        ),
        rescale_k=_as_optional_float(
            _coalesce(opts.rescale_k, _extra(payload, "rescale_k"), None),
            "rescale_k",
        ),
        rescale_sigma=_as_optional_float(
            _coalesce(opts.rescale_sigma, _extra(payload, "rescale_sigma"), None),
            "rescale_sigma",
        ),
        context_kv_cache=bool(
            _coalesce(
                opts.context_kv_cache,
                _extra(payload, "context_kv_cache"),
                settings.default_context_kv_cache,
            )
        ),
        speaker_kv_scale=_as_optional_float(
            _coalesce(opts.speaker_kv_scale, _extra(payload, "speaker_kv_scale"), None),
            "speaker_kv_scale",
        ),
        speaker_kv_min_t=_as_optional_float(
            _coalesce(opts.speaker_kv_min_t, _extra(payload, "speaker_kv_min_t"), None),
            "speaker_kv_min_t",
        ),
        speaker_kv_max_layers=_as_optional_int(
            _coalesce(
                opts.speaker_kv_max_layers,
                _extra(payload, "speaker_kv_max_layers"),
                None,
            ),
            "speaker_kv_max_layers",
        ),
        seed=_as_optional_int(_coalesce(opts.seed, _extra(payload, "seed"), None), "seed"),
        t_schedule_mode=str(
            _coalesce(
                opts.t_schedule_mode,
                _extra(payload, "t_schedule_mode"),
                settings.default_t_schedule_mode,
            )
        ),
        sway_coeff=_as_float(
            _coalesce(opts.sway_coeff, _extra(payload, "sway_coeff"), settings.default_sway_coeff),
            "sway_coeff",
        ),
        trim_tail=bool(
            _coalesce(opts.trim_tail, _extra(payload, "trim_tail"), settings.default_trim_tail)
        ),
        tail_window_size=_as_int(
            _coalesce(
                opts.tail_window_size,
                _extra(payload, "tail_window_size"),
                settings.default_tail_window_size,
            ),
            "tail_window_size",
        ),
        tail_std_threshold=_as_float(
            _coalesce(
                opts.tail_std_threshold,
                _extra(payload, "tail_std_threshold"),
                settings.default_tail_std_threshold,
            ),
            "tail_std_threshold",
        ),
        tail_mean_threshold=_as_float(
            _coalesce(
                opts.tail_mean_threshold,
                _extra(payload, "tail_mean_threshold"),
                settings.default_tail_mean_threshold,
            ),
            "tail_mean_threshold",
        ),
        lora_adapter=_as_optional_str(
            _coalesce(opts.lora_adapter, _extra(payload, "lora_adapter"), None),
            "lora_adapter",
        ),
    )


def _validate_sampling_request(request: SamplingRequest) -> None:
    if request.duration_scale <= 0:
        raise HTTPException(status_code=400, detail="duration_scale must be greater than 0.")
    if request.max_seconds < request.min_seconds:
        raise HTTPException(
            status_code=400, detail="max_seconds must be greater than or equal to min_seconds."
        )


def _extra(payload: SpeechRequest, key: str) -> Any:
    extra = payload.model_extra or {}
    if not isinstance(extra, Mapping):
        return None
    return extra.get(key)


def _explicit_option(payload: SpeechRequest, key: str, default: Any) -> Any:
    if key in payload.irodori.model_fields_set:
        return getattr(payload.irodori, key)
    extra = payload.model_extra or {}
    if isinstance(extra, Mapping) and key in extra:
        return extra[key]
    return default


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _as_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, name)


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _as_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, name)


def _as_optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{name} must be non-empty when specified.")
    return text


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def openai_error_response(message: str, *, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )
