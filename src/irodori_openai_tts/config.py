from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IRODORI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8088
    api_key: str | None = None

    checkpoint: str | None = None
    hf_checkpoint: str = "Aratako/Irodori-TTS-500M-v3"
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    model_name: str = "irodori-tts"

    model_device: str = "auto"
    codec_device: str = "auto"
    model_precision: str = "fp32"
    codec_precision: str = "fp32"
    codec_deterministic_encode: bool = True
    codec_deterministic_decode: bool = True
    compile_model: bool = False
    compile_dynamic: bool = False
    preload: bool = False
    model_load_timeout: float = 300.0
    max_concurrent_synthesis: int = 1
    synthesis_wait_timeout: float = 300.0

    voices_dir: Path = Path("voices")
    voice_aliases_file: Path | None = None
    default_voice: str | None = None
    allow_no_ref_voice: bool = True

    default_response_format: str = "wav"
    default_num_steps: int = 40
    default_t_schedule_mode: str = "linear"
    default_sway_coeff: float = -1.0
    default_duration_scale: float = 1.0
    default_min_seconds: float = 0.5
    default_max_seconds: float = 30.0
    default_text_scale: float = 3.0
    default_caption_scale: float = 3.0
    default_speaker_scale: float = 5.0
    default_cfg_guidance_mode: str = "independent"
    default_max_caption_len: int | None = None
    default_cfg_min_t: float = 0.5
    default_cfg_max_t: float = 1.0
    default_context_kv_cache: bool = True
    default_max_ref_seconds: float | None = 30.0
    default_ref_normalize_db: float | None = -16.0
    default_ref_ensure_max: bool = True
    default_trim_tail: bool = True
    default_tail_window_size: int = 20
    default_tail_std_threshold: float = 0.05
    default_tail_mean_threshold: float = 0.1
    default_num_candidates: int = 1
    default_decode_mode: str = "sequential"
    default_chunking_enabled: bool = True
    default_chunk_min_chars: int = 80
    default_first_sentence_chunk_min_chars: int | None = None

    cors_origins: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
