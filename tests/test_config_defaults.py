from __future__ import annotations

from irodori_openai_tts.config import Settings


def test_perf_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.default_decode_mode == "batch"
    assert settings.enable_cpu_fallback_on_oom is True
    assert settings.ref_latent_cache_size >= 1
