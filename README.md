# Irodori OpenAI TTS Server

OpenAI Text-to-Speech API compatible server for [Irodori-TTS](https://github.com/Aratako/Irodori-TTS).

This server targets the [Irodori-TTS 500M v3 base model](https://huggingface.co/Aratako/Irodori-TTS-500M-v3). It supports reference-audio voice cloning, OpenAI-style response formats, and automatic long text chunking.

Streaming synthesis is not implemented. Requests return one complete audio response.

## Features

- OpenAI-compatible `POST /v1/audio/speech`
- Reference voices from files, `voices.json`, or HTTP upload
- Response formats: `wav`, `mp3`, `flac`, `opus`, `aac`, `pcm`
- Automatic long text chunking
- Per-request dynamic LoRA adapter loading
- Optional bearer token auth

## Requirements

For local Python:

- Python 3.10
- uv
- FFmpeg for compressed audio formats

For Docker:

- Docker Engine with Docker Compose, or Docker Desktop
- NVIDIA Container Toolkit or Docker Desktop GPU support for CUDA inference
- ROCm-capable Docker host for AMD GPU inference

A CUDA or ROCm GPU is recommended for practical inference.

## Installation

```bash
git clone https://github.com/Aratako/Irodori-TTS-Server.git
cd Irodori-TTS-Server
uv sync --extra cu128
cp .env.example .env
```

Choose one PyTorch backend extra:

```bash
uv sync --extra cu128  # NVIDIA CUDA 12.8
uv sync --extra rocm   # AMD ROCm on Linux
uv sync --extra cpu    # CPU-only
```

The PyTorch backend extras are mutually exclusive. The `cu128` extra uses the PyTorch CUDA 12.8 index, the `rocm` extra uses the PyTorch ROCm index on Linux, and the `cpu` extra uses the CPU PyTorch index on Linux/Windows.

By default, the server downloads [`Aratako/Irodori-TTS-500M-v3`](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) from Hugging Face when the model is first loaded. To use a local checkpoint, set:

```bash
IRODORI_CHECKPOINT=/path/to/model.safetensors
```

## Running

```bash
uv run python -m irodori_openai_tts --host 0.0.0.0 --port 8088
```

For ROCm:

```bash
uv run --extra rocm python -m irodori_openai_tts --host 0.0.0.0 --port 8088
```

Open the health endpoint:

```bash
curl http://localhost:8088/health
```

## Docker

Create `.env` first:

```bash
cp .env.example .env
```

Set the backend used when the image is built:

```env
IRODORI_TTS_BACKEND=cu128
```

Supported values are `cu128`, `rocm`, and `cpu`.

On the first run, or after updating the server code, build and recreate the container:

```bash
docker compose up --build --force-recreate
```

After that, start the existing image normally:

```bash
docker compose up
```

For NVIDIA GPU settings, build and recreate with both Compose files:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build --force-recreate
```

Then use this for normal GPU startup:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up
```

For AMD ROCm, set `IRODORI_TTS_BACKEND=rocm` in `.env`, then build and recreate with the ROCm Compose file:

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up --build --force-recreate
```

Then use this for normal ROCm startup:

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up
```

For CPU-only Docker images, set `IRODORI_TTS_BACKEND=cpu` in `.env` before building.

Reference voices placed in `./voices` are available inside the container. Downloaded Hugging Face files are kept in a Docker volume so they are reused across container recreations.

## Quick Usage

Put a reference voice in `voices/`. Files can be added before or after the server starts; the directory is scanned when a request resolves a voice.

```text
voices/
  sample.wav
```

Then call the speech endpoint:

```bash
curl http://localhost:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "irodori-tts",
    "input": "こんにちは。これはIrodori-TTSのAPIテストです。",
    "voice": "sample",
    "response_format": "wav"
  }' \
  --output speech.wav
```

Using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8088/v1",
    api_key="not-used",
)

with client.audio.speech.with_streaming_response.create(
    model="irodori-tts",
    voice="sample",
    input="こんにちは。これはIrodori-TTSのAPIテストです。",
    response_format="wav",
) as response:
    response.stream_to_file("speech.wav")
```

The SDK method name contains `streaming_response`, but this server still generates a complete response internally.

## API

### `GET /health`

Returns server status and current configuration. This endpoint does not load the model.

### `GET /v1/models`

Returns the model ID accepted by the speech endpoint.

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "irodori-tts",
      "object": "model",
      "created": 0,
      "owned_by": "irodori-tts"
    }
  ]
}
```

### `POST /v1/audio/speech`

Synthesizes speech and returns audio bytes.

Request fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `model` | string | yes | Use `irodori-tts` unless you changed `IRODORI_MODEL_NAME`. |
| `input` | string | yes | Text to synthesize. |
| `voice` | string or object | no | Voice ID, or `{ "id": "voice_id" }`. Uses `IRODORI_DEFAULT_VOICE` if omitted. |
| `response_format` | string | no | `wav`, `mp3`, `flac`, `opus`, `aac`, or `pcm`. |
| `speed` | number | no | Speaking speed, from `0.25` to `4.0`. Higher is faster; internally this is converted to an inverse duration scale. |
| `irodori` | object | no | Irodori-specific inference options. |

`stream_format: "sse"` is rejected because streaming synthesis is not supported.

Irodori-specific options:

```json
{
  "model": "irodori-tts",
  "input": "こんにちは。",
  "voice": "sample",
  "response_format": "wav",
  "speed": 1.1,
  "irodori": {
    "num_steps": 24,
    "cfg_scale_text": 3.0,
    "cfg_scale_speaker": 5.0,
    "lora_adapter": "/models/adapters/speaker-a",
    "seed": 1234,
    "t_schedule_mode": "sway",
    "sway_coeff": -1.0
  }
}
```

Common `irodori` options:

| Field | Notes |
| --- | --- |
| `num_steps` | Number of diffusion steps. Higher can improve quality but takes longer. |
| `seed` | Fixed random seed for reproducible output. |
| `cfg_scale_text` | Strength of text guidance. |
| `cfg_scale_speaker` | Strength of speaker/reference-voice guidance. |
| `lora_adapter` | PEFT LoRA adapter directory to load dynamically for this request. The adapter is not merged into the base checkpoint. |
| `t_schedule_mode` | Sampling schedule, usually `linear` or `sway`. |
| `sway_coeff` | Sway schedule coefficient when using `t_schedule_mode: "sway"`. |
| `chunking_enabled` | Enable or disable automatic long text chunking for this request. |
| `chunk_min_chars` | Minimum non-space characters before a chunk split point is used. |

Dynamic LoRA loading is per runtime process. The first request for an adapter loads it into memory; later requests for the same adapter reuse the cached adapter. To run the base model after an adapter has been loaded, omit `lora_adapter` or set it to `null`, `"none"`, or `"base"`. Dynamic LoRA is not compatible with `IRODORI_COMPILE_MODEL=true`.

### Voice Management

The server scans `IRODORI_VOICES_DIR` for voice files. File stems become voice IDs.

Supported audio extensions:

- `.wav`
- `.flac`
- `.mp3`
- `.m4a`
- `.ogg`
- `.opus`
- `.aac`
- `.webm`

Latent references are also supported:

- `.pt`
- `.pth`

Examples:

```text
voices/
  alice.wav      -> voice: "alice"
  bob.flac       -> voice: "bob"
  cached.pt      -> voice: "cached"
```

You can also create `voices/voices.json`:

```json
{
  "alice": "alice.wav",
  "bob": "bob_reference.flac",
  "cached": "cached.pt"
}
```

Text-only inference is available with `voice: "none"` when `IRODORI_ALLOW_NO_REF_VOICE=true`.

Voice file endpoints:

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/v1/audio/voices` | List resolved voices. |
| `POST` | `/v1/audio/voices` | Upload voice file with multipart `file` and optional `voice_id`. |
| `GET` | `/v1/audio/voices/{voice_id}` | Get uploaded voice file metadata. |
| `PUT` | `/v1/audio/voices/{voice_id}` | Replace uploaded voice file. |
| `DELETE` | `/v1/audio/voices/{voice_id}` | Delete uploaded voice file. |

Upload example:

```bash
curl http://localhost:8088/v1/audio/voices \
  -F voice_id=sample \
  -F file=@sample.wav
```

## Long Text Chunking

Long text chunking is enabled by default.

When enabled, the server splits text only when both conditions are met:

- the current chunk has at least `chunk_min_chars` non-space characters
- the current character is punctuation or a line break

Each chunk is synthesized sequentially, then concatenated into one audio response.

Per-request override:

```json
{
  "model": "irodori-tts",
  "input": "長い本文...",
  "voice": "sample",
  "response_format": "wav",
  "irodori": {
    "chunking_enabled": true,
    "chunk_min_chars": 80
  }
}
```

If `irodori.seconds` is set, chunking is skipped because that fixed duration applies to the whole request.

## Request Queue

Only one synthesis request runs at a time by default. Additional requests wait for an available slot.

You can tune the queue with:

```env
IRODORI_MAX_CONCURRENT_SYNTHESIS=1
IRODORI_SYNTHESIS_WAIT_TIMEOUT=300
```

If the model is still loading or no synthesis slot becomes available before the configured timeout, the server returns HTTP 503.

## Configuration

Server defaults are configured with environment variables. For local runs and Docker Compose, copy `.env.example` to `.env` and edit it as needed.

All environment variables use the `IRODORI_` prefix. Request fields override these defaults when the corresponding option is provided in the API request.

| Variable | Default | Notes |
| --- | --- | --- |
| `IRODORI_HOST` | `0.0.0.0` | Server host. |
| `IRODORI_PORT` | `8088` | Server port. |
| `IRODORI_TTS_BACKEND` | `cu128` | Docker build backend: `cu128`, `rocm`, or `cpu`. |
| `IRODORI_API_KEY` | unset | Optional bearer token. |
| `IRODORI_MODEL_NAME` | `irodori-tts` | Model ID used in requests. |
| `IRODORI_HF_CHECKPOINT` | `Aratako/Irodori-TTS-500M-v3` | Hugging Face repo containing `model.safetensors`. |
| `IRODORI_CHECKPOINT` | unset | Local checkpoint path. Takes precedence over `IRODORI_HF_CHECKPOINT`. |
| `IRODORI_CODEC_REPO` | `Aratako/Semantic-DACVAE-Japanese-32dim` | DACVAE codec repo or path. |
| `IRODORI_MODEL_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. |
| `IRODORI_CODEC_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. |
| `IRODORI_MODEL_PRECISION` | `fp32` | `fp32` or `bf16`. |
| `IRODORI_CODEC_PRECISION` | `fp32` | `fp32` or `bf16`. |
| `IRODORI_COMPILE_MODEL` | `false` | Enable `torch.compile` for core inference methods. Keep disabled when using dynamic LoRA adapters. |
| `IRODORI_COMPILE_DYNAMIC` | `false` | Use `dynamic=True` for `torch.compile`. |
| `IRODORI_PRELOAD` | `false` | Load the model during startup. |
| `IRODORI_MODEL_LOAD_TIMEOUT` | `300` | Seconds to wait for model loading. |
| `IRODORI_MAX_CONCURRENT_SYNTHESIS` | `1` | Maximum simultaneous synthesis jobs. |
| `IRODORI_SYNTHESIS_WAIT_TIMEOUT` | `300` | Seconds to wait for a synthesis slot. |
| `IRODORI_VOICES_DIR` | `voices` | Directory scanned for reference voices. |
| `IRODORI_DEFAULT_VOICE` | unset | Used when request omits `voice`. |
| `IRODORI_ALLOW_NO_REF_VOICE` | `true` | Allow `voice: "none"` text-only inference. |
| `IRODORI_DEFAULT_RESPONSE_FORMAT` | `wav` | Default response format. |
| `IRODORI_DEFAULT_NUM_STEPS` | `40` | Default diffusion steps. |
| `IRODORI_DEFAULT_T_SCHEDULE_MODE` | `linear` | Default timestep schedule. |
| `IRODORI_DEFAULT_SWAY_COEFF` | `-1.0` | Default sway coefficient. Used only when `t_schedule_mode` is `sway`. |
| `IRODORI_DEFAULT_DURATION_SCALE` | `1.0` | Default duration scale. |
| `IRODORI_DEFAULT_CFG_SCALE_TEXT` | `3.0` | Default text CFG scale. |
| `IRODORI_DEFAULT_CFG_SCALE_SPEAKER` | `5.0` | Default speaker CFG scale. |
| `IRODORI_DEFAULT_CFG_GUIDANCE_MODE` | `independent` | Default CFG guidance mode. |
| `IRODORI_DEFAULT_CHUNKING_ENABLED` | `true` | Enable punctuation-aware chunking by default. |
| `IRODORI_DEFAULT_CHUNK_MIN_CHARS` | `80` | Minimum non-space characters before a split point is used. |

## Development

Run tests:

```bash
uv run --extra dev pytest
```

Run lint:

```bash
uv run --extra dev ruff check src tests
```

Run import/bytecode checks:

```bash
uv run python -m compileall src tests
```

## License

This server code is released under the MIT License. See [LICENSE](LICENSE).

Model weights and codec assets are distributed separately. Check the Hugging Face model cards for their licenses and usage terms:

- [Aratako/Irodori-TTS-500M-v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
- [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
