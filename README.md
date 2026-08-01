# Irodori OpenAI TTS Server

OpenAI Text-to-Speech API compatible server for [Irodori-TTS](https://github.com/Aratako/Irodori-TTS).

This server targets [Irodori-TTS-v4-Small](https://huggingface.co/Aratako/Irodori-TTS-v4-Small). It supports voice cloning, Voice Design, OpenAI-style response formats, and automatic long text chunking.

Standard requests return one complete audio response. Chunk-level Server-Sent Events are also available for long text.

## Features

- OpenAI-compatible `POST /v1/audio/speech`
- Reference voices from files, multiple waveform or latent clips, `voices.json`, or HTTP upload
- Caption-based Voice Design and Speaker Inversion
- Up to 120 seconds of combined reference audio with v4 Small
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

After syncing with a backend extra, use `uv run --no-sync ...` for the commands
below to avoid re-syncing the environment without the selected PyTorch backend
extra.

By default, the server downloads [`Aratako/Irodori-TTS-v4-Small`](https://huggingface.co/Aratako/Irodori-TTS-v4-Small) and its bundled tokenizer from Hugging Face when the model is first loaded. To use a local checkpoint, set:

```bash
IRODORI_CHECKPOINT=/path/to/model.safetensors
```

For v4 checkpoints, keep the exported `tokenizer/` directory next to
`model.safetensors`. Older checkpoints without bundled tokenizer assets continue
to use the tokenizer repository recorded in their checkpoint metadata.

## Running

```bash
uv run --no-sync python -m irodori_openai_tts --host 0.0.0.0 --port 8088
```

This uses the PyTorch backend selected during `uv sync`.

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

### Voice Design

Irodori-TTS-v4-Small supports both pure Voice Design without reference audio and
caption-controlled voice cloning.

For pure Voice Design, use the built-in `none` voice and describe the desired
voice and delivery with `irodori.caption`:

```bash
curl http://localhost:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "irodori-tts",
    "input": "本日はお越しいただき、ありがとうございます。",
    "voice": "none",
    "response_format": "wav",
    "irodori": {
      "caption": "落ち着いた低めの女性の声。丁寧で穏やかな話し方。"
    }
  }' \
  --output voice_design.wav
```

For caption-controlled voice cloning, specify both a registered reference voice
and a caption. The reference supplies the speaker identity while the caption
guides voice characteristics and delivery:

```bash
curl http://localhost:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "irodori-tts",
    "input": "それでは、元気よく始めましょう！",
    "voice": "alice",
    "response_format": "wav",
    "irodori": {
      "caption": "明るく元気で、楽しそうな話し方。"
    }
  }' \
  --output styled_clone.wav
```

The `none` voice is available when `IRODORI_ALLOW_NO_REF_VOICE=true`, which is
enabled by default.

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
| `stream_format` | string | no | Set to `sse` to receive chunk-level Server-Sent Events. |
| `irodori` | object | no | Irodori-specific inference options. |

When `stream_format: "sse"` is set, the response is `text/event-stream`.
The server synthesizes each text chunk sequentially and emits one `audio_chunk`
event per chunk, followed by a final `done` event:

For consistent voice tone across chunks, specify a reference voice with `voice`,
`irodori.ref_wav`, or `irodori.ref_wavs`. Without a reference, each chunk is synthesized
independently and the perceived voice tone may vary between chunks.

```bash
curl -N http://localhost:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "irodori-tts",
    "input": "最初の文です。次の文です。",
    "voice": "sample",
    "response_format": "wav",
    "stream_format": "sse",
    "irodori": {
      "chunking_enabled": true,
      "chunk_min_chars": 1
    }
  }'
```

```text
event: audio_chunk
data: {"index":0,"text":"最初の文です。","format":"wav","media_type":"audio/wav","audio_base64":"...","seed":123,"total_to_decode":0.1}

event: audio_chunk
data: {"index":1,"text":"次の文です。","format":"wav","media_type":"audio/wav","audio_base64":"...","seed":123,"total_to_decode":0.1}

event: done
data: {"chunks":2}
```

Each `audio_base64` value contains a complete audio file for that chunk, so
clients can decode and enqueue chunks while later chunks are still generating.

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
| `first_sentence_chunk_min_chars` | Optional minimum non-space characters used only for splitting the first sentence. |
| `ref_wav` / `ref_latent` | One reference waveform or precomputed latent path. |
| `ref_wavs` / `ref_latents` | Ordered arrays of reference paths. The runtime concatenates them and applies the reference-duration limit. Do not combine singular and plural reference fields. |
| `max_ref_seconds` | Override the reference-duration limit. When omitted, v4 Small uses its checkpoint value of 120 seconds and legacy checkpoints fall back to 30 seconds. |
| `caption` | Voice/style description for caption-enabled VoiceDesign checkpoints. Ignored by checkpoints without caption conditioning. |
| `cfg_scale_caption` | Strength of caption guidance. |
| `max_caption_len` | Optional maximum caption token length. |

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

Latent references and Speaker Inversion are also supported:

- `.pt`
- `.pth`
- `.speaker.safetensors`

Examples:

```text
voices/
  alice.wav      -> voice: "alice"
  bob.flac       -> voice: "bob"
  cached.pt      -> voice: "cached"
```

Each file discovered this way becomes a separate voice. The server does not infer
that similarly named files belong to the same speaker.

Reference voices can be supplied in four ways:

| Method | Single reference | Multiple references | Lifetime |
| --- | --- | --- | --- |
| Place a file in `voices/` | yes | no | Persistent; the filename stem becomes the voice ID. |
| Voice upload API | yes | no | Persistent; each upload creates or replaces one voice file. |
| `voices/voices.json` alias | yes | yes | Persistent; use `ref_wavs` or `ref_latents` to group multiple files under one voice ID. |
| Request-level `irodori.ref_wav` / `irodori.ref_wavs` | yes | yes | One request only; no voice is registered. |

Paths passed directly in a request are resolved on the server, not on the client
machine. They must be local paths visible to the server process; HTTP URLs are not
accepted. With Docker, use paths visible inside the container. A remote client
that cannot provide a server-side path can upload a single reference and use the
resulting voice ID. Reusable multi-file groups still require a server-side
`voices.json` definition.

To register multiple clips as one persistent voice, create `voices/voices.json`:

```json
{
  "alice": "alice.wav",
  "bob": "bob_reference.flac",
  "cached": "cached.pt",
  "alice_long": {
    "ref_wavs": ["alice_01.wav", "alice_02.wav", "alice_03.wav"]
  }
}
```

Paths in `voices.json` are resolved relative to `IRODORI_VOICES_DIR`. Array entries
are processed in the order shown. The example above registers the three clips as
one voice named `alice_long`, which can then be used like any other voice:

```json
{
  "model": "irodori-tts",
  "input": "登録済みの複数参照音声を使用します。",
  "voice": "alice_long",
  "response_format": "wav"
}
```

The upload API accepts one file at a time and does not append clips to an existing
voice group. To create a reusable multi-clip voice, place or upload the individual
files and define their grouping in `voices.json`. For a one-off request that does
not need registration, pass the paths directly with `irodori.ref_wavs` or
`irodori.ref_latents` as described below.

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

## Long Reference Audio

Irodori-TTS-v4-Small accepts up to 120 seconds of combined reference audio. Pass
multiple clips in input order with `irodori.ref_wavs`. These paths refer to files
visible to the server process, or to the container when running with Docker:

```json
{
  "model": "irodori-tts",
  "input": "複数の参照音声を使った音声合成です。",
  "irodori": {
    "ref_wavs": [
      "voices/speaker_01.wav",
      "voices/speaker_02.wav",
      "voices/speaker_03.wav"
    ]
  }
}
```

The clips are encoded separately, concatenated in the supplied order, and cut at
the checkpoint-specific reference limit. v4 Small was trained using concatenated
short clips, so multiple representative clips from the same speaker are the
recommended way to use the extended context. A single long recording is accepted,
but its behavior is less established because it does not match the primary
training construction.

`ref_latents` provides the equivalent ordered input for precomputed latent files.
Do not mix waveform and latent references, or singular and plural forms in the
same request. Set `irodori.max_ref_seconds` only when an explicit override is
needed; omitting it uses checkpoint metadata and preserves the 30-second fallback
for older checkpoints.

## Long Text Chunking

Long text chunking is enabled by default.

When enabled, the server splits text only when both conditions are met:

- the current chunk has at least `chunk_min_chars` non-space characters
- the current character is punctuation or a line break

Set `irodori.first_sentence_chunk_min_chars` to use a smaller threshold only
for the first sentence. Later sentences keep the normal `chunk_min_chars`
threshold.

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
    "chunk_min_chars": 80,
    "first_sentence_chunk_min_chars": 1
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
| `IRODORI_HF_CHECKPOINT` | `Aratako/Irodori-TTS-v4-Small` | Hugging Face repo containing `model.safetensors` and optional bundled tokenizer assets. |
| `IRODORI_CHECKPOINT` | unset | Local checkpoint path. Takes precedence over `IRODORI_HF_CHECKPOINT`; keep a bundled `tokenizer/` beside v4 checkpoints. |
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
| `IRODORI_DEFAULT_MAX_REF_SECONDS` | unset | Reference-duration override. Unset uses checkpoint metadata; v4 Small uses 120 seconds and legacy checkpoints fall back to 30 seconds. |
| `IRODORI_DEFAULT_CHUNKING_ENABLED` | `true` | Enable punctuation-aware chunking by default. |
| `IRODORI_DEFAULT_CHUNK_MIN_CHARS` | `80` | Minimum non-space characters before a split point is used. |
| `IRODORI_DEFAULT_FIRST_SENTENCE_CHUNK_MIN_CHARS` | unset | Minimum non-space characters before the first sentence split point is used. Unset keeps normal `chunk_min_chars` behavior. |

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

- [Aratako/Irodori-TTS-v4-Small](https://huggingface.co/Aratako/Irodori-TTS-v4-Small)
- [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
