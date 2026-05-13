# Faster Voxtral 4B TTS

OpenAI-compatible text-to-speech server powered by [Voxtral 4B](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).

The central design goals are:

- **VRAM efficiency without audible quality loss** — NF4 quantization halves VRAM use compared to full precision while producing output that is perceptually identical in blind listening tests, giving the best balance of memory footprint and realtime factor.
- **Low time-to-first-audio** — autoregressive streaming yields the first audio chunk within ~200 ms, making the server suitable for real-time voice assistants and interactive applications.
- **Fast startup** — a persistent quantized-weight cache and parallel model loading cut cold-start time significantly, so the first request can be served within seconds of launch.

The server targets a **single concurrent speech-generation client**; there is no request queue or multi-GPU scaling.

## Features

- **9 languages** - English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi
- **20 voice presets** - male and female voices in almost every of the mentioned languages, with 3 English styles (casual, cheerful, neutral)
- **Streaming** - chunked audio delivery with low time-to-first-audio (<200 ms on an NVIDIA RTX 3090)
- **3 output formats** - MP3, WAV, PCM
- **3 quantization modes** - NF4 (~5 GB VRAM), INT8 (~6 GB), full BF16 (~9 GB)
- **OpenAI API compatible** - drop-in replacement for `/v1/audio/speech`

## Requirements

- `git`
- NVIDIA GPU with ≥ 5 GB VRAM and CUDA 12.6 driver (see [Other CUDA versions](#other-cuda-versions))
- Python 3.12 recommended (3.11–3.13 supported)

| Quantization | Approx. VRAM |
|--------------|-------------|
| `nf4`        | ~5 GB       |
| `int8`       | ~6 GB       |
| `none` (BF16)| ~9 GB       |

## Installation

```bash
git clone <REPO_URL>
cd <repo-dir>
./install.sh
```

That's it. The script installs [uv](https://docs.astral.sh/uv/) if needed, creates `.venv/`, installs PyTorch from the CUDA 12.6 index, builds `flash-attn`, and installs the package.

> **Note:** The first install takes a long time (up to an hour or more) because `flash-attn` compiles CUDA kernels from source.

#### Other CUDA versions

The installer defaults to the CUDA 12.6 PyTorch wheel index. If your driver supports a different CUDA version (e.g. 12.4, 12.8), pass `--cuda cuXXX` to match:

```bash
./install.sh --cuda cu124   # CUDA 12.4
./install.sh --cuda cu128   # CUDA 12.8
```

Check your driver's maximum supported CUDA version with `nvidia-smi`. Using a wheel index newer than your driver supports will cause runtime errors.

**Quantization variants:**

```bash
./install.sh          # NF4 — recommended (~5 GB VRAM)
./install.sh --int8   # INT8 (~6 GB VRAM)
./install.sh --bf16   # BF16 full precision (~9 GB VRAM)
```

Model weights are downloaded automatically from Hugging Face on first launch.

## Quick start

```bash
source .venv/bin/activate
voxtral-server
```

Or without activating the environment:

```bash
.venv/bin/voxtral-server
```

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is Voxtral speaking.",
    "voice": "casual_female",
    "response_format": "mp3"
  }' -o output.mp3
```

## Examples

Two browser-based demos are included in `examples/`.

### Python API demo (`examples/tts_python_api.py`)

Runs TTS **directly in-process** (no separate server needed). Serves a web UI at `http://localhost:8080` where you can type text, pick a voice, adjust the inter-sentence pause, and hear audio stream in real time.

```bash
python examples/tts_python_api.py
python examples/tts_python_api.py --port 9090 --quantize nf4 --voice-dir /path/to/voices
```

### Server client demo (`examples/tts_server_client.py`)

Connects to a **running `voxtral-server`** and proxies audio to the browser. Useful when the GPU machine and the browser are on different hosts.

```bash
# Start the server on the GPU machine first:
voxtral-server --port 8000

# Then start the client (can be on a different machine):
python examples/tts_server_client.py --tts-url http://localhost:8000 --port 8081
```

Both demos split input text into sentences and stream each sentence separately, appending configurable trailing silence between them for a natural cadence.

## CLI reference

```
voxtral-server [OPTIONS]
```

| Flag                 | Default   | Description                                                        |
|----------------------|-----------|-----------------------------------------------------------------|
| `--device`           | `cuda`    | Torch device (`cuda`, `cuda:1`, `cpu`)                            |
| `--host`             | `0.0.0.0` | Server bind address                                               |
| `--port`             | `8000`    | Server port                                                       |
| `--quantize`         | `nf4`     | LLM quantization: `nf4`, `int8`, or `none` (BF16)                |
| `--voice-dir`        | *(none)*  | Directory of custom `.pt` voice embeddings to load at startup     |
| `--pause-ms`         | `400`     | Milliseconds of silence appended after each response (max: 1000)  |
| `--compile`          | off       | Enable `torch.compile` for ~8% faster inference (+4 GB VRAM)      |
| `--no-cache`         | off       | Disable persistent quantized weight cache                         |
| `--no-sanitize-text` | off       | Disable Latin-script filter — required for Arabic, Hindi, etc.    |

## API reference

### `POST /v1/audio/speech`

Generate speech from text. Returns a streaming audio response.

**Request body**

| Field                | Type             | Default           | Description                                                              |
|----------------------|------------------|-------------------|--------------------------------------------------------------------------|
| `input`              | `string`         | *(required)*      | Text to synthesize (max 4096 characters)                                 |
| `model`              | `string`         | `"voxtral-4b"`    | Model identifier (only `voxtral-4b` supported)                           |
| `voice`              | `string \| dict` | `"neutral_female"` | Voice preset name or `{"id": "voice_name"}`                             |
| `response_format`    | `string`         | `"mp3"`           | `mp3`, `wav`, `pcm`                                                       |
| `speed`              | `float`          | `1.0`             | Speech speed (only `1.0` supported)                                      |
| `stream_format`      | `string \| null` | `null`            | Streaming mode (only `"audio"` supported)                                |
| `trailing_silence_ms`| `integer \| null`| `null`            | Milliseconds of silence appended after synthesis (0–1000; null = server default) |
| `sanitize_text`      | `boolean \| null`| `null`            | `false` required for Arabic, Hindi, and other non-Latin scripts (null = server default) |

**Response** - Streaming audio bytes with the appropriate `Content-Type`.

### `GET /v1/voices`

Returns the list of available voice names.

```json
{"voices": ["casual_female", "casual_male", ...]}
```

Returns `503` while the model is still loading.

### `GET /healthz`

Liveness probe. Always returns `200 OK`.

### `GET /readyz`

Readiness probe. Returns `200` when the model is loaded, `503` while loading.

## Available voices

| Language   | Voices                                   |
|------------|------------------------------------------|
| English    | `casual_female`, `casual_male`, `cheerful_female`, `neutral_female`, `neutral_male` |
| French     | `fr_male`, `fr_female`                   |
| Spanish    | `es_male`, `es_female`                   |
| German     | `de_male`, `de_female`                   |
| Italian    | `it_male`, `it_female`                   |
| Portuguese | `pt_male`, `pt_female`                   |
| Dutch      | `nl_male`, `nl_female`                   |
| Arabic     | `ar_male`                                |
| Hindi      | `hi_male`, `hi_female`                   |

> **Non-Latin scripts (Arabic, Hindi):** The server applies a Latin-script character filter by default to protect against unsupported input. Disable it per-request with `"sanitize_text": false`, or globally for the server process with `--no-sanitize-text`.

## Architecture

```
Text ──► Mistral 3B LLM ──► FlowMatching Acoustic Transformer ──► Codec Decoder ──► 24 kHz Waveform
              │                        │                                │
         26 layers               3 layers, Euler ODE            4-stage cascaded
         3072-dim                 7 steps + CFG                 ALiBi attention
```

The pipeline runs autoregressively - the LLM emits one acoustic frame embedding
per step, which the flow-matching transformer converts to codec tokens via a
7-step ODE with classifier-free guidance. The codec decoder then synthesizes the
waveform in streaming chunks.

## Acknowledgements

Huge thanks to the Mistral team for sharing the Voxtral 4B TTS model!

## License

[CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) - inherited from the [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) model weights and voice references.

Do not forget to attribute the model creators if you use this work in your projects!