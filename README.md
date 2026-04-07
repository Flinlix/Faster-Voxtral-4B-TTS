# Faster Voxtral 4B TTS

OpenAI-compatible text-to-speech server powered by [Voxtral 4B](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).

## Features

- **9 languages** - English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, Hindi
- **20 voice presets** - male and female voices in almost every of the mentioned languages, with 3 English styles (casual, cheerful, neutral)
- **Streaming** - chunked audio delivery with low time-to-first-audio (<200 ms on a NVIDIA RTX 3090), perfect for real-time applications like voice assistants
- **3 output formats** - MP3, WAV, PCM
- **3 quantization modes** - NF4 (~5 GB VRAM), INT8 (~6 GB), full BF16 (~9 GB)
- **OpenAI API compatible** - drop-in replacement for `/v1/audio/speech`

## Requirements

- Python ≥ 3.11
- NVIDIA GPU with ≥ 5 GB VRAM (NF4 quantization)

| Quantization | Approx. VRAM |
|--------------|-------------|
| `nf4`        | ~5 GB       |
| `int8`       | ~6 GB       |
| `none` (BF16)| ~9 GB       |

In a personal blind listening test, no perceptible quality difference was found between full precision, NF4, and INT8 quantization. That is why NF4 is the default - it offers the best VRAM efficiency with no audible quality tradeoff.

## Installation

```bash
# Create a virtual environment and install
uv venv --python 3.13 && source .venv/bin/activate

# For NF4 quantization (recommended, default)
uv pip install -e ".[nf4]"

# For full precision (BF16)
uv pip install -e .

# For INT8 quantization
uv pip install -e ".[int8]"
```

Model weights are downloaded automatically from Hugging Face on first launch.

## Quick start

```bash
python server.py
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

## CLI reference

```
python server.py [OPTIONS]
```

| Flag          | Default   | Description                                       |
|---------------|-----------|---------------------------------------------------|
| `--device`    | `cuda`    | Torch device (`cuda`, `cuda:1`, `cpu`)            |
| `--host`      | `0.0.0.0` | Server bind address                               |
| `--port`      | `8000`    | Server port                                       |
| `--quantize`  | `nf4`     | LLM quantization: `nf4`, `int8`, or `none` (BF16) |

## API reference

### `POST /v1/audio/speech`

Generate speech from text. Returns a streaming audio response.

**Request body**

| Field             | Type             | Default         | Description                                    |
|-------------------|------------------|-----------------|------------------------------------------------|
| `input`           | `string`         | *(required)*    | Text to synthesize (max 4096 characters)       |
| `model`           | `string`         | `"voxtral-4b"`  | Model identifier (only `voxtral-4b` supported) |
| `voice`           | `string \| dict` | `"neutral_female"`   | Voice preset name or `{"id": "voice_name"}`    |
| `response_format` | `string`         | `"mp3"`         | `mp3`, `wav`, `pcm`                             |
| `speed`           | `float`          | `1.0`           | Speech speed (only `1.0` supported)            |
| `stream_format`   | `string \| null` | `null`          | Streaming mode (only `"audio"` supported)      |

**Response** - Streaming audio bytes with the appropriate `Content-Type`.

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

## Architecture

```
Text ──► Mistral 3B LLM ──► FlowMatching Acoustic Transformer ──► Codec Decoder ──► 24 kHz Waveform
              │                        │                                │
         26 layers               3 layers, Euler ODE            4-stage cascaded
         3072-dim                 8 steps + CFG                 ALiBi attention
```

The pipeline runs autoregressively - the LLM emits one acoustic frame embedding
per step, which the flow-matching transformer converts to codec tokens via an
8-step ODE with classifier-free guidance. The codec decoder then synthesizes the
waveform in streaming chunks.

## Acknowledgements

Huge thanks to the Mistral team for sharing the Voxtral 4B TTS model!

## License

[CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) - inherited from the [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) model weights and voice references.

Do not forget to attribute the model creators if you use this work in your projects!