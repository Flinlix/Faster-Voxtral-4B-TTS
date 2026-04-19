"""OpenAI-compatible TTS API server backed by VoxtralTTS.

Usage:
    python server.py
    python server.py --port 8000 --quantize nf4
    python server.py --quantize none --device cuda:1
"""

import argparse
import asyncio
import logging
import struct
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from voxtral import VoxtralTTS
from voxtral.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


# ── Request schema ──────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    model: str = "voxtral-4b"
    input: str
    voice: str | dict = "neutral_female"
    response_format: str = "mp3"
    speed: float = 1.0
    stream_format: str | None = None


# ── Server state ────────────────────────────────────────────────────────


class ServerState:
    """Mutable runtime state managed via FastAPI lifespan."""

    tts_engine: VoxtralTTS | None = None
    inference_lock: asyncio.Lock | None = None


_state = ServerState()
_orphaned_thread_count = [0]  # mutable counter for orphaned generation threads

# Set by CLI before server boots
_cli_device: str = "cuda"
_cli_quantize: str | None = "nf4"
_cli_voice_dir: str | None = None
_cli_compile: bool = False


# ── Lifespan ────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.inference_lock = asyncio.Lock()
    logger.info("Loading model ...")
    _state.tts_engine = VoxtralTTS(
        device=_cli_device, quantize=_cli_quantize, custom_voice_dir=_cli_voice_dir,
        compile=_cli_compile,
    )
    logger.info("Warming up ...")
    _state.tts_engine.generate("warmup", max_frames=5, verbose=False)
    logger.info("Ready.")
    yield
    # Graceful shutdown: free GPU memory
    logger.info("Shutting down - releasing model resources ...")
    _state.tts_engine = None
    _state.inference_lock = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("Shutdown complete.")


app = FastAPI(title="Voxtral TTS", lifespan=lifespan)


# ── Health / Readiness ──────────────────────────────────────────────────────────


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/v1/voices")
async def list_voices() -> JSONResponse:
    if _state.tts_engine is None:
        return JSONResponse(status_code=503, content={"error": "not ready"})
    voices = sorted(_state.tts_engine.voice_embeddings.keys())
    return JSONResponse({"voices": voices})


@app.get("/readyz")
async def readyz() -> JSONResponse:
    if _state.tts_engine is None or _state.inference_lock is None:
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return JSONResponse({"status": "ready"})


# ── Audio encoding helpers ──────────────────────────────────────────────


SUPPORTED_FORMATS = ("mp3", "wav", "pcm")

MEDIA_TYPES = {
    "mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm",
}


def _build_wav_header(sample_rate: int = SAMPLE_RATE, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """WAV header with max-size placeholder for streaming."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    max_data_size = 0x7FFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + max_data_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", max_data_size,
    )


# ── Streaming generator ────────────────────────────────────────────────


async def _generate_and_stream_audio(
    text: str,
    voice: str,
    max_frames: int,
    output_format: str,
    client_disconnect_event: threading.Event,
) -> AsyncGenerator[bytes, None]:
    """Run TTS in a background thread, yield encoded audio chunks via an async queue."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=64)

    def on_audio_chunk(chunk: np.ndarray) -> None:
        if client_disconnect_event.is_set():
            raise RuntimeError("Client disconnected")
        pcm_bytes = (chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        future = asyncio.run_coroutine_threadsafe(queue.put(pcm_bytes), loop)
        future.result(timeout=30)  # block producer; timeout prevents deadlock on teardown

    generation_error: list[Exception | None] = [None]

    def generation_thread() -> None:
        try:
            _state.tts_engine.generate(
                text, voice=voice, max_frames=max_frames,
                stream_callback=on_audio_chunk,
            )
        except RuntimeError as exc:
            if "Client disconnected" in str(exc):
                logger.debug("Generation thread stopped: %s", exc)
            else:
                logger.error("Generation failed: %s", exc)
                generation_error[0] = exc
        except TimeoutError as exc:
            logger.debug("Generation thread stopped: %s", exc)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # safe even when queue is full

    # Set up encoder
    mp3_encoder = None

    if output_format == "mp3":
        import lameenc
        mp3_encoder = lameenc.Encoder()
        mp3_encoder.set_bit_rate(128)
        mp3_encoder.set_in_sample_rate(SAMPLE_RATE)
        mp3_encoder.set_channels(1)
        mp3_encoder.set_quality(2)
        mp3_encoder.silence()
    elif output_format == "wav":
        yield _build_wav_header()

    thread = threading.Thread(target=generation_thread, daemon=True)
    thread.start()

    received_chunks = 0
    encoded_chunks = 0
    try:
        while True:
            data = await queue.get()
            if data is None:
                break
            received_chunks += 1
            if mp3_encoder:
                encoded = mp3_encoder.encode(data)
                if encoded:
                    encoded_chunks += 1
                    yield bytes(encoded)
            else:
                encoded_chunks += 1
                yield data

        # Flush remaining MP3 data
        if mp3_encoder:
            remaining = mp3_encoder.flush()
            if remaining:
                yield bytes(remaining)

        if mp3_encoder and received_chunks > 0 and encoded_chunks == 0:
            logger.error("MP3 encoder produced no output after %d audio chunks", received_chunks)

        if generation_error[0] is not None:
            logger.error("Generation error during streaming: %s", generation_error[0])
    except (asyncio.CancelledError, GeneratorExit):
        client_disconnect_event.set()
        raise
    finally:
        await asyncio.to_thread(thread.join, timeout=30)
        if thread.is_alive():
            _orphaned_thread_count[0] += 1
            logger.warning(
                "Generation thread did not exit within 30 s - orphaned (total: %d)",
                _orphaned_thread_count[0],
            )


# ── API endpoint ────────────────────────────────────────────────────────


@app.post("/v1/audio/speech", response_model=None)
async def create_speech(request: TTSRequest) -> StreamingResponse | JSONResponse:
    if _state.tts_engine is None or _state.inference_lock is None:
        return JSONResponse(status_code=503, content={
            "error": {"message": "Service not ready", "type": "server_error"},
        })

    if not request.input or not request.input.strip():
        return JSONResponse(status_code=400, content={
            "error": {"message": "input is required", "type": "invalid_request_error"},
        })

    if len(request.input) > 4096:
        return JSONResponse(status_code=400, content={
            "error": {
                "message": f"input is too long ({len(request.input)} chars). Maximum is 4096.",
                "type": "invalid_request_error",
            },
        })

    if request.model != "voxtral-4b":
        return JSONResponse(status_code=400, content={
            "error": {
                "message": f"Unknown model '{request.model}'. Only 'voxtral-4b' is available.",
                "type": "invalid_request_error",
            },
        })

    if request.speed != 1.0:
        return JSONResponse(status_code=400, content={
            "error": {
                "message": "speed parameter is not supported. Only 1.0 is accepted.",
                "type": "invalid_request_error",
            },
        })

    if request.stream_format is not None and request.stream_format != "audio":
        return JSONResponse(status_code=400, content={
            "error": {
                "message": f"Unsupported stream_format '{request.stream_format}'. Only 'audio' is supported.",
                "type": "invalid_request_error",
            },
        })

    # OpenAI-compatible: voice can be a string or an object with an ``id`` field
    if isinstance(request.voice, dict):
        voice = request.voice.get("id")
    else:
        voice = request.voice

    if not isinstance(voice, str) or not voice:
        return JSONResponse(status_code=400, content={
            "error": {
                "message": "voice must be a string or an object with an 'id' field.",
                "type": "invalid_request_error",
            },
        })

    if voice not in _state.tts_engine.voice_embeddings:
        available = ", ".join(sorted(_state.tts_engine.voice_embeddings.keys()))
        return JSONResponse(status_code=400, content={
            "error": {
                "message": f"Unknown voice '{voice}'. Available: {available}",
                "type": "invalid_request_error",
            },
        })

    output_format = request.response_format.lower()
    if output_format not in SUPPORTED_FORMATS:
        return JSONResponse(status_code=400, content={
            "error": {
                "message": (
                    f"Unsupported format '{output_format}'. "
                    f"Use one of: {', '.join(SUPPORTED_FORMATS)}."
                ),
                "type": "invalid_request_error",
            },
        })

    media_type = MEDIA_TYPES[output_format]
    client_disconnect_event = threading.Event()

    async def locked_stream():
        async with _state.inference_lock:
            async for chunk in _generate_and_stream_audio(
                request.input, voice, 2000, output_format, client_disconnect_event,
            ):
                yield chunk

    return StreamingResponse(locked_stream(), media_type=media_type)


# ── CLI entrypoint ──────────────────────────────────────────────────────


def main():
    global _cli_device, _cli_quantize, _cli_voice_dir, _cli_compile
    parser = argparse.ArgumentParser(description="Voxtral TTS - OpenAI-compatible API server")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--quantize", choices=["nf4", "int8", "none"], default="nf4",
        help="LLM quantization: nf4 (default), int8, or none for full bf16",
    )
    parser.add_argument(
        "--voice-dir", default=None,
        help="Directory containing custom .pt voice embeddings",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile for ~8%% faster inference (+4 GB VRAM)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _cli_device = args.device
    _cli_quantize = None if args.quantize == "none" else args.quantize
    _cli_voice_dir = args.voice_dir
    _cli_compile = args.compile
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
