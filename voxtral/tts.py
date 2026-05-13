"""VoxtralTTS - the top-level orchestrator for text-to-speech synthesis.

Loads all model components (LLM, acoustic transformer, codec decoder,
audio token embedding), performs weight loading + optional quantization,
and exposes a ``generate()`` method for inference.
"""

import contextlib
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flash_attn import flash_attn_with_kvcache
from huggingface_hub import hf_hub_download
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.args import TransformerArgs
from mistral_inference.cache import BufferCache
from mistral_inference.rope import apply_rotary_emb
from mistral_inference.transformer import Transformer

from voxtral.config import (
    VoxtralConfig,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    CODEC_CONTEXT_FRAMES,
    NUM_SPECIAL_TOKENS,
    END_AUDIO_ID,
)
from voxtral.acoustic.flow_matching import FlowMatchingAudioTransformer
from voxtral.codec.decoder import CodecDecoder
from voxtral.embedding import AudioTokenEmbedding
from voxtral.weights import get_checkpoint_path, load_checkpoint_weights

MAX_CACHE_LEN = 4096  # max prompt + generation tokens for CUDA graph KV cache

logger = logging.getLogger(__name__)

# Available voice presets shipped with the model
VOICE_PRESETS = [
    "casual_female", "casual_male", "cheerful_female",
    "neutral_female", "neutral_male",
    "pt_male", "pt_female", "nl_male", "nl_female",
    "it_male", "it_female", "fr_male", "fr_female",
    "es_male", "es_female", "de_male", "de_female",
    "ar_male", "hi_male", "hi_female",
]


class VoxtralTTS:
    """End-to-end Voxtral TTS pipeline.

    The pipeline runs LLM → flow-matching acoustic transformer → codec decoder
    autoregressively, yielding 24 kHz mono audio.

    Example:
        >>> from voxtral import VoxtralTTS
        >>> tts = VoxtralTTS()
        >>> audio = tts.generate("Hello world", voice="neutral_female")

    Args:
        config: Model configuration. Defaults to ``VoxtralConfig.voxtral_4b()``.
        device: Torch device string (e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``).
        dtype: Floating-point dtype for non-quantized weights.
        quantize: LLM quantization mode. One of ``"nf4"`` (default), ``"int8"``,
            or ``None`` for full ``dtype`` precision.
        custom_voice_dir: Optional directory of ``*.pt`` voice embeddings to
            register at startup.
        compile: If ``True``, apply ``torch.compile`` before CUDA graph capture
            (~8% faster, +4 GB VRAM).
        use_cache: If ``True``, persist quantized weights to disk and reuse
            them on subsequent runs.
    """

    model_name: str = "voxtral-4b"

    def __init__(
        self,
        config: VoxtralConfig | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quantize: str | None = "nf4",
        custom_voice_dir: str | None = None,
        compile: bool = False,
        use_cache: bool = True,
    ):
        if config is None:
            config = VoxtralConfig.voxtral_4b()
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self._compile = compile
        self._use_cache = use_cache
        self._graphs_ready = threading.Event()
        self._graph_capture_error: BaseException | None = None

        self._llm_args = TransformerArgs(
            dim=config.llm.model_dim,
            n_layers=config.llm.num_layers,
            head_dim=config.llm.head_dim,
            hidden_dim=config.llm.hidden_dim,
            n_heads=config.llm.num_heads,
            n_kv_heads=config.llm.num_kv_heads,
            norm_eps=config.llm.norm_eps,
            vocab_size=config.llm.vocab_size,
            max_batch_size=1,
            rope_theta=config.llm.rope_theta,
        )

        # On a cache hit, tokenizer.from_hf_hub (0.9 s) and CodecDecoder
        # construction (0.6 s) are independent of torch.load (2.9 s for the
        # 3.7 GB cache file). Run all three in parallel to hide the two
        # shorter tasks behind the longer one.
        _tokenizer_result: list = []
        _tokenizer_exc: list[BaseException] = []
        _codec_result: list = []
        _codec_exc: list[BaseException] = []
        _cache_result: list = []
        _cache_exc: list[BaseException] = []

        def _load_tokenizer():
            try:
                _tokenizer_result.append(MistralTokenizer.from_hf_hub(config.repo_id))
            except BaseException as exc:
                _tokenizer_exc.append(exc)

        def _build_codec():
            try:
                _codec_result.append(CodecDecoder(config.codec))
            except BaseException as exc:
                _codec_exc.append(exc)

        cache_path = self._cache_path(quantize)
        cache_exists = self._use_cache and cache_path.exists()

        if cache_exists:
            def _do_torch_load():
                try:
                    try:
                        blob = torch.load(
                            cache_path, map_location=self.device,
                            weights_only=False, mmap=True,
                        )
                    except TypeError:
                        # mmap= unsupported on older torch builds
                        blob = torch.load(cache_path, map_location=self.device, weights_only=False)
                    _cache_result.append(blob)
                except BaseException as exc:
                    _cache_exc.append(exc)

            logger.info("Loading weights (cache) + tokenizer in parallel ...")
            _threads = [
                threading.Thread(target=_load_tokenizer, daemon=True),
                threading.Thread(target=_build_codec, daemon=True),
                threading.Thread(target=_do_torch_load, daemon=True),
            ]
        else:
            logger.info("Loading tokenizer + building codec ...")
            _threads = [
                threading.Thread(target=_load_tokenizer, daemon=True),
                threading.Thread(target=_build_codec, daemon=True),
            ]

        for _t in _threads: _t.start()
        for _t in _threads: _t.join()

        if _tokenizer_exc:
            raise RuntimeError("Tokenizer initialization failed") from _tokenizer_exc[0]
        if _codec_exc:
            raise RuntimeError("Codec initialization failed") from _codec_exc[0]
        if _cache_exc:
            raise RuntimeError("Weight cache load failed") from _cache_exc[0]

        self.tokenizer = _tokenizer_result[0]
        self.codec_decoder = _codec_result[0]

        logger.info("Loading weights ...")
        if not self._try_load_from_cache(quantize, preloaded_blob=_cache_result[0] if _cache_result else None):
            # Cold path: build the remaining modules, load weights, quantize.
            logger.info("Building modules ...")
            self.llm = Transformer(self._llm_args, pipeline_rank=0, num_pipeline_ranks=1)
            self.acoustic_transformer = FlowMatchingAudioTransformer(config.acoustic)
            self.audio_token_embedding = AudioTokenEmbedding(
                embedding_dim=config.llm.model_dim,
                semantic_codebook_size=config.codec.semantic_codebook_size,
                acoustic_codebook_size=config.codec.acoustic_codebook_size,
                num_acoustic_codebooks=config.codec.num_acoustic_codebooks,
            )
            load_checkpoint_weights(
                self.llm, self.acoustic_transformer,
                self.codec_decoder, self.audio_token_embedding,
                config.repo_id,
            )
            self._apply_quantization_and_move_to_device(quantize)
            self._save_to_cache(quantize)

        self.voice_embeddings: dict[str, torch.Tensor] = {}
        self._load_voice_embeddings()
        self._load_custom_voices(custom_voice_dir)

        self._audio_placeholder_token_id = (
            self.tokenizer.instruct_tokenizer.audio_encoder.special_ids.audio
        )

        if self.device.type == "cuda":
            self._graph_thread = threading.Thread(
                target=self._init_cuda_graphs_background,
                name="voxtral-graph-capture",
                daemon=True,
            )
            self._graph_thread.start()
        else:
            self._graphs_ready.set()

    def _apply_quantization_and_move_to_device(self, quantize: str | None) -> None:
        """Quantize the LLM (if requested) and move all modules to device."""
        if quantize == "int8":
            from torchao.quantization import quantize_, Int8WeightOnlyConfig

            logger.info("Quantizing to INT8 ...")
            self.llm = self.llm.to(dtype=self.dtype).eval()
            quantize_(self.llm, Int8WeightOnlyConfig())
            self.llm = self.llm.to(device=self.device)

        elif quantize == "nf4":
            import bitsandbytes as bnb

            logger.info("Quantizing to NF4 ...")
            self.llm = self.llm.to(dtype=self.dtype).eval()
            for name, module in self.llm.named_modules():
                if isinstance(module, nn.Linear):
                    nf4_linear = bnb.nn.LinearNF4(
                        module.in_features, module.out_features,
                        bias=module.bias is not None,
                    )
                    nf4_linear.weight = bnb.nn.Params4bit(
                        module.weight.data, requires_grad=False, quant_type="nf4",
                    )
                    if module.bias is not None:
                        nf4_linear.bias = module.bias
                    # Replace the linear layer in its parent module
                    parts = name.rsplit(".", 1)
                    parent = (
                        self.llm if len(parts) == 1
                        else dict(self.llm.named_modules())[parts[0]]
                    )
                    setattr(parent, parts[-1], nf4_linear)
            self.llm = self.llm.to(device=self.device)

        else:
            self.llm = self.llm.to(device=self.device, dtype=self.dtype).eval()

        self.acoustic_transformer = (
            self.acoustic_transformer.to(device=self.device, dtype=self.dtype).eval()
        )
        self.codec_decoder = (
            self.codec_decoder.to(device=self.device, dtype=self.dtype).eval()
        )
        self.audio_token_embedding = (
            self.audio_token_embedding.to(device=self.device, dtype=self.dtype).eval()
        )

    def _load_voice_embeddings(self) -> None:
        """Download and cache all available voice embedding presets in parallel."""
        def _load_one(voice_name: str) -> tuple[str, torch.Tensor | None]:
            try:
                path = hf_hub_download(self.config.repo_id, f"voice_embedding/{voice_name}.pt")
                tensor = torch.load(path, map_location="cpu", weights_only=True)
                return voice_name, tensor.to(device=self.device, dtype=self.dtype)
            except Exception as exc:
                logger.warning("Failed to load voice '%s': %s", voice_name, exc)
                return voice_name, None

        with ThreadPoolExecutor(max_workers=min(8, len(VOICE_PRESETS))) as pool:
            for voice_name, tensor in pool.map(_load_one, VOICE_PRESETS):
                if tensor is not None:
                    self.voice_embeddings[voice_name] = tensor
        if not self.voice_embeddings:
            raise RuntimeError(
                "No voice embeddings loaded - cannot serve TTS requests. "
                "Check network connectivity and the HuggingFace repo."
            )
        logger.info("Loaded %d voice embeddings: %s",
                    len(self.voice_embeddings), list(self.voice_embeddings.keys()))

    def _load_custom_voices(self, voice_dir: str | None = None) -> None:
        """Scan a directory for custom .pt voice embeddings and register them."""
        if voice_dir is None:
            return
        voice_path = Path(voice_dir)
        if not voice_path.exists():
            logger.debug("Custom voice directory does not exist: %s", voice_path)
            return
        for pt_file in sorted(voice_path.glob("*.pt")):
            name = pt_file.stem
            try:
                embedding = torch.load(str(pt_file), map_location="cpu", weights_only=True)
                self.register_voice(name, embedding)
            except Exception as exc:
                logger.warning("Failed to load custom voice '%s': %s", name, exc)

    def register_voice(self, name: str, embedding: torch.Tensor) -> None:
        """Register a custom voice embedding for use with ``generate()``.

        Args:
            name: Voice name (passed as ``voice=name`` to ``generate``/``stream``).
            embedding: 2-D tensor of shape ``[N, model_dim]`` (``model_dim`` is
                3072 for Voxtral-4B).

        Raises:
            ValueError: If ``embedding`` is not 2-D or has the wrong width.
        """
        if name in self.voice_embeddings:
            logger.warning("Voice '%s' already exists, overwriting", name)
        expected_width = self.config.llm.model_dim
        if embedding.ndim != 2:
            raise ValueError(
                f"Voice embedding '{name}' must be a 2-D tensor [N, {expected_width}], "
                f"got shape {tuple(embedding.shape)}"
            )
        if embedding.shape[1] != expected_width:
            raise ValueError(
                f"Voice embedding '{name}' has width {embedding.shape[1]}, "
                f"expected {expected_width}"
            )
        self.voice_embeddings[name] = embedding.to(
            device=self.device, dtype=self.dtype
        )
        num_tokens = embedding.shape[0]
        # Patch the tokenizer config so it knows how many audio tokens this voice needs
        self.tokenizer.instruct_tokenizer.audio_encoder.audio_config.voice_num_audio_tokens[
            name
        ] = num_tokens
        logger.info("Registered voice '%s' (%d tokens)", name, num_tokens)

    # ── Public library API ──────────────────────────────────────────────

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return SAMPLE_RATE

    def list_voices(self) -> list[str]:
        """Return sorted list of available voice names."""
        return sorted(self.voice_embeddings.keys())

    def has_voice(self, voice: str) -> bool:
        """Return True if *voice* is loaded and available."""
        return voice in self.voice_embeddings

    def wait_for_ready(self, timeout: float | None = None) -> None:
        """Block until background CUDA graph capture has completed.

        ``stream()`` and ``generate()`` call this automatically, but you can
        invoke it explicitly to make sure the first synthesis has no startup
        latency (e.g. right before serving the first request).

        Raises:
            TimeoutError: if *timeout* elapses before capture finishes.
            RuntimeError: if graph capture failed in the background.
        """
        if not self._graphs_ready.wait(timeout=timeout):
            raise TimeoutError("CUDA graph capture did not complete within timeout")
        if self._graph_capture_error is not None:
            raise RuntimeError("CUDA graph capture failed") from self._graph_capture_error

    # ── Persistent quantized weight cache ────────────────────────────

    def _cache_path(self, quantize: str | None) -> Path:
        """Filesystem location of the persistent state-dict cache.

        Filename axes (any change → different file): config repo, quantize
        mode, dtype, device kind, torch & CUDA versions.
        """
        cache_dir = Path(os.environ.get(
            "VOXTRAL_CACHE_DIR",
            Path.home() / ".cache" / "voxtral",
        ))
        cache_dir.mkdir(parents=True, exist_ok=True)
        repo_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.config.repo_id)
        dtype_str = str(self.dtype).removeprefix("torch.")
        cuda_ver = (torch.version.cuda or "none").replace(".", "")
        torch_ver = torch.__version__.split("+")[0].replace(".", "")
        stamp = f"{repo_slug}-{quantize or 'full'}-{dtype_str}-{self.device.type}-pt{torch_ver}-cu{cuda_ver}"
        return cache_dir / f"voxtral-{stamp}.pt"

    def _try_load_from_cache(self, quantize: str | None, preloaded_blob=None) -> bool:
        """Try to restore quantized + device-resident modules from disk.

        Returns True on success; False if the cache is missing, stale, or
        invalid (caller must then run the full load + quantize path).

        We pickle whole module objects rather than state_dicts because bnb's
        ``Params4bit`` packs quantized weights into an opaque shape that
        ``load_state_dict`` cannot reconstruct from a fresh ``LinearNF4``.

        Security: ``weights_only=False`` deserialises arbitrary Python objects via
        pickle. Only load caches from trusted paths (files written by this process).
        Never point ``VOXTRAL_CACHE_DIR`` at a world-writable or untrusted location.
        """
        if not self._use_cache:
            return False
        cache_path = self._cache_path(quantize)
        if not cache_path.exists():
            return False
        try:
            source_mtime = Path(get_checkpoint_path(self.config.repo_id)).stat().st_mtime
        except Exception:
            source_mtime = None

        if preloaded_blob is not None:
            blob = preloaded_blob
        else:
            try:
                try:
                    blob = torch.load(
                        cache_path, map_location=self.device,
                        weights_only=False, mmap=True,
                    )
                except TypeError:
                    blob = torch.load(cache_path, map_location=self.device, weights_only=False)
            except Exception as exc:
                logger.warning("Could not load weight cache (%s); regenerating", exc)
                return False

        if source_mtime is not None and blob.get("source_mtime") != source_mtime:
            logger.info("Source weights changed since cache was written; regenerating")
            return False

        logger.info("Restoring weights from cache: %s", cache_path)
        try:
            self.llm = blob["llm"].eval()
            self.acoustic_transformer = blob["acoustic"].eval()
            self.audio_token_embedding = blob["embedding"].eval()
            # codec uses torch.nn.utils.parametrize and can't be pickled, so
            # we cache its state_dict and load it back into the freshly built module.
            self.codec_decoder.load_state_dict(blob["codec"], strict=False)
            self.codec_decoder.to(device=self.device, dtype=self.dtype).eval()
        except Exception as exc:
            logger.warning("Cache load failed (%s); regenerating", exc)
            return False

        # Invalidate cached semantic embedding (recomputed on first use).
        self.codec_decoder.quantizer.semantic_codebook._embedding = None
        return True

    def _save_to_cache(self, quantize: str | None) -> None:
        if not self._use_cache:
            return
        cache_path = self._cache_path(quantize)
        try:
            source_mtime = Path(get_checkpoint_path(self.config.repo_id)).stat().st_mtime
        except Exception:
            source_mtime = None
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            torch.save(
                {
                    "llm": self.llm,
                    "acoustic": self.acoustic_transformer,
                    "codec": self.codec_decoder.state_dict(),
                    "embedding": self.audio_token_embedding,
                    "source_mtime": source_mtime,
                },
                tmp_path,
            )
            tmp_path.replace(cache_path)
            logger.info("Saved quantized weight cache: %s", cache_path)
        except Exception as exc:
            logger.warning("Could not save weight cache: %s", exc)
            with contextlib.suppress(Exception):
                tmp_path.unlink()

    # ── CUDA Graph infrastructure ───────────────────────────────────────
    def _init_cuda_graphs_background(self) -> None:
        """Run graph capture in a background thread, signalling readiness."""
        try:
            self._init_cuda_graphs()
        except BaseException as exc:  # noqa: BLE001 - propagate to wait_for_ready
            self._graph_capture_error = exc
            logger.exception("CUDA graph capture failed")
        finally:
            self._graphs_ready.set()
    def _init_cuda_graphs(self) -> None:
        """Pre-allocate persistent buffers and capture CUDA graphs for decode."""
        n_layers = self._llm_args.n_layers
        n_kv_heads = self._llm_args.n_kv_heads
        head_dim = self._llm_args.head_dim
        dim = self._llm_args.dim

        # Persistent KV cache shared between prefill (BufferCache) and decode (graph)
        self._persistent_cache = BufferCache(
            n_layers=n_layers,
            max_batch_size=1,
            max_seq_len=MAX_CACHE_LEN,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        self._persistent_cache.to(device=self.device, dtype=self.dtype)

        # Static tensors for CUDA graphs
        self._graph_cache_seqlens = torch.zeros(1, device=self.device, dtype=torch.int32)
        self._decode_input = torch.zeros(1, dim, device=self.device, dtype=self.dtype)
        self._acoustic_input = torch.zeros(1, dim, device=self.device, dtype=self.dtype)

        # Optional torch.compile for kernel fusion before graph capture
        if self._compile:
            compile_opts = {"mode": "max-autotune-no-cudagraphs"}
            for layer in self.llm.layers.values():
                layer.feed_forward = torch.compile(layer.feed_forward, **compile_opts)
                layer.attention_norm = torch.compile(layer.attention_norm, **compile_opts)
                layer.ffn_norm = torch.compile(layer.ffn_norm, **compile_opts)
            self.llm.norm = torch.compile(self.llm.norm, **compile_opts)
            self.acoustic_transformer._predict_velocity = torch.compile(
                self.acoustic_transformer._predict_velocity, **compile_opts
            )

        # Capture graphs
        logger.info("Capturing CUDA graphs ...")
        self._llm_decode_graph, self._decode_output = self._capture_llm_decode()
        self._acoustic_graph, self._acoustic_output = self._capture_acoustic()
        logger.info("CUDA graphs captured")

    def _llm_decode_step(self) -> torch.Tensor:
        """One LLM decode step using flash_attn_with_kvcache on static buffers."""
        hidden = self._decode_input  # [1, D]
        n_heads = self._llm_args.n_heads
        n_kv_heads = self._llm_args.n_kv_heads
        head_dim = self._llm_args.head_dim

        freqs_cis = self.llm.freqs_cis[self._graph_cache_seqlens.long()]

        for layer_id, layer in enumerate(self.llm.layers.values()):
            normed = layer.attention_norm(hidden)

            xq = layer.attention.wq(normed).view(1, n_heads, head_dim)
            xk = layer.attention.wk(normed).view(1, n_kv_heads, head_dim)
            xv = layer.attention.wv(normed).view(1, n_kv_heads, head_dim)

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            attn_out = flash_attn_with_kvcache(
                xq.unsqueeze(0),
                self._persistent_cache.cache_k[layer_id],
                self._persistent_cache.cache_v[layer_id],
                xk.unsqueeze(0), xv.unsqueeze(0),
                cache_seqlens=self._graph_cache_seqlens,
                causal=True,
            )
            attn_out = layer.attention.wo(attn_out.view(1, n_heads * head_dim))
            hidden = hidden + attn_out

            hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))

        self._graph_cache_seqlens.add_(1)
        return self.llm.norm(hidden)

    def _capture_llm_decode(self) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        """Capture LLM decode step as a CUDA graph."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._graph_cache_seqlens.fill_(100)
            self._llm_decode_step()
        torch.cuda.current_stream().wait_stream(s)

        self._graph_cache_seqlens.fill_(100)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._llm_decode_step()
        return graph, output

    def _capture_acoustic(self) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        """Capture acoustic transformer forward as a CUDA graph."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.acoustic_transformer(self._acoustic_input)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.acoustic_transformer(self._acoustic_input)
        return graph, output

    def _graphed_llm_decode(self, input_embedding: torch.Tensor) -> torch.Tensor:
        """Run one LLM decode step via CUDA graph replay."""
        self._decode_input.copy_(input_embedding)
        self._llm_decode_graph.replay()
        return self._decode_output

    def _graphed_acoustic(self, llm_hidden_state: torch.Tensor) -> torch.Tensor:
        """Run acoustic transformer forward via CUDA graph replay."""
        self._acoustic_input.copy_(llm_hidden_state)
        self._acoustic_graph.replay()
        return self._acoustic_output

    # ── LLM helpers ─────────────────────────────────────────────────────

    def _run_llm_with_precomputed_embeddings(
        self,
        input_embeddings: torch.Tensor,
        sequence_lengths: list[int],
        cache: BufferCache,
    ) -> torch.Tensor:
        """Run LLM layers on pre-computed embeddings, bypassing ``tok_embeddings``.

        Args:
            input_embeddings: [num_tokens, dim] tensor (voice embedding already injected).
            sequence_lengths: List of per-sequence lengths for the attention mask.
            cache: KV cache for incremental decoding.

        Returns:
            [num_tokens, dim] normalised hidden states.
        """
        input_metadata = cache.get_input_metadata(sequence_lengths)
        freqs_cis = self.llm.freqs_cis[input_metadata[0].positions]
        hidden_state = input_embeddings
        for local_layer_id, layer in enumerate(self.llm.layers.values()):
            cache_view = cache.get_view(local_layer_id, input_metadata[local_layer_id])
            hidden_state = layer(hidden_state, freqs_cis, cache_view)
        cache.update_seqlens(sequence_lengths)
        return self.llm.norm(hidden_state)

    # Whitelist: Latin + Latin Extended (covers DE/FR/ES/etc.), digits,
    # whitespace, and punctuation a TTS model can meaningfully pronounce.
    _TEXT_FILTER_RE = re.compile(
        r'[^\u0020-\u007E'           # Basic Latin (ASCII printable)
        r'\u00A1-\u024F'             # Latin-1 Supplement + Latin Extended A/B
        r'\u2018\u2019\u201C\u201D'  # typographic quotes \u2018\u2019\u201C\u201D
        r'\u2013\u2014'              # en-dash, em-dash
        r'\u2026'                    # ellipsis
        r']',
        re.UNICODE,
    )

    @classmethod
    def _clean_text(cls, text: str, *, sanitize: bool = True) -> str:
        """Normalise whitespace and optionally strip non-Latin characters.

        Args:
            text: Raw input text.
            sanitize: If ``True``, drop characters outside the Latin/typographic
                whitelist.

        Returns:
            Cleaned text with collapsed whitespace and trimmed ends.
        """
        text = text.replace("\n", " ").replace("\r", " ")
        if sanitize:
            text = cls._TEXT_FILTER_RE.sub("", text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def stream(
        self,
        text: str,
        voice: str = "neutral_female",
        max_frames: int = 2000,
        trailing_silence_ms: int = 0,
        sanitize_text: bool = True,
    ) -> Iterator[np.ndarray]:
        """Synthesise text and yield audio chunks as they are generated.

        The first chunk is yielded after the very first frame is decoded, giving
        minimal time-to-first-audio. Each chunk is approximately 80 ms of audio.

        Args:
            text: Input text to synthesise.
            voice: Voice preset name or a previously registered custom voice.
            max_frames: Hard cap on generated audio frames (12.5 Hz, ~80 ms each).
            trailing_silence_ms: Milliseconds of silence to append after the
                last audio chunk. Clamped to ``[0, 1000]``. Useful for adding
                natural pauses between sentences when calling ``stream()``
                repeatedly in sequence.
            sanitize_text: If ``True``, strip non-Latin characters before
                synthesis. Set to ``False`` for Arabic, Hindi, or other
                non-Latin scripts.

        Yields:
            ``np.ndarray`` of ``float32`` audio samples in ``[-1, 1]`` at
            24 kHz, mono.

        Raises:
            ValueError: If ``voice`` is unknown, or if ``text`` is empty after
                sanitization (and ``sanitize_text`` was ``True``).
            RuntimeError: If the model produces no output frames.
        """
        with torch.inference_mode():
            # 0. Normalise whitespace, cap silence, and optionally filter characters
            trailing_silence_ms = min(max(trailing_silence_ms, 0), 1000)
            text = self._clean_text(text, sanitize=sanitize_text)
            if not text:
                if sanitize_text:
                    raise ValueError(
                        "Text is empty after sanitization. "
                        "Use sanitize_text=False to synthesise Arabic, Hindi, or other non-Latin scripts."
                    )
                return
            if not self.has_voice(voice):
                available = ", ".join(self.list_voices())
                raise ValueError(f"Unknown voice '{voice}'. Available: {available}")

            # Wait for background CUDA graph capture to finish before deciding
            # whether the graphed fast path is available.
            use_graphs = self.device.type == "cuda"
            if use_graphs:
                self.wait_for_ready()

            # 1. Tokenize
            request = SpeechRequest(input=text, voice=voice)
            tokenized = self.tokenizer.encode_speech_request(request)
            input_ids = torch.tensor(tokenized.tokens, dtype=torch.long, device=self.device)

            # 2. Build input embeddings with voice embedding injection
            voice_embedding = self.voice_embeddings[voice]
            audio_token_mask = input_ids == self._audio_placeholder_token_id
            input_embeddings = self.llm.tok_embeddings(input_ids)  # [seq_len, D]
            input_embeddings[audio_token_mask] = voice_embedding

            # 3. Create KV cache and run prefill
            prefill_length = input_ids.shape[0]
            total_seq_len = prefill_length + max_frames + 16

            if use_graphs and total_seq_len <= MAX_CACHE_LEN:
                self._persistent_cache.reset()
                cache = self._persistent_cache
            else:
                if use_graphs:
                    logger.warning(
                        "Sequence length %d exceeds MAX_CACHE_LEN %d, falling back to non-graphed decode",
                        total_seq_len, MAX_CACHE_LEN,
                    )
                    use_graphs = False
                cache = BufferCache(
                    n_layers=self._llm_args.n_layers,
                    max_batch_size=1,
                    max_seq_len=total_seq_len,
                    n_kv_heads=self._llm_args.n_kv_heads,
                    head_dim=self._llm_args.head_dim,
                )
                cache.to(device=self.device, dtype=self.dtype)
                cache.reset()

            hidden_state = self._run_llm_with_precomputed_embeddings(
                input_embeddings, sequence_lengths=[prefill_length], cache=cache,
            )
            hidden_state = hidden_state[-1:]  # last token → [1, D]

            if use_graphs:
                self._graph_cache_seqlens.fill_(prefill_length)

            # 4. Autoregressive generation — yield audio chunks as frames are decoded
            generated_audio_codes: list[torch.Tensor] = []
            num_yielded_frames = 0

            for _ in range(max_frames):
                if use_graphs:
                    audio_codes = self._graphed_acoustic(hidden_state)
                else:
                    audio_codes = self.acoustic_transformer(hidden_state)  # [1, 37]

                if audio_codes[0, 0].item() == END_AUDIO_ID:
                    break

                generated_audio_codes.append(audio_codes[0].cpu())

                # Decode and yield new audio using windowed codec context
                context_start = max(0, num_yielded_frames - CODEC_CONTEXT_FRAMES)
                chunk_audio = self._decode_audio_codes_to_waveform(generated_audio_codes[context_start:])
                skip_samples = (num_yielded_frames - context_start) * SAMPLES_PER_FRAME
                new_audio = chunk_audio[skip_samples:]
                if len(new_audio) > 0:
                    yield new_audio
                num_yielded_frames = len(generated_audio_codes)

                # LLM decode for next step
                codes_for_embedding = audio_codes.unsqueeze(-1)  # [1, 37, 1]
                next_token_embedding = self.audio_token_embedding(codes_for_embedding)  # [1, 1, D]
                next_token_embedding = next_token_embedding.squeeze(0)  # [1, D]

                if use_graphs:
                    hidden_state = self._graphed_llm_decode(next_token_embedding)
                else:
                    hidden_state = self._run_llm_with_precomputed_embeddings(
                        next_token_embedding, sequence_lengths=[1], cache=cache,
                    )

            if not generated_audio_codes:
                raise RuntimeError("No audio frames generated - the model produced no output for this input")

        if trailing_silence_ms > 0:
            silence_samples = int(SAMPLE_RATE * trailing_silence_ms / 1000)
            yield np.zeros(silence_samples, dtype=np.float32)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        voice: str = "neutral_female",
        max_frames: int = 2000,
        trailing_silence_ms: int = 0,
        sanitize_text: bool = True,
        stream_callback: Callable[[np.ndarray], None] | None = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """Generate and return a complete audio waveform from text.

        For real-time applications prefer ``stream()`` which yields chunks
        as they are generated, giving minimal time-to-first-audio.

        Args:
            text: Input text to synthesise.
            voice: Voice preset name or registered custom voice.
            max_frames: Hard cap on generated audio frames (12.5 Hz, ~80 ms each).
            trailing_silence_ms: Milliseconds of silence appended after
                synthesis. Clamped to ``[0, 1000]``.
            sanitize_text: If ``True``, strip non-Latin characters before
                synthesis. Set to ``False`` for non-Latin scripts.
            stream_callback: Optional callback invoked with each decoded chunk.
            verbose: If ``True``, log generation speed statistics.

        Returns:
            ``np.ndarray`` of ``float32`` audio samples in ``[-1, 1]`` at
            24 kHz, mono.

        Raises:
            ValueError: If ``voice`` is unknown, or if ``text`` is empty after
                sanitization (and ``sanitize_text`` was ``True``).
            RuntimeError: If the model produces no output frames.
        """
        generation_start_time = time.perf_counter()
        chunks: list[np.ndarray] = []

        for chunk in self.stream(text, voice=voice, max_frames=max_frames,
                                  trailing_silence_ms=trailing_silence_ms,
                                  sanitize_text=sanitize_text):
            chunks.append(chunk)
            if stream_callback is not None:
                stream_callback(chunk)

        if not chunks:
            raise RuntimeError("No audio frames generated - the model produced no output for this input")

        audio = np.concatenate(chunks)
        generation_time = time.perf_counter() - generation_start_time

        if verbose:
            # Exclude the trailing-silence chunk (if any) from the frame count.
            num_frames = len(chunks) - (1 if trailing_silence_ms > 0 else 0)
            duration_seconds = len(audio) / SAMPLE_RATE
            logger.info(
                "%d frames (%.2fs audio) in %.2fs (%.1f frames/s, %.2fx realtime)",
                num_frames, duration_seconds, generation_time,
                num_frames / generation_time, duration_seconds / generation_time,
            )

        return audio

    def _decode_audio_codes_to_waveform(
        self,
        audio_code_frames: list[torch.Tensor],
        chunk_size: int = 375,
    ) -> np.ndarray:
        """Decode a list of per-frame code tensors to a numpy audio waveform.

        Uses chunked decoding to avoid OOM from large attention bias matrices
        in the codec decoder.

        Args:
            audio_code_frames: List of [37] integer code tensors (one per frame).
            chunk_size: Maximum frames per codec decoder forward pass.

        Returns:
            Float32 numpy array of audio samples, clipped to [-1, 1].
        """
        stacked_codes = torch.stack(audio_code_frames)           # [T, 37]
        stacked_codes = stacked_codes - NUM_SPECIAL_TOKENS       # remove special token offset
        stacked_codes = stacked_codes.to(device=self.device)
        total_frames = stacked_codes.shape[0]

        audio_chunks: list[torch.Tensor] = []
        for start in range(0, total_frames, chunk_size):
            chunk = stacked_codes[start : start + chunk_size]             # [chunk_T, 37]
            chunk = chunk.unsqueeze(0).permute(0, 2, 1)                   # [1, 37, chunk_T]
            ctx = (
                torch.autocast(device_type=self.device.type, dtype=self.dtype)
                if self.device.type in ("cuda", "xpu")
                else contextlib.nullcontext()
            )
            with ctx:
                decoded = self.codec_decoder.decode(chunk, dtype=self.dtype)
            audio_chunks.append(decoded.squeeze().cpu().float())

        audio = (
            torch.cat(audio_chunks).numpy()
            if len(audio_chunks) > 1
            else audio_chunks[0].numpy()
        )
        return np.clip(audio, -1.0, 1.0)
