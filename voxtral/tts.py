"""VoxtralTTS - the top-level orchestrator for text-to-speech synthesis.

Loads all model components (LLM, acoustic transformer, codec decoder,
audio token embedding), performs weight loading + optional quantization,
and exposes a ``generate()`` method for inference.
"""

import contextlib
import logging
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.args import TransformerArgs
from mistral_inference.cache import BufferCache
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
from voxtral.weights import load_checkpoint_weights

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

    Pipeline: Mistral-3B LLM → FlowMatching Acoustic Transformer → Voxtral Codec Decoder

    Args:
        config: Model configuration (defaults to Voxtral-4B).
        device: Torch device string.
        dtype: Floating-point dtype for model weights.
        quantize: LLM quantization mode - ``"nf4"`` (default),
            ``"int8"``, or ``None`` for full bf16.
    """

    def __init__(
        self,
        config: VoxtralConfig | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quantize: str | None = "nf4",
    ):
        if config is None:
            config = VoxtralConfig.voxtral_4b()
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype

        logger.info("Loading tokenizer ...")
        self.tokenizer = MistralTokenizer.from_hf_hub(config.repo_id)

        logger.info("Building modules ...")
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
        self.llm = Transformer(self._llm_args, pipeline_rank=0, num_pipeline_ranks=1)
        self.acoustic_transformer = FlowMatchingAudioTransformer(config.acoustic)
        self.codec_decoder = CodecDecoder(config.codec)
        self.audio_token_embedding = AudioTokenEmbedding(
            embedding_dim=config.llm.model_dim,
            semantic_codebook_size=config.codec.semantic_codebook_size,
            acoustic_codebook_size=config.codec.acoustic_codebook_size,
            num_acoustic_codebooks=config.codec.num_acoustic_codebooks,
        )

        logger.info("Loading weights ...")
        load_checkpoint_weights(
            self.llm, self.acoustic_transformer,
            self.codec_decoder, self.audio_token_embedding,
            config.repo_id,
        )

        self._apply_quantization_and_move_to_device(quantize)

        self.voice_embeddings: dict[str, torch.Tensor] = {}
        self._load_voice_embeddings()

        self._audio_placeholder_token_id = (
            self.tokenizer.instruct_tokenizer.audio_encoder.special_ids.audio
        )

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
        """Download and cache all available voice embedding presets."""
        for voice_name in VOICE_PRESETS:
            try:
                path = hf_hub_download(self.config.repo_id, f"voice_embedding/{voice_name}.pt")
                self.voice_embeddings[voice_name] = torch.load(
                    path, map_location="cpu", weights_only=True,
                ).to(device=self.device, dtype=self.dtype)
            except Exception as exc:
                logger.warning("Failed to load voice '%s': %s", voice_name, exc)
        if not self.voice_embeddings:
            raise RuntimeError(
                "No voice embeddings loaded - cannot serve TTS requests. "
                "Check network connectivity and the HuggingFace repo."
            )
        logger.info("Loaded %d voice embeddings: %s",
                    len(self.voice_embeddings), list(self.voice_embeddings.keys()))

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

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        voice: str = "neutral_female",
        max_frames: int = 2000,
        stream_callback: Callable[[np.ndarray], None] | None = None,
        stream_after_n_frames: int = 1,
        stream_interval_frames: int = 1,
        verbose: bool = True,
    ) -> np.ndarray:
        """Generate audio waveform from text.

        Args:
            text: Input text to synthesise.
            voice: Voice preset name.
            max_frames: Maximum audio frames (at 12.5 Hz → 80 ms each).
            stream_callback: Optional ``callback(audio_chunk_np)`` called with each
                decoded audio chunk during generation.
            stream_after_n_frames: Frames to generate before the first stream emission.
            stream_interval_frames: Frames between subsequent decode-and-emit cycles.
            verbose: Whether to print generation statistics.

        Returns:
            Float32 numpy array of audio samples at 24 kHz.
        """
        # 0. Validate voice
        if voice not in self.voice_embeddings:
            available = ", ".join(sorted(self.voice_embeddings.keys()))
            raise ValueError(f"Unknown voice '{voice}'. Available: {available}")

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
        cache = BufferCache(
            n_layers=self._llm_args.n_layers,
            max_batch_size=1,
            max_seq_len=input_ids.shape[0] + max_frames + 16,
            n_kv_heads=self._llm_args.n_kv_heads,
            head_dim=self._llm_args.head_dim,
        )
        cache.to(device=self.device, dtype=self.dtype)
        cache.reset()

        prefill_length = input_ids.shape[0]
        generation_start_time = time.perf_counter()

        hidden_state = self._run_llm_with_precomputed_embeddings(
            input_embeddings, sequence_lengths=[prefill_length], cache=cache,
        )
        hidden_state = hidden_state[-1:]  # last token → [1, D]

        # 4. Autoregressive generation
        generated_audio_codes: list[torch.Tensor] = []
        num_streamed_frames = 0
        next_stream_frame = stream_after_n_frames if stream_callback else float("inf")

        for _ in range(max_frames):
            audio_codes = self.acoustic_transformer(hidden_state)  # [1, 37]

            if audio_codes[0, 0].item() == END_AUDIO_ID:
                break

            generated_audio_codes.append(audio_codes[0].cpu())

            # Streaming: decode new codes with bounded context window
            if stream_callback and len(generated_audio_codes) >= next_stream_frame:
                context_start = max(0, num_streamed_frames - CODEC_CONTEXT_FRAMES)
                chunk_audio = self._decode_audio_codes_to_waveform(generated_audio_codes[context_start:])
                skip_samples = (num_streamed_frames - context_start) * SAMPLES_PER_FRAME
                new_audio = chunk_audio[skip_samples:]
                if len(new_audio) > 0:
                    stream_callback(new_audio)
                num_streamed_frames = len(generated_audio_codes)
                next_stream_frame = len(generated_audio_codes) + stream_interval_frames

            # Convert codes to embedding for next LLM step
            codes_for_embedding = audio_codes.unsqueeze(-1)  # [1, 37, 1]
            next_token_embedding = self.audio_token_embedding(codes_for_embedding)  # [1, 1, D]
            next_token_embedding = next_token_embedding.squeeze(0)  # [1, D]

            hidden_state = self._run_llm_with_precomputed_embeddings(
                next_token_embedding, sequence_lengths=[1], cache=cache,
            )

        generation_time = time.perf_counter() - generation_start_time

        if not generated_audio_codes:
            raise RuntimeError("No audio frames generated - the model produced no output for this input")

        # 5. Final decode of all generated codes
        audio = self._decode_audio_codes_to_waveform(generated_audio_codes)

        # Emit any remaining audio not yet streamed
        if stream_callback and num_streamed_frames < len(generated_audio_codes):
            context_start = max(0, num_streamed_frames - CODEC_CONTEXT_FRAMES)
            chunk_audio = self._decode_audio_codes_to_waveform(generated_audio_codes[context_start:])
            skip_samples = (num_streamed_frames - context_start) * SAMPLES_PER_FRAME
            new_audio = chunk_audio[skip_samples:]
            if len(new_audio) > 0:
                stream_callback(new_audio)

        num_frames = len(generated_audio_codes)
        duration_seconds = len(audio) / SAMPLE_RATE
        if verbose:
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
