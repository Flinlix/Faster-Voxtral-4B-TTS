"""Typed configuration dataclasses for all Voxtral TTS components.

Each config provides a ``voxtral_4b()`` factory that returns the default
hyperparameters matching the Voxtral-4B-TTS-2603 checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Global constants ────────────────────────────────────────────────────

REPO_ID = "mistralai/Voxtral-4B-TTS-2603"
SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 1_920  # codec downsample factor: patch_size (240) × prod(strides) (8)
CODEC_CONTEXT_FRAMES = 20  # left-context frames for incremental codec streaming

EMPTY_AUDIO_ID = 0
END_AUDIO_ID = 1
NUM_SPECIAL_TOKENS = 2


# ── Component configs ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CodecConfig:
    semantic_dim: int = 256
    acoustic_dim: int = 36
    model_dim: int = 1024
    hidden_dim: int = 4096
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int = 128
    patch_size: int = 240
    norm_eps: float = 0.01
    qk_norm_eps: float = 1e-6
    layer_scale_init: float = 0.01

    decoder_num_layers: tuple[int, ...] = (2, 2, 2, 2)
    decoder_kernel_sizes: tuple[int, ...] = (3, 4, 4, 4)
    decoder_strides: tuple[int, ...] = (1, 2, 2, 2)

    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    num_acoustic_codebooks: int = 36

    @classmethod
    def voxtral_4b(cls) -> CodecConfig:
        return cls()

    @property
    def latent_dim(self) -> int:
        return self.semantic_dim + self.acoustic_dim


@dataclass(frozen=True)
class AcousticTransformerConfig:
    model_dim: int = 3072
    num_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 9216
    num_heads: int = 32
    num_kv_heads: int = 8
    use_biases: bool = False
    norm_eps: float = 1e-5
    input_dim: int = 3072  # LLM hidden size

    semantic_codebook_size: int = 8192
    acoustic_levels: int = 21
    num_acoustic_codes: int = 36

    num_ode_steps: int = 8
    classifier_free_guidance_scale: float = 1.2
    initial_noise_scale: float = 1.0

    @classmethod
    def voxtral_4b(cls) -> AcousticTransformerConfig:
        return cls()


@dataclass(frozen=True)
class LLMConfig:
    model_dim: int = 3072
    num_layers: int = 26
    head_dim: int = 128
    hidden_dim: int = 9216
    num_heads: int = 32
    num_kv_heads: int = 8
    norm_eps: float = 1e-5
    vocab_size: int = 131_072
    rope_theta: float = 1_000_000.0

    @classmethod
    def voxtral_4b(cls) -> LLMConfig:
        return cls()


@dataclass(frozen=True)
class VoxtralConfig:
    repo_id: str = REPO_ID
    sample_rate: int = SAMPLE_RATE
    samples_per_frame: int = SAMPLES_PER_FRAME
    codec_context_frames: int = CODEC_CONTEXT_FRAMES

    llm: LLMConfig = field(default_factory=LLMConfig.voxtral_4b)
    acoustic: AcousticTransformerConfig = field(default_factory=AcousticTransformerConfig.voxtral_4b)
    codec: CodecConfig = field(default_factory=CodecConfig.voxtral_4b)

    @classmethod
    def voxtral_4b(cls) -> VoxtralConfig:
        return cls()
