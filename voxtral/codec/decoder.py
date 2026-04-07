"""Voxtral codec decoder - converts discrete audio codes to 24 kHz waveform."""

import torch
import torch.nn as nn
from einops import rearrange

from voxtral.config import CodecConfig
from voxtral.codec.convolutions import CausalConv1d, CausalConvTranspose1d
from voxtral.codec.attention import CodecTransformer
from voxtral.codec.quantizer import MistralAudioCodebook


class CodecDecoder(nn.Module):
    """Converts quantized latent codes [B, 37, T] into a 24 kHz audio waveform."""

    def __init__(self, config: CodecConfig | None = None):
        super().__init__()
        if config is None:
            config = CodecConfig.voxtral_4b()

        self.patch_size = config.patch_size
        self.quantizer = MistralAudioCodebook(
            config.semantic_codebook_size, config.semantic_dim,
            config.acoustic_codebook_size, config.num_acoustic_codebooks,
        )

        blocks: list[nn.Module] = []

        # Initial projection: latent_dim → model_dim
        blocks.append(CausalConv1d(
            config.latent_dim, config.model_dim,
            kernel_size=config.decoder_kernel_sizes[0],
            stride=config.decoder_strides[0],
            pad_mode="replicate", use_bias=False,
        ))

        current_window_size = 2  # after encoder downsampling (16 / 2^3)
        for stage_index, num_layers in enumerate(config.decoder_num_layers):
            transformer_kwargs = dict(
                model_dim=config.model_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                norm_eps=config.norm_eps,
                qk_norm_eps=config.qk_norm_eps,
                layer_scale_init=config.layer_scale_init,
                sliding_window=current_window_size,
                causal=True,
                qk_norm=True,
                use_biases=False,
                layer_scale=True,
            )
            blocks.append(CodecTransformer(num_layers=num_layers, **transformer_kwargs))

            if stage_index + 1 < len(config.decoder_num_layers):
                next_kernel = config.decoder_kernel_sizes[stage_index + 1]
                next_stride = config.decoder_strides[stage_index + 1]
                blocks.append(CausalConvTranspose1d(
                    config.model_dim, config.model_dim,
                    kernel_size=next_kernel, stride=next_stride, use_bias=False,
                ))
                if next_stride > 1:
                    current_window_size *= 2

        self.decoder_blocks = nn.ModuleList(blocks)
        self.output_proj = CausalConv1d(config.model_dim, config.patch_size, kernel_size=7, use_bias=False)

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode integer codes to waveform.

        Args:
            codes: Integer tensor [B, num_codebooks=37, T_frames].
            dtype: Computation dtype.

        Returns:
            Audio waveform [B, 1, T_samples].
        """
        latent_embeddings = self.quantizer.decode(codes, dtype)  # [B, latent_dim, T]
        latent_embeddings = rearrange(latent_embeddings, "b d t -> b t d").contiguous()

        for block in self.decoder_blocks:
            if isinstance(block, (CausalConv1d, CausalConvTranspose1d)):
                latent_embeddings = rearrange(latent_embeddings, "b t d -> b d t")
                latent_embeddings = block(latent_embeddings)
                latent_embeddings = rearrange(latent_embeddings, "b d t -> b t d")
            else:
                latent_embeddings = block(latent_embeddings)

        latent_embeddings = rearrange(latent_embeddings, "b t d -> b d t")
        latent_embeddings = self.output_proj(latent_embeddings)
        return rearrange(latent_embeddings, "b (c h) t -> b c (t h)", h=self.patch_size)
