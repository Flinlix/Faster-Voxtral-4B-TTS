"""Vector-quantization codebooks for the Voxtral codec (semantic VQ + acoustic FSQ)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SemanticCodebook(nn.Module):
    """Semantic codebook using exponential moving average (EMA) vector quantization."""

    def __init__(self, codebook_size: int, embedding_dim: int):
        super().__init__()
        self.epsilon = 1e-5
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, embedding_dim))
        self.register_buffer("_embedding", None, persistent=False)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            computed = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            self.register_buffer("_embedding", computed, persistent=False)
            return computed
        return self._embedding

    @property
    def num_codebooks(self) -> int:
        return 1

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode semantic codes [B, 1, T] → [B, embedding_dim, T]."""
        codes = codes.squeeze(1)
        quantized = F.embedding(codes, self.embedding.to(codes.device))
        return rearrange(quantized, "b t d -> b d t").to(dtype)


class AcousticCodebook(nn.Module):
    """Finite scalar quantization (FSQ) codebook for acoustic features."""

    def __init__(self, num_levels: int, num_codebooks: int):
        super().__init__()
        self.num_levels = num_levels
        self.num_codebooks = num_codebooks

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode acoustic codes [B, K, T] → [B, K, T] continuous values in [-1, 1]."""
        return ((codes * 2.0 / (self.num_levels - 1)) - 1).to(dtype)


class MistralAudioCodebook(nn.Module):
    """Combined semantic + acoustic codebook for Voxtral audio tokenization."""

    def __init__(
        self,
        semantic_codebook_size: int,
        semantic_dim: int,
        acoustic_codebook_size: int,
        acoustic_dim: int,
    ):
        super().__init__()
        self.semantic_codebook = SemanticCodebook(semantic_codebook_size, semantic_dim)
        self.acoustic_codebook = AcousticCodebook(acoustic_codebook_size, acoustic_dim)
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim

    @property
    def num_codebooks(self) -> int:
        return 1 + self.acoustic_codebook.num_codebooks

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode full codes [B, 1+K, T] → [B, semantic_dim+acoustic_dim, T]."""
        semantic_embeddings = self.semantic_codebook.decode(codes[:, :1, :], dtype)
        acoustic_embeddings = self.acoustic_codebook.decode(codes[:, 1:, :], dtype)
        return torch.cat([semantic_embeddings, acoustic_embeddings], dim=1)
