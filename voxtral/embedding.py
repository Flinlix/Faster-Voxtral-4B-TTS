"""Multi-codebook audio token embedding - maps 37-codebook audio codes to LLM input space."""

import torch
import torch.nn as nn

from voxtral.config import NUM_SPECIAL_TOKENS


class AudioTokenEmbedding(nn.Module):
    """Looks up 37-codebook audio codes and sums them into a single LLM embedding.

    Attribute name (embeddings) matches checkpoint key.
    """

    def __init__(
        self,
        embedding_dim: int = 3072,
        semantic_codebook_size: int = 8192,
        acoustic_codebook_size: int = 21,
        num_acoustic_codebooks: int = 36,
    ):
        super().__init__()
        semantic_vocab_size = semantic_codebook_size + NUM_SPECIAL_TOKENS
        acoustic_vocab_size = acoustic_codebook_size + NUM_SPECIAL_TOKENS

        codebook_sizes = [semantic_vocab_size] + [acoustic_vocab_size] * num_acoustic_codebooks
        offsets = [0] + codebook_sizes[:-1]
        self.register_buffer(
            "offsets",
            torch.tensor(offsets).cumsum(0).long(),
            persistent=False,
        )
        total_vocab = sum(codebook_sizes)
        padded_vocab = 128 * ((total_vocab + 127) // 128)
        self.embeddings = nn.Embedding(padded_vocab, embedding_dim)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum per-codebook embeddings into a single vector per timestep.

        Args:
            codes: [B, 37, L] integer audio codes.

        Returns:
            [B, L, embedding_dim] summed embedding across all codebooks.
        """
        offset_codes = codes + self.offsets[None, :, None].to(codes.device)
        codebook_embeddings = self.embeddings(offset_codes)  # [B, 37, L, D]
        return codebook_embeddings.sum(dim=1)                # [B, L, D]
