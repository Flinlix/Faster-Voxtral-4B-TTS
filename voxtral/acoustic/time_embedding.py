"""Sinusoidal timestep embedding for the flow-matching ODE."""

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion/flow-matching timesteps."""

    def __init__(self, embedding_dim: int, theta: float = 10000.0):
        super().__init__()
        inverse_frequencies = torch.exp(
            -math.log(theta)
            * torch.arange(embedding_dim // 2).float()
            / (embedding_dim // 2)
        )
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=True)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Embed a timestep scalar/vector into a sinusoidal vector.

        Args:
            timestep: [B, 1] tensor of timestep values.

        Returns:
            [B, embedding_dim] sinusoidal embedding.
        """
        angles = torch.einsum("bi, j -> bj", timestep, self.inverse_frequencies)
        return torch.cat((angles.cos(), angles.sin()), dim=-1)
