"""Shared neural network primitives used across codec and acoustic modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU gated feed-forward network (Shazeer 2020).

    Attribute names (w1, w2, w3) match the checkpoint key convention.
    """

    def __init__(self, model_dim: int, hidden_dim: int, use_biases: bool = False):
        super().__init__()
        self.w1 = nn.Linear(model_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, model_dim, bias=use_biases)
        self.w3 = nn.Linear(model_dim, hidden_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(features)) * self.w3(features))
