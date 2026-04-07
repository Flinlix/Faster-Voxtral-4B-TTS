"""Bidirectional attention and transformer blocks for the acoustic flow-matching model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from voxtral.nn import SwiGLU


class BidirectionalAttention(nn.Module):
    """Standard bidirectional grouped-query attention (no ALiBi, no sliding window).

    Attribute names (wq, wk, wv, wo) match checkpoint keys.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_biases: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_repeats = num_heads // num_kv_heads

        self.wq = nn.Linear(model_dim, num_heads * head_dim, bias=use_biases)
        self.wk = nn.Linear(model_dim, num_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(model_dim, num_kv_heads * head_dim, bias=use_biases)
        self.wo = nn.Linear(num_heads * head_dim, model_dim, bias=use_biases)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = 1 if features.dim() == 2 else features.shape[0]
        seq_len = features.shape[-2]

        query = self.wq(features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.wk(features).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = self.wv(features).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Expand KV heads for grouped-query attention
        if self.kv_repeats > 1:
            key = key.unsqueeze(3).expand(-1, -1, -1, self.kv_repeats, -1).flatten(2, 3)
            value = value.unsqueeze(3).expand(-1, -1, -1, self.kv_repeats, -1).flatten(2, 3)

        # [batch, heads, seq, dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        return output.squeeze(0) if features.dim() == 2 else output


class AcousticTransformerBlock(nn.Module):
    """Pre-norm transformer block for the acoustic flow-matching model."""

    def __init__(
        self,
        layer_id: int,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_biases: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self._layer_id = layer_id
        self.attention = BidirectionalAttention(
            model_dim, num_heads, num_kv_heads, head_dim, use_biases,
        )
        self.feed_forward = SwiGLU(model_dim, hidden_dim, use_biases)
        self.attention_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(model_dim, eps=norm_eps)

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden_state = features + self.attention(self.attention_norm(features))
        return hidden_state + self.feed_forward(self.ffn_norm(hidden_state))
