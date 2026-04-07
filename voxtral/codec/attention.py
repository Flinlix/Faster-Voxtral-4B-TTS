"""ALiBi sliding-window causal attention and transformer blocks for the Voxtral codec."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from voxtral.nn import SwiGLU


class CodecAttention(nn.Module):
    """ALiBi + sliding-window causal attention for the codec decoder.

    Attribute names (wq, wk, wv, wo, q_norm, k_norm) match checkpoint keys.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int,
        causal: bool = True,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_biases: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_repeats = num_heads // num_kv_heads
        self.causal = causal
        self.sliding_window = sliding_window

        self.register_buffer(
            "alibi_slopes", self._compute_alibi_slopes(num_heads), persistent=False,
        )

        self.wq = nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(model_dim, num_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(model_dim, num_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, model_dim, bias=use_biases)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(num_heads * head_dim, eps=qk_norm_eps)
            self.k_norm = nn.RMSNorm(num_kv_heads * head_dim, eps=qk_norm_eps)

    @staticmethod
    def _compute_alibi_slopes(num_heads: int) -> torch.Tensor:
        def _power_of_2(n: int) -> torch.Tensor:
            ratio = 2.0 ** (-8.0 / n)
            return torch.tensor([ratio ** i for i in range(n)], dtype=torch.float32)
        if math.log2(num_heads).is_integer():
            return _power_of_2(num_heads)
        nearest_power = 2 ** math.floor(math.log2(num_heads))
        return torch.cat([
            _power_of_2(nearest_power),
            _power_of_2(2 * nearest_power)[::2][: num_heads - nearest_power],
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = 1 if features.dim() == 2 else features.shape[0]
        seq_len = features.shape[-2]

        query, key, value = self.wq(features), self.wk(features), self.wv(features)
        if self.qk_norm:
            query, key = self.q_norm(query), self.k_norm(key)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Expand KV heads for grouped-query attention
        if self.kv_repeats > 1:
            key = key.unsqueeze(3).expand(-1, -1, -1, self.kv_repeats, -1).flatten(2, 3)
            value = value.unsqueeze(3).expand(-1, -1, -1, self.kv_repeats, -1).flatten(2, 3)

        # [batch, heads, seq, dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Build ALiBi + causal + sliding-window attention bias
        positions = torch.arange(seq_len, device=features.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # [S, S]
        slopes = self.alibi_slopes.to(dtype=features.dtype, device=features.device)
        attention_bias = (
            slopes.view(self.num_heads, 1, 1)
            * relative_positions.unsqueeze(0).to(features.dtype)
        )
        if self.causal:
            attention_bias = attention_bias.masked_fill(
                relative_positions.unsqueeze(0) > 0, float("-inf"),
            )
        window_left = self.sliding_window
        window_right = 0 if self.causal else self.sliding_window
        outside_window = (relative_positions < -window_left) | (relative_positions > window_right)
        attention_bias = attention_bias.masked_fill(outside_window.unsqueeze(0), float("-inf"))

        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_bias.unsqueeze(0),
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        return output.squeeze(0) if features.dim() == 2 else output


class CodecTransformerBlock(nn.Module):
    """Pre-norm transformer block with optional layer scaling for the codec."""

    def __init__(
        self,
        layer_id: int,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int,
        causal: bool = True,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_biases: bool = False,
        norm_eps: float = 0.01,
        layer_scale: bool = True,
        layer_scale_init: float = 0.01,
    ):
        super().__init__()
        self._layer_id = layer_id
        self.attention = CodecAttention(
            model_dim, num_heads, num_kv_heads, head_dim, sliding_window,
            causal=causal, qk_norm=qk_norm, qk_norm_eps=qk_norm_eps, use_biases=use_biases,
        )
        self.feed_forward = SwiGLU(model_dim, hidden_dim, use_biases)
        self.attention_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.layer_scale = layer_scale
        if layer_scale:
            self.attention_scale = nn.Parameter(torch.full((model_dim,), layer_scale_init))
            self.ffn_scale = nn.Parameter(torch.full((model_dim,), layer_scale_init))

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.attention(self.attention_norm(features))
        if self.layer_scale:
            residual = self.attention_scale * residual
        hidden_state = features + residual
        residual = self.feed_forward(self.ffn_norm(hidden_state))
        if self.layer_scale:
            residual = self.ffn_scale * residual
        return hidden_state + residual


class CodecTransformer(nn.Module):
    """Stack of ``CodecTransformerBlock`` layers."""

    def __init__(self, num_layers: int, **block_kwargs):
        super().__init__()
        self.layers_ids = list(range(num_layers))
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            self.layers[str(layer_id)] = CodecTransformerBlock(layer_id=layer_id, **block_kwargs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer_id in self.layers_ids:
            features = self.layers[str(layer_id)](features)
        return features
