"""Causal 1D convolution primitives for the Voxtral codec."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

weight_norm = nn.utils.parametrizations.weight_norm


def pad_1d(
    input_tensor: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """1D padding with reflect-mode support for inputs shorter than the pad size."""
    pad_left, pad_right = paddings
    if mode == "reflect":
        max_pad = max(pad_left, pad_right)
        extra_padding = 0
        if input_tensor.shape[-1] <= max_pad:
            extra_padding = max_pad - input_tensor.shape[-1] + 1
            input_tensor = F.pad(input_tensor, (0, extra_padding))
        padded = F.pad(input_tensor, paddings, mode, value)
        return padded[..., : padded.shape[-1] - extra_padding]
    return F.pad(input_tensor, paddings, mode, value)


class CausalConv1d(nn.Module):
    """Left-padded causal 1D convolution with optional weight normalisation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation, bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.pad_mode = pad_mode
        self._stride = self.conv.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.conv.dilation[0] + 1
        self._total_padding = self._effective_kernel_size - self._stride

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        num_frames = (
            (features.shape[-1] - self._effective_kernel_size + self._total_padding)
            / self._stride + 1
        )
        target_length = (
            (math.ceil(num_frames) - 1) * self._stride
            + (self._effective_kernel_size - self._total_padding)
        )
        extra_padding = target_length - features.shape[-1]
        features = pad_1d(features, (self._total_padding, extra_padding), mode=self.pad_mode)
        return self.conv(features)


class CausalConvTranspose1d(nn.Module):
    """Causal transposed 1D convolution with output trimming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        trim_ratio: float = 1.0,
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, groups=groups, bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.trim_ratio = trim_ratio

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        total_padding = kernel_size - stride
        output = self.conv(features)
        right_trim = math.ceil(total_padding * self.trim_ratio)
        left_trim = total_padding - right_trim
        return output[..., left_trim : output.shape[-1] - right_trim]
