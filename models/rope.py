"""
Rotary Position Embeddings (RoPE) -- rotate-half formulation.

Supports two modes:
  * 1 axis  -> full head_dim rotates with one position index (use for `1d`).
  * 2 axes  -> head_dim split in half; first half rotates with axis 0,
               second half with axis 1 (use for `multi_1d` and `2d`).

Constraints on head_dim:
  * 1 axis : head_dim % 2 == 0
  * 2 axes : head_dim % 4 == 0   (each axis subspace needs even size)
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def _rope_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    return 1.0 / (base ** (torch.arange(half).float() / half))


class RopeCache(nn.Module):
    """Builds (cos, sin) tensors for rotate-half RoPE from per-axis position tensors."""

    def __init__(self, head_dim: int, num_axes: int, base: float = 10000.0):
        super().__init__()
        assert num_axes in (1, 2), "RopeCache supports 1 or 2 axes"
        if num_axes == 1:
            assert head_dim % 2 == 0, "head_dim must be divisible by 2 for 1-axis RoPE"
            axis_dims = [head_dim]
        else:
            assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2-axis RoPE"
            axis_dims = [head_dim // 2, head_dim // 2]
        self.head_dim = head_dim
        self.num_axes = num_axes
        for i, d in enumerate(axis_dims):
            self.register_buffer(f"freqs_{i}", _rope_freqs(d, base), persistent=False)

    def build(self, axes_positions: List[torch.Tensor]):
        """axes_positions: list of (B, N) tensors, one per axis.
        Returns cos, sin: each (B, N, head_dim)."""
        assert len(axes_positions) == self.num_axes
        cos_parts, sin_parts = [], []
        for i, pos in enumerate(axes_positions):
            freq = getattr(self, f"freqs_{i}").to(pos.device)
            angles = pos.unsqueeze(-1).float() * freq           # (B, N, axis_dim/2)
            c, s = angles.cos(), angles.sin()
            cos_parts.append(torch.cat([c, c], dim=-1))         # (B, N, axis_dim)
            sin_parts.append(torch.cat([s, s], dim=-1))
        return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


class RopeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3)      # applies linear projection to get q, k, v by applying matmul with W and b
        self.proj = nn.Linear(dim, dim)         # applies linear projection to mix the heads back together (768, 256)
        self.proj_drop = nn.Dropout(dropout)    # applies dropout to the output of the projection layer (only qkv, not the attention weights)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                          # each (B, H, N, head_dim)
        cos_b = cos.unsqueeze(1)                         # (B, 1, N, head_dim) -> broadcast across heads
        sin_b = sin.unsqueeze(1)
        q = apply_rope(q, cos_b, sin_b)
        k = apply_rope(k, cos_b, sin_b)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )                                                # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class RopeBlock(nn.Module):
    """Pre-norm Transformer block with RoPE attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RopeAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x
