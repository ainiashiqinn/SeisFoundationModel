from typing import Tuple, Union

import torch
import torch.nn as nn


class PatchEmbed1D(nn.Module):
    """Single 1D seismic trace.  Input: (B, T) or (B, 1, T)."""

    def __init__(self, trace_length: int, patch_size: int, embed_dim: int, in_channels: int = 1):
        super().__init__()
        assert trace_length % patch_size == 0, "trace_length must be divisible by patch_size"
        self.trace_length = trace_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = trace_length // patch_size
        self.patch_dim = in_channels * patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)        # embedding layer: (B, C, T) -> (B, D, N) where N = T / patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.proj(x)              # (B, D, N)
        return x.transpose(1, 2)      # (B, N, D)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        x = x.reshape(B, C, self.num_patches, self.patch_size)                              # (B, C, T) -> (B, C, N, P)
        return x.permute(0, 2, 1, 3).reshape(B, self.num_patches, C * self.patch_size)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C*P) -> (B, C, T)
        B, N, _ = x.shape
        x = x.reshape(B, N, self.in_channels, self.patch_size)
        return x.permute(0, 2, 1, 3).reshape(B, self.in_channels, N * self.patch_size)


class PatchEmbedMulti1D(nn.Module):
    """Multi-trace 1D, designed for angle gathers: same subsurface location,
    different reflection angles.  Input: (B, A, T) where A indexes angle (or
    trace) and T indexes time.  Produces one token per (angle, time-patch).

    `num_traces` is the angle-axis size A.  When RoPE is on, the angle axis
    is rotated separately from the time axis, so the model sees relative
    angle differences naturally."""

    def __init__(self, num_traces: int, trace_length: int, patch_size: int, embed_dim: int):
        super().__init__()
        assert trace_length % patch_size == 0, "trace_length must be divisible by patch_size"
        self.num_traces = num_traces
        self.trace_length = trace_length
        self.patch_size = patch_size
        self.time_patches = trace_length // patch_size
        self.num_patches = num_traces * self.time_patches
        self.patch_dim = patch_size
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                       # (B, 1, C, T)
        x = self.proj(x)                         # (B, D, C, time_patches)
        return x.flatten(2).transpose(1, 2)      # (B, C * time_patches, D)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = x.reshape(B, C, self.time_patches, self.patch_size)
        return x.reshape(B, C * self.time_patches, self.patch_size)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C * time_patches, P) -> (B, C, T)
        B = x.shape[0]
        x = x.reshape(B, self.num_traces, self.time_patches, self.patch_size)
        return x.reshape(B, self.num_traces, self.time_patches * self.patch_size)


class PatchEmbed2D(nn.Module):
    """2D seismic patch (e.g. shot gather slice).  Input: (B, H, W) or (B, 1, H, W)."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        embed_dim: int,
        in_channels: int = 1,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_h = img_size[0] // patch_size[0]
        self.grid_w = img_size[1] // patch_size[1]
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = in_channels * patch_size[0] * patch_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.proj(x)                         # (B, D, gh, gw)
        return x.flatten(2).transpose(1, 2)      # (B, gh*gw, D)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        x = x.reshape(B, C, self.grid_h, ph, self.grid_w, pw)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, C * ph * pw)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C*ph*pw) -> (B, C, H, W)
        B = x.shape[0]
        ph, pw = self.patch_size
        x = x.reshape(B, self.grid_h, self.grid_w, self.in_channels, ph, pw)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.in_channels, self.grid_h * ph, self.grid_w * pw
        )


def build_patch_embed(cfg: dict) -> nn.Module:
    t = cfg["input_type"]
    embed_dim = cfg["embed_dim"]
    if t == "1d":
        return PatchEmbed1D(
            trace_length=cfg["trace_length"],
            patch_size=cfg["patch_size"],
            embed_dim=embed_dim,
            in_channels=cfg.get("in_channels", 1),
        )
    if t == "multi_1d":
        return PatchEmbedMulti1D(
            num_traces=cfg["num_traces"],
            trace_length=cfg["trace_length"],
            patch_size=cfg["patch_size"],
            embed_dim=embed_dim,
        )
    if t == "2d":
        return PatchEmbed2D(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            embed_dim=embed_dim,
            in_channels=cfg.get("in_channels", 1),
        )
    raise ValueError(f"Unknown input_type: {t!r}.  Expected one of: '1d', 'multi_1d', '2d'.")
