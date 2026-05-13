from typing import List, Optional

import torch
import torch.nn as nn

from utils.masking import random_masking

from .patch_embed import build_patch_embed
from .rope import RopeBlock, RopeCache
from .transformer import Block


class SeisFoundation(nn.Module):
    """
    Unified ViT-style foundation model for seismic data.

    Switches via config:
        input_type:     '1d' | 'multi_1d' | '2d'
        use_mae:        true  -> Masked Autoencoder
                        false -> Plain autoencoder
        pos_embed_type: 'rope'        -> rotary position embeddings (default)
                        'sinusoidal'  -> fixed sin/cos absolute embeddings (Vaswani et al.;
                                         factorized 2D for `multi_1d` and `2d`)
                        'learnable'   -> learnable absolute embeddings (MAE-paper style)

    `multi_1d` is designed for angle gathers: input (B, A, T) where A indexes
    reflection angle and T indexes time.  When RoPE is on, axis 0 of the 2D
    factorized rotation is the angle axis and axis 1 is the time axis -- the
    model sees relative differences in both, which matches the physics.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.input_type = cfg["input_type"]
        self.use_mae = bool(cfg.get("use_mae", True))
        self.mask_ratio = float(cfg.get("mask_ratio", 0.75))
        self.norm_pix_loss = bool(cfg.get("norm_pix_loss", False))
        self.pos_embed_type = cfg.get("pos_embed_type", "rope")
        assert self.pos_embed_type in ("rope", "sinusoidal", "learnable"), self.pos_embed_type

        embed_dim = cfg["embed_dim"]
        depth = cfg["depth"]
        num_heads = cfg["num_heads"]
        mlp_ratio = float(cfg.get("mlp_ratio", 4.0))

        decoder_embed_dim = cfg.get("decoder_embed_dim", embed_dim // 2)
        decoder_depth = cfg.get("decoder_depth", max(2, depth // 4))
        decoder_num_heads = cfg.get("decoder_num_heads", max(1, num_heads // 2))
        rope_base = float(cfg.get("rope_base", 10000.0))

        # Patch embedding (modality-specific)
        self.patch_embed = build_patch_embed(cfg)
        num_patches = self.patch_embed.num_patches
        patch_dim = self.patch_embed.patch_dim
        self.num_patches = num_patches

        # Position bookkeeping -- per-axis original-grid indices for each patch token
        self._register_axis_positions()

        # CLS / mask tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Encoder backbone
        if self.pos_embed_type == "rope":
            enc_head_dim = embed_dim // num_heads
            self.encoder_rope = RopeCache(enc_head_dim, self.num_axes, base=rope_base)
            self.encoder_blocks = nn.ModuleList(
                [RopeBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
            )
        else:  # 'learnable' or 'sinusoidal' -- both add a (1, N+1, D) embedding before the blocks
            if self.pos_embed_type == "learnable":
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            else:  # sinusoidal -- precomputed, stored as a buffer (not learnable)
                self.register_buffer("pos_embed", self._build_sincos_pos_embed(embed_dim))
            self.encoder_blocks = nn.ModuleList(
                [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
            )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if self.pos_embed_type == "rope":
            dec_head_dim = decoder_embed_dim // decoder_num_heads
            self.decoder_rope = RopeCache(dec_head_dim, self.num_axes, base=rope_base)
            self.decoder_blocks = nn.ModuleList(
                [RopeBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)]
            )
        else:
            if self.pos_embed_type == "learnable":
                self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches + 1, decoder_embed_dim)
                )
            else:  # sinusoidal
                self.register_buffer(
                    "decoder_pos_embed", self._build_sincos_pos_embed(decoder_embed_dim)
                )
            self.decoder_blocks = nn.ModuleList(
                [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)]
            )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)

        self._init_weights()

    # ---- position bookkeeping ----

    def _register_axis_positions(self):
        """Register per-token original-grid coordinates as buffers."""
        if self.input_type == "1d":
            self.num_axes = 1
            self.register_buffer("pos_axis_0", torch.arange(self.patch_embed.num_patches))
        elif self.input_type == "multi_1d":
            self.num_axes = 2
            nt = self.patch_embed.num_traces       # angles (for angle gathers) or traces
            tp = self.patch_embed.time_patches
            angle_idx = torch.arange(nt).repeat_interleave(tp)   # token k -> angle = k // tp
            time_idx = torch.arange(tp).repeat(nt)               # token k -> time  = k %  tp
            self.register_buffer("pos_axis_0", angle_idx)        # angle axis
            self.register_buffer("pos_axis_1", time_idx)         # time axis
        elif self.input_type == "2d":
            self.num_axes = 2
            gh = self.patch_embed.grid_h
            gw = self.patch_embed.grid_w
            row_idx = torch.arange(gh).repeat_interleave(gw)
            col_idx = torch.arange(gw).repeat(gh)
            self.register_buffer("pos_axis_0", row_idx)
            self.register_buffer("pos_axis_1", col_idx)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type!r}")

    def _axes_buffers(self) -> List[torch.Tensor]:
        if self.num_axes == 1:
            return [self.pos_axis_0]
        return [self.pos_axis_0, self.pos_axis_1]

    def _full_axes_positions(self, B: int) -> List[torch.Tensor]:
        """All N patch positions, broadcast to batch.  Returns list of (B, N)."""
        return [p.unsqueeze(0).expand(B, -1) for p in self._axes_buffers()]

    def _kept_axes_positions(self, ids_keep: torch.Tensor) -> List[torch.Tensor]:
        """Gather positions for kept tokens.  Returns list of (B, len_keep)."""
        B = ids_keep.shape[0]
        out = []
        for buf in self._axes_buffers():
            pos = buf.unsqueeze(0).expand(B, -1)
            out.append(torch.gather(pos, 1, ids_keep))
        return out

    def _prepend_cls_positions(self, axes_positions: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepend a position-0 slot for the CLS token (rotation is identity at pos 0)."""
        B = axes_positions[0].shape[0]
        cls_pos = torch.zeros(
            B, 1, dtype=axes_positions[0].dtype, device=axes_positions[0].device
        )
        return [torch.cat([cls_pos, p], dim=1) for p in axes_positions]

    # ---- sinusoidal position embedding (precomputed once at __init__) ----

    @staticmethod
    def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
        """Standard sinusoidal encoding: dim must be even; pos: (N,) -> (N, dim)."""
        assert dim % 2 == 0, "sincos dim must be even"
        half = dim // 2
        omega = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))
        angles = pos.float().unsqueeze(-1) * omega.unsqueeze(0)     # (N, half)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)      # (N, dim)

    def _build_sincos_pos_embed(self, dim: int) -> torch.Tensor:
        """Build fixed sinusoidal positional embedding for the current modality.

        1 axis  -> standard 1D sincos.
        2 axes  -> factorized: first dim/2 from axis 0, second dim/2 from axis 1
                   (each half itself sin/cos-split).  Requires dim % 4 == 0.
        Returns: (1, N + 1, dim) with a zero slot for CLS at index 0.
        """
        if self.num_axes == 1:
            pe = self._sincos_1d(dim, self.pos_axis_0)              # (N, dim)
        else:
            assert dim % 4 == 0, (
                f"sinusoidal 2-axis encoding needs dim % 4 == 0, got dim={dim}"
            )
            half = dim // 2
            pe0 = self._sincos_1d(half, self.pos_axis_0)            # (N, dim/2)
            pe1 = self._sincos_1d(half, self.pos_axis_1)            # (N, dim/2)
            pe = torch.cat([pe0, pe1], dim=-1)                      # (N, dim)
        cls_slot = torch.zeros(1, dim, dtype=pe.dtype)
        return torch.cat([cls_slot, pe], dim=0).unsqueeze(0)        # (1, N+1, dim)

    # ---- weight init ----

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Only initialize the learnable variant -- sinusoidal embeddings are
        # already populated by _build_sincos_pos_embed and stored as buffers.
        if isinstance(getattr(self, "pos_embed", None), nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if isinstance(getattr(self, "decoder_pos_embed", None), nn.Parameter):
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---- patchify / unpatchify (delegate to per-modality patch_embed) ----

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed.patchify(x)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Inverse of patchify: tokens (B, N, patch_dim) -> input shape."""
        return self.patch_embed.unpatchify(patches)

    # ---- parameter views ----

    _ENCODER_PREFIXES = (
        "patch_embed.", "cls_token", "pos_embed", "encoder_blocks.", "encoder_norm.",
        "pos_axis_0", "pos_axis_1", "encoder_rope.",
    )

    def encoder_state_dict(self) -> dict:
        """State dict containing only the encoder (transferable artifact).

        Includes:
            patch_embed, cls_token, pos_embed (if learnable), encoder_blocks,
            encoder_norm, pos_axis_* buffers, encoder_rope freqs (if RoPE).
        """
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith(self._ENCODER_PREFIXES)
        }

    # ---- forward ----

    def _run_encoder_blocks(self, x: torch.Tensor, axes_positions: Optional[List[torch.Tensor]]):
        if self.pos_embed_type == "rope":
            cos, sin = self.encoder_rope.build(axes_positions)
            for blk in self.encoder_blocks:
                x = blk(x, cos, sin)
        else:
            for blk in self.encoder_blocks:
                x = blk(x)
        return x

    def _run_decoder_blocks(self, x: torch.Tensor, axes_positions: Optional[List[torch.Tensor]]):
        if self.pos_embed_type == "rope":
            cos, sin = self.decoder_rope.build(axes_positions)
            for blk in self.decoder_blocks:
                x = blk(x, cos, sin)
        else:
            for blk in self.decoder_blocks:
                x = blk(x)
        return x

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        B = x.shape[0]
        x = self.patch_embed(x)                       # (B, N, D)

        additive = self.pos_embed_type in ("learnable", "sinusoidal")
        if additive:
            x = x + self.pos_embed[:, 1:, :]

        if mask_ratio > 0:
            x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)
            kept_axes = self._kept_axes_positions(ids_keep)
        else:
            mask = None
            ids_restore = None
            kept_axes = self._full_axes_positions(B)

        if additive:
            cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        else:
            cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        axes_with_cls = self._prepend_cls_positions(kept_axes) if self.pos_embed_type == "rope" else None
        x = self._run_encoder_blocks(x, axes_with_cls)
        return self.encoder_norm(x), mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.decoder_embed(x)
        B = x.shape[0]

        if ids_restore is not None:
            num_keep = x.shape[1] - 1
            num_total = ids_restore.shape[1]
            D = x.shape[2]
            mask_tokens = self.mask_token.expand(B, num_total - num_keep, -1)
            body = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
            body = torch.gather(body, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
            x = torch.cat([x[:, :1, :], body], dim=1)

        # After restoration the sequence is in canonical order.
        if self.pos_embed_type in ("learnable", "sinusoidal"):
            x = x + self.decoder_pos_embed
            axes_with_cls = None
        else:  # rope
            full = self._full_axes_positions(B)
            axes_with_cls = self._prepend_cls_positions(full)

        x = self._run_decoder_blocks(x, axes_with_cls)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]                            # drop cls

    def forward_loss(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = ((pred - target) ** 2).mean(dim=-1)    # (B, N)
        if mask is not None:
            denom = mask.sum().clamp(min=1.0)
            return (loss * mask).sum() / denom
        return loss.mean()

    def forward(self, x: torch.Tensor) -> dict:
        target = self.patchify(x)
        mask_ratio = self.mask_ratio if self.use_mae else 0.0
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(target, pred, mask)
        return {"loss": loss, "pred": pred, "target": target, "mask": mask}

    # ---- inference helpers (no masking by default) ----

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> dict:
        was_training = self.training
        self.eval()
        latent, _, _ = self.forward_encoder(x, mask_ratio=0.0)
        if was_training:
            self.train()
        return {"latent": latent, "cls": latent[:, 0], "tokens": latent[:, 1:]}

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, mask_ratio: float = 0.0) -> dict:
        was_training = self.training
        self.eval()
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio=mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        recon = self.unpatchify(pred)
        if was_training:
            self.train()
        return {
            "latent": latent,
            "cls": latent[:, 0],
            "pred_patches": pred,
            "recon": recon,
            "mask": mask,
        }
