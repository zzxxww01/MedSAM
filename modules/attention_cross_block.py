"""Attention fusion block for query/support feature interaction."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionCrossBlock(nn.Module):
    """Fuse query feature maps with support feature maps via cross-attention."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden = int(embed_dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        # Keep the residual contribution stable in early training.
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor | None = None,
        support_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query_features: [B, C, H, W]
            support_features: [B, N, C, Hs, Ws] or [B, C, Hs, Ws] or None
            support_masks: reserved argument for future weighted masking.
        """
        del support_masks  # reserved for compatibility

        if query_features.dim() != 4:
            raise ValueError("query_features must be [B, C, H, W]")

        if support_features is None:
            support_features = query_features.unsqueeze(1)
        elif support_features.dim() == 4:
            support_features = support_features.unsqueeze(1)
        elif support_features.dim() != 5:
            raise ValueError("support_features must be [B, N, C, H, W] or [B, C, H, W]")

        bsz, channels, height, width = query_features.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"embed_dim mismatch: got C={channels}, expected {self.embed_dim}"
            )

        q_tokens = query_features.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
        kv_tokens = support_features.permute(0, 1, 3, 4, 2).reshape(bsz, -1, channels)

        q_norm = self.norm_q(q_tokens)
        kv_norm = self.norm_kv(kv_tokens)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm, need_weights=False)

        x = q_tokens + torch.tanh(self.gate) * attn_out
        x = x + self.ffn(self.norm_ffn(x))
        fused = x.reshape(bsz, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return fused
