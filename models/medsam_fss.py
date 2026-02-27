"""MedSAM variant with optional attention-based feature fusion."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import AttentionCrossBlock


class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha

        self.linear = linear_layer
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def apply_lora_to_image_encoder(image_encoder: nn.Module, rank: int = 4):
    """Recursively replaces qkv projections in the ViT with LoRA-injected linear layers."""
    for name, module in image_encoder.named_modules():
        if name.endswith("attn"):
            # Inside segment_anything's Attention block, qkv is a single Linear layer.
            # We apply LoRA to the entire qkv projection.
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                module.qkv = LoRALinear(module.qkv, rank=rank)


class LocalGlobalAdapter(nn.Module):
    """
    A lightweight multi-scale bottleneck adapter inserted after the ViT image encoder.
    Captures high-frequency local details (edges) to complement SAM's global features.
    """

    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )
        self.conv2 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=2, dilation=2, groups=embed_dim
        )
        self.pointwise = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W]
        feat1 = self.act(self.conv1(x))
        feat2 = self.act(self.conv2(x))
        fused = torch.cat([feat1, feat2], dim=1)
        out = self.pointwise(fused)
        return x + out  # residual connection


class MedSAMFSS(nn.Module):
    """MedSAM with optional support-aware attention fusion."""

    def __init__(
        self,
        image_encoder: nn.Module,
        mask_decoder: nn.Module,
        prompt_encoder: nn.Module,
        use_support_attention: bool = False,
        attention_embed_dim: int = 256,
        attention_num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.use_support_attention = use_support_attention

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        if use_support_attention:
            self.attention_fusion = AttentionCrossBlock(
                embed_dim=attention_embed_dim,
                num_heads=attention_num_heads,
            )
        else:
            self.attention_fusion = None

        # Optional Local-Global Adapter
        # Activated externally if the training script requests it.
        self.local_global_adapter = None

    def _encode_support(self, support_images: torch.Tensor) -> torch.Tensor:
        # support_images: [B, N, C, H, W]
        bsz, num_support, channels, height, width = support_images.shape
        support_flat = support_images.reshape(
            bsz * num_support, channels, height, width
        )
        support_embed = self.image_encoder(support_flat)
        support_embed = support_embed.reshape(
            bsz, num_support, *support_embed.shape[1:]
        )
        return support_embed

    def forward(
        self,
        image: torch.Tensor,
        box,
        support_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_embedding = self.image_encoder(image)

        if getattr(self, "local_global_adapter", None) is not None:
            image_embedding = self.local_global_adapter(image_embedding)

        if self.use_support_attention:
            if support_images is not None:
                if support_images.dim() == 4:
                    support_images = support_images.unsqueeze(1)
                if support_images.dim() != 5:
                    raise ValueError(
                        "support_images must be [B, N, C, H, W] or [B, C, H, W]"
                    )
                support_embedding = self._encode_support(support_images)
            else:
                # Fallback to self-support so the block remains trainable
                # when no explicit support set is provided.
                support_embedding = image_embedding.unsqueeze(1)

            image_embedding = self.attention_fusion(image_embedding, support_embedding)

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks
