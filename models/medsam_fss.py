"""MedSAM variant with optional attention-based feature fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import AttentionCrossBlock


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

    def _encode_support(self, support_images: torch.Tensor) -> torch.Tensor:
        # support_images: [B, N, C, H, W]
        bsz, num_support, channels, height, width = support_images.shape
        support_flat = support_images.reshape(bsz * num_support, channels, height, width)
        support_embed = self.image_encoder(support_flat)
        support_embed = support_embed.reshape(bsz, num_support, *support_embed.shape[1:])
        return support_embed

    def forward(
        self,
        image: torch.Tensor,
        box,
        support_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_embedding = self.image_encoder(image)

        if self.use_support_attention:
            if support_images is not None:
                if support_images.dim() == 4:
                    support_images = support_images.unsqueeze(1)
                if support_images.dim() != 5:
                    raise ValueError("support_images must be [B, N, C, H, W] or [B, C, H, W]")
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
