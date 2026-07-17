from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """
    Image patch embedding (ViT-style).

    Args:
        image_size: Input image size (assumed square)
        patch_size: Patch size (assumed square)
        in_channels: Input image channels (default: 3 for RGB)
        d_vision: Vision feature dimension
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_vision: int = 768,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, d_vision,
            kernel_size=patch_size, stride=patch_size,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_vision) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_vision) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, H, W] — input image

        Returns:
            embeddings: [batch, n_patches + 1, d_vision]
        """
        batch = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        return x


class VisionTransformerEncoder(nn.Module):
    """
    Simplified Vision Transformer (ViT) encoder — processes image patches
    into a single visual representation for NSLT fusion.

    Uses a small Transformer (attention-based) since this is the vision
    modality where attention works well for spatial relationships.

    Args:
        d_vision: Vision feature dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads (must divide d_vision)
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_vision: int = 768,
        n_layers: int = 6,
        n_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        n_heads = n_heads or max(1, d_vision // 64)
        d_ff = d_ff or d_vision * 4
        super().__init__()
        self.d_vision = d_vision

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_vision,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,

        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_vision)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_patches + 1, d_vision] — patch embeddings + cls token

        Returns:
            visual_features: [batch, d_vision] — CLS token representation
        """
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token [batch, d_vision]


class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP-style vision encoder for NSLT multimodal fusion.

    Encodes images into the NSLT compressed state space (d_state),
    enabling the model to process both text and images through the
    same O(1) memory pathway.

    Architecture:
        1. Patch embedding (ViT-style)
        2. Transformer encoder (SigLIP-style, 6 layers)
        3. Projection head: d_vision -> d_state

    Args:
        d_state: Target NSLT compressed state dimension
        image_size: Input image size
        patch_size: Patch size for ViT
        d_vision: Vision feature dimension
        n_layers: Number of ViT encoder layers
    """

    def __init__(
        self,
        d_state: int = 2048,
        image_size: int = 224,
        patch_size: int = 16,
        d_vision: int = 768,
        n_layers: int = 6,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_vision = d_vision

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            d_vision=d_vision,
        )

        self.encoder = VisionTransformerEncoder(
            d_vision=d_vision,
            n_layers=n_layers,
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(d_vision),
            nn.Linear(d_vision, d_state, bias=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, channels, H, W] — input images (normalized)

        Returns:
            visual_state: [batch, d_state] — image representation in NSLT state space
        """
        patches = self.patch_embed(images)  # [batch, n_patches+1, d_vision]
        visual_feats = self.encoder(patches)  # [batch, d_vision]
        visual_state = self.projection(visual_feats)  # [batch, d_state]
        return visual_state

    def get_config(self) -> dict:
        return {
            "type": "SigLIPVisionEncoder",
            "d_state": self.d_state,
            "d_vision": self.d_vision,
            "patch_embed": self.patch_embed.n_patches,
        }
