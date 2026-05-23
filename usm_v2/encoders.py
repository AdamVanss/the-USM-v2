"""
USM v2 — Concept and Vision encoders.

Both encoders use frozen backbones + trainable projection into the Poincaré ball.
The key change from v1: all hyperbolic operations go through the learnable manifold
so curvature is part of the computation graph.
"""

import torch
import torch.nn as nn
import geoopt
from typing import List, Tuple, Optional

from .manifold import (
    LearnablePoincareBall, clamp_to_ball, expmap, EPS
)


class ConceptEncoder(nn.Module):
    """
    Text concept encoder: frozen MiniLM backbone -> trainable projection -> Poincaré ball.
    """

    def __init__(self, manifold: LearnablePoincareBall, d_out: int = 1024,
                 backbone: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 hyperbolic: bool = True, device: torch.device = torch.device("cpu"),
                 backbone_device: Optional[torch.device] = None):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.hyperbolic = hyperbolic
        self.manifold = manifold
        self.device = device
        self.backbone_device = backbone_device or device

        self.backbone = SentenceTransformer(backbone, device=str(self.backbone_device))
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        d_in = self.backbone.get_sentence_embedding_dimension()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
        )

        if hyperbolic:
            self.mu0 = geoopt.ManifoldParameter(
                torch.zeros(d_out),
                manifold=geoopt.PoincareBall(c=1.0),
            )

    @torch.no_grad()
    def encode_backbone(self, texts: List[str]) -> torch.Tensor:
        embs = self.backbone.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embs.to(self.device).float()

    def project(self, bb_emb: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project backbone embeddings onto the manifold.

        Args:
            bb_emb: backbone embeddings [B, d_backbone]
            c: effective curvature (if None, uses manifold.c)
        """
        if c is None:
            c = self.manifold.c

        bb_emb = bb_emb.clone()
        v = self.proj(bb_emb)

        if self.hyperbolic and c.item() > EPS:
            mu0 = self.mu0.unsqueeze(0).expand(v.shape[0], -1)
            return clamp_to_ball(expmap(mu0, v, c), c)
        return v

    def forward(self, texts: List[str],
                c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bb_emb = self.encode_backbone(texts)
        z = self.project(bb_emb, c=c)
        return z, bb_emb


class VisionEncoder(nn.Module):
    """
    Vision encoder: frozen CLIP backbone -> trainable projection -> Poincaré ball.
    """

    def __init__(self, manifold: LearnablePoincareBall, d_out: int = 1024,
                 clip_model: str = "openai/clip-vit-large-patch14",
                 d_clip: int = 768, hyperbolic: bool = True,
                 device: torch.device = torch.device("cpu"),
                 backbone_device: Optional[torch.device] = None):
        super().__init__()
        from transformers import CLIPModel

        self.hyperbolic = hyperbolic
        self.manifold = manifold
        self.device = device
        self.backbone_device = backbone_device or device

        self.clip = CLIPModel.from_pretrained(clip_model).to(self.backbone_device)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.Linear(d_clip, d_out),
            nn.LayerNorm(d_out),
        )

        if hyperbolic:
            self.mu0 = geoopt.ManifoldParameter(
                torch.zeros(d_out),
                manifold=geoopt.PoincareBall(c=1.0),
            )

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_out = self.clip.vision_model(pixel_values=pixel_values.to(self.backbone_device))
        pooled = vision_out.pooler_output
        pooled = self.clip.visual_projection(pooled)
        return pooled.to(self.device).float()

    def project(self, clip_emb: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if c is None:
            c = self.manifold.c

        clip_emb = clip_emb.clone()
        v = self.proj(clip_emb)

        if self.hyperbolic and c.item() > EPS:
            mu0 = self.mu0.unsqueeze(0).expand(v.shape[0], -1)
            return clamp_to_ball(expmap(mu0, v, c), c)
        return v

    def forward(self, pixel_values: torch.Tensor,
                c: Optional[torch.Tensor] = None) -> torch.Tensor:
        clip_emb = self.encode_images(pixel_values)
        return self.project(clip_emb, c=c)
