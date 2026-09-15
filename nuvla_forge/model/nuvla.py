"""A compact nuVLA-shaped model.

    6 camera views -> shared ViT -> per-view tokens + camera embeddings
                   -> fusion transformer -> scene tokens
                   -> trajectory DiT     (5s / 3s ego waypoints, flow matching)
                   -> reasoning head     (VQA tokens, chunked cross entropy)

This is deliberately small. The repo is about step time, not leaderboard rank,
and a model you can train to convergence overnight on two consumer cards makes
the optimisation work legible. Every architectural choice that costs throughput
(number of views, token count, hidden width) is a config knob so the benchmarks
can sweep it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..kernels import chunked_cross_entropy
from .dit import RMSNorm, SwiGLUMLP, TrajectoryDiT


@dataclass
class NuVLAConfig:
    image_size: int = 224
    patch_size: int = 16
    n_views: int = 6
    vision_dim: int = 384
    vision_depth: int = 6
    vision_heads: int = 6
    fusion_depth: int = 4
    dit_depth: int = 6
    horizon: int = 6          # 6 waypoints @ 0.5s = 3s, the nuScenes L2 convention
    vocab_size: int = 32000
    max_text_len: int = 64
    ce_chunk_size: int = 1024
    fused: bool = True

    @property
    def tokens_per_view(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, fused=True):
        super().__init__()
        from .dit import Attention

        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, dim * 4, fused=fused)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MultiViewEncoder(nn.Module):
    """Shared-weight ViT over N views, with learned per-camera embeddings.

    Views are folded into the batch dimension so the ViT sees [B*V, P, D]. This
    is the shape that makes the dataloader matter: at 6 views and 196 tokens per
    view, a batch of 8 is 48 JPEG decodes per step, which a naive CPU loader
    cannot keep up with. See ``nuvla_forge/data/loader.py``.
    """

    def __init__(self, cfg: NuVLAConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.vision_dim
        self.patch = nn.Conv2d(3, d, cfg.patch_size, cfg.patch_size)
        self.pos = nn.Parameter(torch.zeros(1, cfg.tokens_per_view, d))
        self.cam_embed = nn.Parameter(torch.zeros(cfg.n_views, d))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cam_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, cfg.vision_heads, cfg.fused) for _ in range(cfg.vision_depth)]
        )
        self.norm = RMSNorm(d)

    def forward(self, images):
        """images: [B, V, 3, H, W] -> [B, V * P, D]"""
        b, v = images.shape[:2]
        x = self.patch(images.flatten(0, 1))          # [B*V, D, h, w]
        x = x.flatten(2).transpose(1, 2) + self.pos   # [B*V, P, D]
        x = x + self.cam_embed.repeat(b, 1).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.reshape(b, v * x.shape[1], -1)


class NuVLA(nn.Module):
    def __init__(self, cfg: NuVLAConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or NuVLAConfig()
        d = cfg.vision_dim

        self.encoder = MultiViewEncoder(cfg)
        self.fusion = nn.ModuleList(
            [TransformerBlock(d, cfg.vision_heads, cfg.fused) for _ in range(cfg.fusion_depth)]
        )
        # A small learned query set compresses V*P view tokens down to a fixed
        # scene summary, so the DiT's cross-attention cost is independent of how
        # many cameras or patches the encoder produced.
        self.scene_queries = nn.Parameter(torch.zeros(1, 32, d))
        nn.init.normal_(self.scene_queries, std=0.02)
        self.scene_attn = nn.MultiheadAttention(d, cfg.vision_heads, batch_first=True)

        self.dit = TrajectoryDiT(
            dim=d, depth=cfg.dit_depth, n_heads=cfg.vision_heads,
            horizon=cfg.horizon, fused=cfg.fused,
        )

        self.text_embed = nn.Embedding(cfg.vocab_size, d)
        self.text_blocks = nn.ModuleList(
            [TransformerBlock(d, cfg.vision_heads, cfg.fused) for _ in range(2)]
        )
        self.lm_head = nn.Parameter(torch.empty(cfg.vocab_size, d))
        nn.init.normal_(self.lm_head, std=0.02)

    def encode_scene(self, images):
        x = self.encoder(images)
        for blk in self.fusion:
            x = blk(x)
        q = self.scene_queries.expand(x.shape[0], -1, -1)
        scene, _ = self.scene_attn(q, x, x, need_weights=False)
        return scene

    def forward(self, images, trajectory=None, text_tokens=None, text_targets=None):
        """Joint objective: flow-matching plan loss + reasoning cross entropy.

        The nuReasoning paper's finding is that reasoning supervision improves
        planning even when the text is not used at inference, so both heads train
        together and both contribute to the loss.
        """
        scene = self.encode_scene(images)
        out = {}
        total = scene.new_zeros(())

        if trajectory is not None:
            plan = self.dit.flow_loss(trajectory, scene)
            out["plan_loss"] = plan.detach()
            total = total + plan

        if text_tokens is not None and text_targets is not None:
            h = self.text_embed(text_tokens)
            h = torch.cat([scene, h], dim=1)
            for blk in self.text_blocks:
                h = blk(h)
            h = h[:, scene.shape[1]:]  # drop scene positions, keep text
            reason = chunked_cross_entropy(
                h.flatten(0, 1),
                self.lm_head,
                text_targets.flatten(),
                chunk_size=self.cfg.ce_chunk_size,
                force_reference=not self.cfg.fused,
            )
            out["reason_loss"] = reason.detach()
            total = total + reason

        out["loss"] = total
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
