"""Trajectory DiT with adaLN-Zero conditioning.

Follows the nuVLA arrangement: encoded scene hidden states condition a diffusion
transformer that denoises a future ego trajectory. Conditioning enters through
adaLN-Zero, which is what makes ``fused_adaln_residual_rmsnorm`` the right kernel
for this model rather than a generic one.

The block is written so the fused path and the eager path are the *same*
arithmetic, selected by a flag. That is deliberate: the benchmark compares two
code paths through one model, so any step-time difference is attributable to the
kernels and not to an accidental architecture change.

Flow matching rather than DDPM
------------------------------
The denoiser is trained with rectified flow: sample t ~ U(0,1), interpolate
``z_t = (1-t) * noise + t * trajectory``, and regress the constant velocity
``trajectory - noise``. One loss term, no noise schedule, no variance
parameterisation, and sampling is a handful of Euler steps. For a 3-second
waypoint trajectory this trains faster and more stably than a DDPM schedule,
which matters when the whole run has to finish overnight.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import fused_adaln_residual_rmsnorm, fused_swiglu


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rstd).to(x.dtype) * self.weight


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal flow-time embedding, projected to the conditioning width."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context=None):
        b, t, c = x.shape
        if context is None:
            qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        else:
            # Cross-attention into the scene tokens: q from the trajectory,
            # k/v from the encoder. Reuses the same fused qkv weight by slicing.
            q = F.linear(x, self.qkv.weight[:c]).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            kv = F.linear(context, self.qkv.weight[c:])
            s = context.shape[1]
            k, v = kv.view(b, s, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(b, t, c))


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, fused: bool = True):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.fused = fused

    def forward(self, x):
        a, b = self.w_gate(x), self.w_up(x)
        if self.fused:
            act = fused_swiglu(a.flatten(0, -2), b.flatten(0, -2)).view_as(a)
        else:
            act = F.silu(a) * b
        return self.w_down(act)


class DiTBlock(nn.Module):
    """One adaLN-Zero DiT block.

    Eager path (``fused=False``) is the canonical formulation. Fused path routes
    the residual-add / norm / modulate seam through the Triton kernel. Both
    compute the same function; only the memory traffic differs.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, fused: bool = True):
        super().__init__()
        self.dim = dim
        self.fused = fused
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.cross = Attention(dim, n_heads)
        self.norm_cross = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, int(dim * mlp_ratio), fused=fused)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        # adaLN-*Zero*: the conditioning projection starts at zero so every block
        # is an identity at init and the residual stream is untouched. Without
        # this, deep DiTs diverge in the first few hundred steps.
        nn.init.zeros_(self.cond[1].weight)
        nn.init.zeros_(self.cond[1].bias)
        # Cross-attention uses an unmodulated norm. Rather than allocating
        # zeros_like(scale) twice per block per step inside the fused path --
        # which would quietly tax the very path we are benchmarking -- keep one
        # zero buffer and one ones buffer and slice them.
        self.register_buffer("_zeros", torch.zeros(1, dim), persistent=False)
        self.register_buffer("_ones", torch.ones(1, dim), persistent=False)

    def forward(self, x, c, context):
        b, t, _ = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.cond(c).chunk(6, dim=-1)

        if self.fused:
            h = modulate(self.norm1(x), shift1, scale1)
            attn_out = self.attn(h)
            # Seam 1: gated residual for attention, then norm+modulate for cross.
            zeros = self._zeros.to(x.dtype).expand(b, self.dim)
            x, h = fused_adaln_residual_rmsnorm(
                x.flatten(0, 1), attn_out.flatten(0, 1), gate1,
                self.norm_cross.weight, zeros, zeros,
                tokens_per_sample=t,
            )
            x = x.view(b, t, -1)
            cross_out = self.cross(h.view(b, t, -1), context)
            # Seam 2: gated residual for cross-attention, then norm+modulate for MLP.
            ones = self._ones.to(x.dtype).expand(b, self.dim)
            x, h = fused_adaln_residual_rmsnorm(
                x.flatten(0, 1), cross_out.flatten(0, 1), ones,
                self.norm2.weight, scale2, shift2,
                tokens_per_sample=t,
            )
            x = x.view(b, t, -1)
            mlp_out = self.mlp(h.view(b, t, -1))
            x = x + gate2.unsqueeze(1) * mlp_out
        else:
            x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
            x = x + self.cross(self.norm_cross(x), context)
            x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))

        return x


class TrajectoryDiT(nn.Module):
    """Denoises a [B, horizon, 2] ego trajectory conditioned on scene tokens."""

    def __init__(
        self,
        dim: int = 384,
        depth: int = 6,
        n_heads: int = 6,
        horizon: int = 6,
        fused: bool = True,
    ):
        super().__init__()
        self.horizon = horizon
        self.in_proj = nn.Linear(2, dim)
        self.pos = nn.Parameter(torch.zeros(1, horizon, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.t_embed = TimestepEmbedder(dim)
        self.scene_proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, n_heads, fused=fused) for _ in range(depth)]
        )
        self.norm_out = RMSNorm(dim)
        self.out = nn.Linear(dim, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_t, t, scene_tokens):
        x = self.in_proj(z_t) + self.pos
        c = self.t_embed(t) + self.scene_proj(scene_tokens.mean(dim=1))
        for blk in self.blocks:
            x = blk(x, c, scene_tokens)
        return self.out(self.norm_out(x))

    def flow_loss(self, trajectory, scene_tokens):
        """Rectified-flow objective. Returns a scalar."""
        b = trajectory.shape[0]
        noise = torch.randn_like(trajectory)
        t = torch.rand(b, device=trajectory.device)
        tt = t[:, None, None]
        z_t = (1 - tt) * noise + tt * trajectory
        target_v = trajectory - noise
        return F.mse_loss(self(z_t, t, scene_tokens).float(), target_v.float())

    @torch.no_grad()
    def sample(self, scene_tokens, steps: int = 10):
        """Euler integration from noise to trajectory."""
        b = scene_tokens.shape[0]
        z = torch.randn(b, self.horizon, 2, device=scene_tokens.device, dtype=scene_tokens.dtype)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((b,), i * dt, device=z.device)
            z = z + self(z, t, scene_tokens) * dt
        return z
