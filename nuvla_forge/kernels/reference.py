"""Pure-PyTorch reference implementations.

Every fused kernel in this package has a reference here. Tests check the kernel
against the reference; benchmarks time the kernel against the reference. The
reference is also the CPU fallback path, so the whole repo runs (slowly) without
a GPU, which is what makes the test suite runnable in CI.

Nothing here is optimised. That is the point.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _expand_cond(c: torch.Tensor, tokens_per_sample: int) -> torch.Tensor:
    """[B, N] conditioning -> [B * T, N], matching row-major (b, t) flattening."""
    return c.repeat_interleave(tokens_per_sample, dim=0)


def adaln_residual_rmsnorm_ref(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    tokens_per_sample: int,
    eps: float = 1e-6,
):
    """Gated residual add, then RMSNorm, then adaLN-Zero modulation.

    This is the exact sequence that appears between two consecutive sub-blocks
    of a DiT: the previous branch's gated residual write, immediately followed by
    the next branch's normalisation and modulation.

        h   = x + gate * y                      (residual stream, flows onward)
        out = rmsnorm(h) * w * (1 + scale) + shift   (input to the next branch)

    Args:
        x: [M, N] incoming residual stream, M = B * tokens_per_sample.
        y: [M, N] output of the previous branch (attention or MLP).
        gate: [B, N] adaLN-Zero gate for the previous branch.
        weight: [N] RMSNorm learnable gain.
        scale: [B, N] adaLN-Zero scale for the next branch.
        shift: [B, N] adaLN-Zero shift for the next branch.
        tokens_per_sample: T, so row m belongs to sample m // T.
        eps: RMSNorm epsilon, applied inside the sqrt.

    Returns:
        (h, out), both [M, N] and both in the dtype of ``x``.
    """
    t = tokens_per_sample
    g = _expand_cond(gate, t)
    s = _expand_cond(scale, t)
    sh = _expand_cond(shift, t)

    h = x + g * y

    hf = h.float()
    rstd = torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + eps)
    xhat = hf * rstd
    out = xhat * weight.float() * (1.0 + s.float()) + sh.float()

    return h, out.to(x.dtype)


def swiglu_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: silu(a) * b.

    ``a`` and ``b`` are the two halves of the gated MLP's first projection.
    """
    return (F.silu(a.float()) * b.float()).to(a.dtype)


def chunked_cross_entropy_ref(
    hidden: torch.Tensor,
    lm_weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Unfused cross entropy: materialise every logit, then reduce.

    Peak memory here is O(M * V) for the logits plus another O(M * V) for their
    gradient. For a 32k vocabulary and 8k tokens in flight that is ~2 GB per
    copy in bf16, which is the single largest allocation in the reasoning head.
    ``nuvla_forge.kernels.chunked_ce`` exists to delete it.
    """
    logits = F.linear(hidden.float(), lm_weight.float())
    return F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index,
        reduction="mean",
    )
