"""Fused gated-residual + RMSNorm + adaLN-Zero modulation.

Why this kernel exists
----------------------
A DiT block conditioned with adaLN-Zero looks like this in eager PyTorch::

    shift1, scale1, gate1, shift2, scale2, gate2 = cond(c).chunk(6, dim=1)
    x = x + gate1.unsqueeze(1) * attn(modulate(norm1(x), shift1, scale1))
    x = x + gate2.unsqueeze(1) * mlp( modulate(norm2(x), shift2, scale2))

where ``modulate(v, shift, scale) = v * (1 + scale) + shift``.

Between the end of one sub-block and the start of the next, PyTorch executes:
a broadcast multiply, an add, a square, a mean, an rsqrt, a multiply, a gain
multiply, a broadcast multiply and a broadcast add. Every one is its own CUDA
kernel and every one makes a full round trip through global memory. For a
[B*T, N] activation in bf16 that is on the order of 9 reads and 9 writes of the
whole tensor when the arithmetic needs 2 reads and 2 writes.

This is pure memory-bandwidth waste. The block is bandwidth bound, not compute
bound, so the round trips *are* the cost.

This kernel does the whole sequence in one pass: load x, y and the conditioning
vectors once, keep the row in registers, and write h and out exactly once.

    h   = x + gate * y
    out = rmsnorm(h) * weight * (1 + scale) + shift

``h`` is returned because it is the residual stream and the next sub-block needs
it. ``out`` is the modulated input to the next branch.

Shapes
------
x, y, h, out : [M, N] where M = B * T, row-major, row m belongs to sample m // T
gate, scale, shift : [B, N]  (one conditioning vector per sample)
weight : [N]

Backward strategy
-----------------
dgate, dscale, dshift are [B, N] and dweight is [N], so all four need a
reduction across the M axis. Rather than atomics (non-deterministic, slow under
contention) or locks (the Triton layer-norm tutorial's approach), the grid is
(B * PROGRAMS_PER_SAMPLE,). Each program owns one sample and a strided slice of
that sample's rows, accumulates its four partials in registers, and writes them
once to a [B * P, N] scratch buffer. The final reduction is a couple of
``.sum()`` calls on a tiny tensor in PyTorch. Deterministic, lock-free, and the
scratch buffer is B * P * N floats rather than M * N.
"""

from __future__ import annotations

import torch

from .reference import adaln_residual_rmsnorm_ref
from .utils import HAS_TRITON, next_power_of_2, warps_for_block

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _adaln_rmsnorm_fwd(
        X, Y, GATE, W, SCALE, SHIFT,
        H, OUT, RSTD,
        stride_row,
        N, T,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // T

        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Y + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(GATE + b * N + cols, mask=mask, other=0.0).to(tl.float32)

        h = x + g * y
        tl.store(H + row * stride_row + cols, h, mask=mask)

        # RMSNorm over the feature axis. Masked lanes hold 0.0 so they add nothing
        # to the sum of squares, and we divide by the true N rather than BLOCK_N.
        var = tl.sum(h * h, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(RSTD + row, rstd)

        xhat = h * rstd
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(SCALE + b * N + cols, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(SHIFT + b * N + cols, mask=mask, other=0.0).to(tl.float32)

        out = xhat * w * (1.0 + s) + sh
        tl.store(OUT + row * stride_row + cols, out, mask=mask)

    @triton.jit
    def _adaln_rmsnorm_bwd(
        DH_UP, DOUT,
        X, Y, GATE, W, SCALE, H, RSTD,
        DX, DY,
        DGATE_P, DSCALE_P, DSHIFT_P, DW_P,
        stride_row,
        N, T,
        PROGRAMS_PER_SAMPLE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // PROGRAMS_PER_SAMPLE
        p = pid % PROGRAMS_PER_SAMPLE

        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(GATE + b * N + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(SCALE + b * N + cols, mask=mask, other=0.0).to(tl.float32)
        one_plus_s = 1.0 + s

        acc_dgate = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc_dscale = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc_dshift = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc_dw = tl.zeros([BLOCK_N], dtype=tl.float32)

        for t in range(p, T, PROGRAMS_PER_SAMPLE):
            row = b * T + t
            off = row * stride_row + cols

            dout = tl.load(DOUT + off, mask=mask, other=0.0).to(tl.float32)
            dh_up = tl.load(DH_UP + off, mask=mask, other=0.0).to(tl.float32)
            h = tl.load(H + off, mask=mask, other=0.0).to(tl.float32)
            y = tl.load(Y + off, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.load(RSTD + row).to(tl.float32)

            xhat = h * rstd

            # out = xhat * w * (1 + s) + shift
            acc_dshift += dout
            acc_dscale += dout * xhat * w
            acc_dw += dout * xhat * one_plus_s

            dxhat = dout * w * one_plus_s

            # RMSNorm backward:
            #   dh = rstd * (dxhat - xhat * mean(dxhat * xhat))
            c = tl.sum(dxhat * xhat, axis=0) / N
            dh_norm = rstd * (dxhat - xhat * c)

            # h is consumed twice: by the norm, and by whatever downstream op
            # reads the residual stream. Both gradients land here.
            dh = dh_up + dh_norm

            # h = x + gate * y
            tl.store(DX + off, dh, mask=mask)
            tl.store(DY + off, dh * g, mask=mask)
            acc_dgate += dh * y

        slot = b * PROGRAMS_PER_SAMPLE + p
        tl.store(DGATE_P + slot * N + cols, acc_dgate, mask=mask)
        tl.store(DSCALE_P + slot * N + cols, acc_dscale, mask=mask)
        tl.store(DSHIFT_P + slot * N + cols, acc_dshift, mask=mask)
        tl.store(DW_P + slot * N + cols, acc_dw, mask=mask)


class _FusedAdaLNResidualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gate, weight, scale, shift, tokens_per_sample, eps):
        x = x.contiguous()
        y = y.contiguous()
        m, n = x.shape
        t = tokens_per_sample

        h = torch.empty_like(x)
        out = torch.empty_like(x)
        rstd = torch.empty(m, dtype=torch.float32, device=x.device)

        block_n = next_power_of_2(n)
        _adaln_rmsnorm_fwd[(m,)](
            x, y, gate.contiguous(), weight.contiguous(),
            scale.contiguous(), shift.contiguous(),
            h, out, rstd,
            x.stride(0),
            n, t,
            eps,
            BLOCK_N=block_n,
            num_warps=warps_for_block(block_n),
        )

        ctx.save_for_backward(x, y, gate, weight, scale, h, rstd)
        ctx.tokens_per_sample = t
        ctx.block_n = block_n
        return h, out

    @staticmethod
    def backward(ctx, dh_up, dout):
        x, y, gate, weight, scale, h, rstd = ctx.saved_tensors
        t = ctx.tokens_per_sample
        m, n = x.shape
        b = m // t

        dh_up = dh_up.contiguous()
        dout = dout.contiguous()

        dx = torch.empty_like(x)
        dy = torch.empty_like(x)

        # One program per (sample, stride-slice). 16 slices per sample keeps
        # enough programs in flight to fill the SMs without blowing up scratch.
        programs_per_sample = min(16, max(1, t))
        slots = b * programs_per_sample

        fp32 = {"dtype": torch.float32, "device": x.device}
        dgate_p = torch.empty(slots, n, **fp32)
        dscale_p = torch.empty(slots, n, **fp32)
        dshift_p = torch.empty(slots, n, **fp32)
        dw_p = torch.empty(slots, n, **fp32)

        _adaln_rmsnorm_bwd[(slots,)](
            dh_up, dout,
            x, y, gate.contiguous(), weight.contiguous(), scale.contiguous(),
            h, rstd,
            dx, dy,
            dgate_p, dscale_p, dshift_p, dw_p,
            x.stride(0),
            n, t,
            PROGRAMS_PER_SAMPLE=programs_per_sample,
            BLOCK_N=ctx.block_n,
            num_warps=warps_for_block(ctx.block_n),
        )

        dgate = dgate_p.view(b, programs_per_sample, n).sum(1).to(gate.dtype)
        dscale = dscale_p.view(b, programs_per_sample, n).sum(1).to(scale.dtype)
        dshift = dshift_p.view(b, programs_per_sample, n).sum(1).to(scale.dtype)
        dweight = dw_p.sum(0).to(weight.dtype)

        return dx, dy, dgate, dweight, dscale, dshift, None, None


def fused_adaln_residual_rmsnorm(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    tokens_per_sample: int,
    eps: float = 1e-6,
    force_reference: bool = False,
):
    """Fused gated residual + RMSNorm + adaLN-Zero modulation.

    Falls back to the PyTorch reference when Triton is unavailable or the tensors
    are not on CUDA, so the same code path runs in CI on CPU.

    Returns:
        (h, out) — the updated residual stream and the modulated branch input.
    """
    use_triton = (
        HAS_TRITON
        and not force_reference
        and x.is_cuda
        and x.shape[-1] <= 16384
    )
    if not use_triton:
        return adaln_residual_rmsnorm_ref(
            x, y, gate, weight, scale, shift, tokens_per_sample, eps
        )
    return _FusedAdaLNResidualRMSNorm.apply(
        x, y, gate, weight, scale, shift, tokens_per_sample, eps
    )
