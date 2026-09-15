"""Fused SwiGLU activation.

A gated MLP computes ``silu(x @ W_gate) * (x @ W_up)``. The two projections are
GEMMs and belong in cuBLAS; there is nothing to win by rewriting them. The
elementwise tail is a different story. Eager PyTorch runs ``sigmoid``, a
multiply for silu, and a second multiply for the gate as three separate kernels,
each reading and writing the full [M, N] intermediate. Backward is worse: it
re-reads ``a``, recomputes the sigmoid, and materialises two more full tensors.

One kernel, one pass:

    forward:   out = a * sigmoid(a) * b
    backward:  da = dout * b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
               db = dout * a * sigmoid(a)

The derivative of silu is ``sigmoid(a) * (1 + a * (1 - sigmoid(a)))``, recomputed
from ``a`` rather than stashed, which is the memory/compute trade you want on a
bandwidth-bound op.

This is a flat elementwise kernel, so it takes a 1-D grid over the flattened
tensor and does not care about the logical shape.
"""

from __future__ import annotations

import torch

from .reference import swiglu_ref
from .utils import HAS_TRITON

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_fwd(A, B, OUT, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(a)
        out = a * sig * b
        tl.store(OUT + offs, out, mask=mask)

    @triton.jit
    def _swiglu_bwd(DOUT, A, B, DA, DB, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        dout = tl.load(DOUT + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

        sig = tl.sigmoid(a)
        silu = a * sig
        dsilu = sig * (1.0 + a * (1.0 - sig))

        tl.store(DA + offs, dout * b * dsilu, mask=mask)
        tl.store(DB + offs, dout * silu, mask=mask)


class _FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a = a.contiguous()
        b = b.contiguous()
        out = torch.empty_like(a)
        n = out.numel()

        block = 1024
        grid = (triton.cdiv(n, block),)
        _swiglu_fwd[grid](a, b, out, n, BLOCK=block, num_warps=4)

        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    def backward(ctx, dout):
        a, b = ctx.saved_tensors
        dout = dout.contiguous()
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        n = a.numel()

        block = 1024
        grid = (triton.cdiv(n, block),)
        _swiglu_bwd[grid](dout, a, b, da, db, n, BLOCK=block, num_warps=4)
        return da, db


def fused_swiglu(
    a: torch.Tensor, b: torch.Tensor, force_reference: bool = False
) -> torch.Tensor:
    """SwiGLU: ``silu(a) * b``, fused into a single pass.

    Args:
        a: gate projection, any shape.
        b: up projection, same shape as ``a``.
    """
    if not HAS_TRITON or force_reference or not a.is_cuda:
        return swiglu_ref(a, b)
    return _FusedSwiGLU.apply(a, b)
