"""Kernel correctness. Run this before you believe any benchmark in this repo.

Every fused kernel is checked against the PyTorch reference in
``nuvla_forge.kernels.reference`` for both forward values and all gradients.

Tolerances are stated per dtype rather than tuned until green. fp32 gets a tight
bound. bf16 gets a loose one because bf16 has 8 mantissa bits and a reduction
over N=1024 elements accumulates real error -- the fused kernel reduces in fp32
internally, so it is often *more* accurate than the eager path, but the inputs
and outputs are still bf16 and that quantisation dominates.

On a machine without CUDA the Triton paths are skipped and the reference path is
exercised, which still catches shape and broadcasting bugs. CI runs exactly that.
"""

from __future__ import annotations

import pytest
import torch

from nuvla_forge.kernels import (
    chunked_cross_entropy,
    fused_adaln_residual_rmsnorm,
    fused_swiglu,
)
from nuvla_forge.kernels.reference import (
    adaln_residual_rmsnorm_ref,
    chunked_cross_entropy_ref,
    swiglu_ref,
)
from nuvla_forge.kernels.utils import HAS_TRITON

CUDA = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(
    not (CUDA and HAS_TRITON), reason="needs CUDA and Triton"
)

TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    torch.float16: dict(atol=5e-3, rtol=5e-3),
}


def _close(got, want, dtype, name):
    t = TOL[dtype]
    torch.testing.assert_close(
        got.float(), want.float(), msg=lambda m: f"[{name}] {m}", **t
    )


# --------------------------------------------------------------------------
# fused adaLN-Zero + gated residual + RMSNorm
# --------------------------------------------------------------------------


def _adaln_inputs(b, t, n, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g, device=device, dtype=dtype, requires_grad=True)
    return dict(
        x=mk(b * t, n),
        y=mk(b * t, n),
        gate=mk(b, n),
        weight=mk(n),
        scale=mk(b, n),
        shift=mk(b, n),
    )


def _run_adaln(fn, inp, t, seed=1):
    h, out = fn(**inp, tokens_per_sample=t)
    g = torch.Generator(device=h.device).manual_seed(seed)
    dh = torch.randn(*h.shape, generator=g, device=h.device, dtype=h.dtype)
    dout = torch.randn(*out.shape, generator=g, device=h.device, dtype=h.dtype)
    (h * dh + out * dout).sum().backward()
    return h, out


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("b,t,n", [(2, 16, 256), (4, 32, 1024), (3, 7, 768), (1, 64, 1536)])
def test_adaln_matches_reference(dtype, b, t, n):
    dev = "cuda"
    fused_in = _adaln_inputs(b, t, n, dtype, dev)
    ref_in = {k: v.detach().clone().requires_grad_(True) for k, v in fused_in.items()}

    h_f, out_f = _run_adaln(
        lambda **kw: fused_adaln_residual_rmsnorm(**kw), fused_in, t
    )
    h_r, out_r = _run_adaln(
        lambda **kw: adaln_residual_rmsnorm_ref(**kw), ref_in, t
    )

    _close(h_f, h_r, dtype, "h")
    _close(out_f, out_r, dtype, "out")
    for name in fused_in:
        _close(fused_in[name].grad, ref_in[name].grad, dtype, f"d{name}")


@requires_gpu
def test_adaln_non_power_of_two_hidden():
    """N=1000 exercises the masked tail. A wrong mask silently corrupts the mean."""
    dev, dtype, b, t, n = "cuda", torch.float32, 2, 8, 1000
    fused_in = _adaln_inputs(b, t, n, dtype, dev)
    ref_in = {k: v.detach().clone().requires_grad_(True) for k, v in fused_in.items()}
    h_f, out_f = _run_adaln(lambda **kw: fused_adaln_residual_rmsnorm(**kw), fused_in, t)
    h_r, out_r = _run_adaln(lambda **kw: adaln_residual_rmsnorm_ref(**kw), ref_in, t)
    _close(out_f, out_r, dtype, "out")
    _close(fused_in["weight"].grad, ref_in["weight"].grad, dtype, "dweight")


def test_adaln_reference_is_self_consistent():
    """CPU-only sanity check: the reference matches a naive transcription."""
    b, t, n = 2, 4, 32
    inp = _adaln_inputs(b, t, n, torch.float32, "cpu")
    h, out = adaln_residual_rmsnorm_ref(**inp, tokens_per_sample=t)

    gate = inp["gate"].repeat_interleave(t, 0)
    manual_h = inp["x"] + gate * inp["y"]
    rms = manual_h.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    manual_out = (
        manual_h * rms * inp["weight"]
        * (1 + inp["scale"].repeat_interleave(t, 0))
        + inp["shift"].repeat_interleave(t, 0)
    )
    torch.testing.assert_close(h, manual_h)
    torch.testing.assert_close(out, manual_out, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# fused SwiGLU
# --------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64, 512), (128, 2048), (17, 333)])
def test_swiglu_matches_reference(dtype, shape):
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(*shape, generator=g, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(*shape, generator=g, device="cuda", dtype=dtype, requires_grad=True)
    ar, br = a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True)

    dout = torch.randn(*shape, generator=g, device="cuda", dtype=dtype)

    out_f = fused_swiglu(a, b)
    out_f.backward(dout)
    out_r = swiglu_ref(ar, br)
    out_r.backward(dout)

    _close(out_f, out_r, dtype, "out")
    _close(a.grad, ar.grad, dtype, "da")
    _close(b.grad, br.grad, dtype, "db")


def test_swiglu_reference_gradcheck():
    """Double-precision gradcheck on the reference, on CPU."""
    a = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    b = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(swiglu_ref, (a, b), eps=1e-6, atol=1e-8)


# --------------------------------------------------------------------------
# chunked cross entropy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
@pytest.mark.parametrize("chunk", [128, 4096])
def test_chunked_ce_matches_reference(device, chunk):
    m, h, v = 512, 64, 1000
    g = torch.Generator(device=device).manual_seed(0)
    hid = torch.randn(m, h, generator=g, device=device, requires_grad=True)
    w = torch.randn(v, h, generator=g, device=device, requires_grad=True) * 0.02
    w = w.detach().requires_grad_(True)
    tgt = torch.randint(0, v, (m,), generator=g, device=device)

    hr, wr = hid.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)

    loss_f = chunked_cross_entropy(hid, w, tgt, chunk_size=chunk)
    loss_f.backward()
    loss_r = chunked_cross_entropy_ref(hr, wr, tgt)
    loss_r.backward()

    torch.testing.assert_close(loss_f, loss_r, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(hid.grad, hr.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w.grad, wr.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_chunked_ce_respects_ignore_index(device):
    """Ignored rows must contribute neither loss nor gradient, and must not be
    counted in the mean. Getting the denominator wrong is a silent loss-scale bug."""
    m, h, v = 64, 32, 200
    g = torch.Generator(device=device).manual_seed(3)
    hid = torch.randn(m, h, generator=g, device=device, requires_grad=True)
    w = (torch.randn(v, h, generator=g, device=device) * 0.02).requires_grad_(True)
    tgt = torch.randint(0, v, (m,), generator=g, device=device)
    tgt[::3] = -100

    hr, wr = hid.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)

    loss_f = chunked_cross_entropy(hid, w, tgt, chunk_size=16)
    loss_f.backward()
    loss_r = chunked_cross_entropy_ref(hr, wr, tgt)
    loss_r.backward()

    torch.testing.assert_close(loss_f, loss_r, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(hid.grad, hr.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(hid.grad[::3], torch.zeros_like(hid.grad[::3]))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_chunked_ce_all_ignored_is_finite(device):
    """Degenerate batch: every token masked. Must not divide by zero."""
    m, h, v = 16, 8, 50
    hid = torch.randn(m, h, device=device, requires_grad=True)
    w = torch.randn(v, h, device=device, requires_grad=True)
    tgt = torch.full((m,), -100, dtype=torch.long, device=device)
    loss = chunked_cross_entropy(hid, w, tgt, chunk_size=4)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(hid.grad).all()


@requires_gpu
def test_chunked_ce_peak_memory_is_lower():
    """The whole reason this kernel exists. If this fails, delete the kernel."""
    m, h, v = 4096, 512, 32000
    hid = torch.randn(m, h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(v, h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    tgt = torch.randint(0, v, (m,), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    chunked_cross_entropy_ref(hid, w, tgt).backward()
    naive_peak = torch.cuda.max_memory_allocated()

    hid.grad = None
    w.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    chunked_cross_entropy(hid, w, tgt, chunk_size=1024).backward()
    chunked_peak = torch.cuda.max_memory_allocated()

    print(f"\nnaive peak   {naive_peak / 2**20:8.1f} MiB")
    print(f"chunked peak {chunked_peak / 2**20:8.1f} MiB")
    print(f"reduction    {naive_peak / chunked_peak:8.2f}x")
    assert chunked_peak < naive_peak
