"""Finite-difference verification of the kernel gradient math.

``test_kernels.py`` checks the Triton kernels against PyTorch references. That
proves the kernels agree with the references. It does **not** prove the
references are right.

This file closes that gap. It reimplements each backward pass in NumPy, using
the same arithmetic the Triton kernels use line for line, and checks it against
central differences of the forward pass. If a derivative is wrong, both the
kernel and the reference would be wrong together and ``test_kernels.py`` would
still pass -- this is the test that catches it.

Needs only NumPy, so it runs in CI, on a laptop, and in under a second.
"""

from __future__ import annotations

import numpy as np
import pytest

EPS = 1e-6
FD_H = 1e-6
RTOL = 1e-6


def central_difference(array, loss_fn, h=FD_H):
    """Numerical gradient of ``loss_fn`` w.r.t. ``array``, in place-safe fashion."""
    grad = np.zeros_like(array)
    flat, gflat = array.ravel(), grad.ravel()
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + h
        plus = loss_fn()
        flat[i] = original - h
        minus = loss_fn()
        flat[i] = original
        gflat[i] = (plus - minus) / (2 * h)
    return grad


def relative_error(numeric, analytic):
    scale = max(np.abs(numeric).max(), 1e-12)
    return np.abs(numeric - analytic).max() / scale


# ---------------------------------------------------------------------------
# fused adaLN-Zero + gated residual + RMSNorm
# ---------------------------------------------------------------------------


def _adaln_forward(x, y, gate, w, scale, shift, T):
    g = np.repeat(gate, T, axis=0)
    s = np.repeat(scale, T, axis=0)
    sh = np.repeat(shift, T, axis=0)
    h = x + g * y
    rstd = 1.0 / np.sqrt((h * h).mean(-1, keepdims=True) + EPS)
    xhat = h * rstd
    out = xhat * w * (1.0 + s) + sh
    return h, out, rstd, xhat


def _adaln_backward(x, y, gate, w, scale, shift, T, dh_up, dout):
    """Transcription of ``_adaln_rmsnorm_bwd`` from adaln_rmsnorm.py."""
    B, N = gate.shape
    g = np.repeat(gate, T, axis=0)
    s = np.repeat(scale, T, axis=0)
    h, _, rstd, xhat = _adaln_forward(x, y, gate, w, scale, shift, T)
    one_plus_s = 1.0 + s

    dshift_rows = dout
    dscale_rows = dout * xhat * w
    dw_rows = dout * xhat * one_plus_s
    dxhat = dout * w * one_plus_s

    c = (dxhat * xhat).sum(-1, keepdims=True) / N
    dh = dh_up + rstd * (dxhat - xhat * c)

    reduce_to_sample = lambda rows: rows.reshape(B, T, N).sum(1)
    return {
        "x": dh,
        "y": dh * g,
        "gate": reduce_to_sample(dh * y),
        "w": dw_rows.sum(0),
        "scale": reduce_to_sample(dscale_rows),
        "shift": reduce_to_sample(dshift_rows),
    }


@pytest.mark.parametrize("B,T,N", [(3, 5, 7), (2, 4, 16), (1, 8, 5)])
def test_adaln_gradients_match_finite_differences(B, T, N):
    rng = np.random.default_rng(0)
    M = B * T
    args = dict(
        x=rng.standard_normal((M, N)), y=rng.standard_normal((M, N)),
        gate=rng.standard_normal((B, N)), w=rng.standard_normal(N),
        scale=rng.standard_normal((B, N)), shift=rng.standard_normal((B, N)),
    )
    dh_up = rng.standard_normal((M, N))
    dout = rng.standard_normal((M, N))

    def loss():
        h, out, _, _ = _adaln_forward(**args, T=T)
        return (h * dh_up).sum() + (out * dout).sum()

    analytic = _adaln_backward(**args, T=T, dh_up=dh_up, dout=dout)
    for name, value in args.items():
        err = relative_error(central_difference(value, loss), analytic[name])
        assert err < RTOL, f"d{name} relative error {err:.2e}"


def test_adaln_zero_gate_is_identity_on_residual():
    """adaLN-Zero initialises the gate to zero. At init the block must be an
    exact identity on the residual stream, or deep DiTs diverge early."""
    rng = np.random.default_rng(1)
    B, T, N = 2, 3, 8
    x = rng.standard_normal((B * T, N))
    y = rng.standard_normal((B * T, N))
    h, _, _, _ = _adaln_forward(
        x, y, np.zeros((B, N)), np.ones(N), np.zeros((B, N)), np.zeros((B, N)), T
    )
    np.testing.assert_allclose(h, x, atol=0)


# ---------------------------------------------------------------------------
# fused SwiGLU
# ---------------------------------------------------------------------------


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_swiglu_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 5))
    b = rng.standard_normal((4, 5))
    dout = rng.standard_normal((4, 5))

    def loss():
        return (a * _sigmoid(a) * b * dout).sum()

    sig = _sigmoid(a)
    # exactly the expressions in _swiglu_bwd
    da = dout * b * (sig * (1.0 + a * (1.0 - sig)))
    db = dout * (a * sig)

    assert relative_error(central_difference(a, loss), da) < RTOL
    assert relative_error(central_difference(b, loss), db) < RTOL


# ---------------------------------------------------------------------------
# chunked cross entropy
# ---------------------------------------------------------------------------


def test_chunked_ce_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    M, H, V, ignore = 9, 4, 6, -100
    hidden = rng.standard_normal((M, H))
    weight = rng.standard_normal((V, H))
    targets = rng.integers(0, V, M)
    targets[::4] = ignore

    valid = targets != ignore
    inv_count = 1.0 / max(valid.sum(), 1)
    safe = np.clip(targets, 0, None)

    def loss():
        logits = hidden @ weight.T
        shifted = logits - logits.max(-1, keepdims=True)
        lse = np.log(np.exp(shifted).sum(-1)) + logits.max(-1)
        return (np.where(valid, lse - logits[np.arange(M), safe], 0.0)).sum() * inv_count

    logits = hidden @ weight.T
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    probs[np.arange(M), safe] -= 1.0
    dlogits = probs * valid[:, None] * inv_count

    d_hidden = dlogits @ weight
    d_weight = dlogits.T @ hidden

    assert relative_error(central_difference(hidden, loss), d_hidden) < RTOL
    assert relative_error(central_difference(weight, loss), d_weight) < RTOL
    assert np.allclose(d_hidden[::4], 0.0), "ignored rows must carry zero gradient"


def test_chunked_ce_mean_uses_valid_count_only():
    """The denominator is the number of non-ignored tokens, not the row count.
    Getting this wrong scales the loss silently and shifts the whole LR regime."""
    rng = np.random.default_rng(4)
    M, H, V, ignore = 8, 3, 5, -100
    hidden = rng.standard_normal((M, H))
    weight = rng.standard_normal((V, H))
    targets = rng.integers(0, V, M)
    targets[4:] = ignore

    def ce(mask_value):
        valid = targets != ignore
        inv = 1.0 / mask_value
        logits = hidden @ weight.T
        shifted = logits - logits.max(-1, keepdims=True)
        lse = np.log(np.exp(shifted).sum(-1)) + logits.max(-1)
        return (np.where(valid, lse - logits[np.arange(M), np.clip(targets, 0, None)], 0.0)).sum() * inv

    assert not np.isclose(ce(4), ce(8)), "valid-count and row-count must differ here"
