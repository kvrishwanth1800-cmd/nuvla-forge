"""Shared helpers for the kernel package."""

from __future__ import annotations

import os

try:  # pragma: no cover - import guard
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    HAS_TRITON = True
except Exception:  # noqa: BLE001 -- pragma: no cover -- optional
    # dependency probe; any import failure means Triton is unavailable.
    HAS_TRITON = False

# Escape hatch: NUVLA_FORGE_DISABLE_TRITON=1 forces every kernel onto its
# reference path. Used by the benchmark harness to time the baseline, and
# occasionally useful for bisecting a numerical problem.
if os.environ.get("NUVLA_FORGE_DISABLE_TRITON", "0") == "1":
    HAS_TRITON = False


def next_power_of_2(n: int) -> int:
    """Smallest power of two >= n. Triton block sizes must be powers of two."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def warps_for_block(block_n: int) -> int:
    """Warp count heuristic for single-row reduction kernels.

    Wider rows need more warps to keep the reduction tree shallow, but past 32
    warps occupancy falls off and register pressure starts spilling. These
    thresholds match the PyTorch/Triton layer-norm conventions and are a starting
    point, not a tuned optimum -- `bench_kernels.py --autotune` sweeps them.
    """
    if block_n >= 8192:
        return 32
    if block_n >= 4096:
        return 16
    if block_n >= 2048:
        return 8
    return 4
