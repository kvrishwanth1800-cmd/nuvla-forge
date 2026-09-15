"""Fused Triton kernels for VLA-style driving models.

Each kernel ships with a PyTorch reference (``reference.py``), a correctness test
against that reference, and a benchmark. If a kernel has no measured win on the
target hardware it does not belong here.
"""

from .adaln_rmsnorm import fused_adaln_residual_rmsnorm
from .chunked_ce import chunked_cross_entropy
from .swiglu import fused_swiglu
from .utils import HAS_TRITON

__all__ = [
    "HAS_TRITON",
    "chunked_cross_entropy",
    "fused_adaln_residual_rmsnorm",
    "fused_swiglu",
]
