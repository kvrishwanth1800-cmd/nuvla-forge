"""Kernel microbenchmarks.

Reports latency, achieved bandwidth, and achieved bandwidth as a fraction of the
card's spec sheet. The last one is the only figure that tells you whether to keep
optimising.

    python -m nuvla_forge.bench.bench_kernels --out reports/kernels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..kernels import chunked_cross_entropy, fused_adaln_residual_rmsnorm, fused_swiglu
from ..kernels.reference import (
    adaln_residual_rmsnorm_ref,
    chunked_cross_entropy_ref,
    swiglu_ref,
)
from ..profiling import bandwidth_gb_s, roofline_note

# Spec-sheet peak HBM/GDDR bandwidth. Used only to contextualise a measurement,
# never to compute one. If your card is missing, `--peak-bw` overrides.
PEAK_BW_GB_S = {
    "NVIDIA GeForce RTX 4090": 1008.0,
    "NVIDIA L40S": 864.0,
    "NVIDIA RTX A6000": 768.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA H100 80GB HBM3": 3350.0,
    "NVIDIA H100 SXM5": 3350.0,
    # H100 NVL: 2-die HBM3 SKU, 3.9 TB/s per NVIDIA's datasheet. This exact
    # string is what torch.cuda.get_device_name() returns for it -- it was
    # missing from this table on the first run, which silently fell back to
    # the generic 1000.0 GB/s default and made the SwiGLU kernel read as
    # 178-268% of "peak", a number that cannot be real. The lesson: a
    # missing dict entry here doesn't error, it lies plausibly. Verify
    # `report["device"]` against this table any time a percentage exceeds
    # ~95%.
    "NVIDIA H100 NVL": 3938.0,
}


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_adaln(b, t, n, dtype, device, peak_bw):
    mk = lambda *s: torch.randn(*s, device=device, dtype=dtype, requires_grad=True)
    args = {
        "x": mk(b * t, n), "y": mk(b * t, n), "gate": mk(b, n),
        "weight": mk(n), "scale": mk(b, n), "shift": mk(b, n),
    }
    dh = torch.randn(b * t, n, device=device, dtype=dtype)
    dout = torch.randn(b * t, n, device=device, dtype=dtype)

    def run(fn):
        def inner():
            for v in args.values():
                v.grad = None
            h, o = fn(**args, tokens_per_sample=t)
            (h * dh + o * dout).sum().backward()
        return inner

    ref_ms = timeit(run(adaln_residual_rmsnorm_ref))
    fused_ms = timeit(run(fused_adaln_residual_rmsnorm))

    # Ideal traffic, forward + backward: read x,y + write h,out, then in backward
    # read dh,dout,x,y,h + write dx,dy. 11 tensor-sized transfers is the floor.
    elem = torch.finfo(dtype).bits // 8
    moved = 11 * b * t * n * elem
    bw = bandwidth_gb_s(moved, fused_ms)

    return {
        "shape": f"B{b}xT{t}xN{n}",
        "dtype": str(dtype).split(".")[-1],
        "eager_ms": ref_ms,
        "fused_ms": fused_ms,
        "speedup": ref_ms / fused_ms,
        "achieved_gb_s": bw,
        "roofline": roofline_note(bw, peak_bw),
    }


def bench_swiglu(m, n, dtype, device, peak_bw):
    a = torch.randn(m, n, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(m, n, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(m, n, device=device, dtype=dtype)

    def run(fn):
        def inner():
            a.grad = b.grad = None
            fn(a, b).backward(dout)
        return inner

    ref_ms = timeit(run(swiglu_ref))
    fused_ms = timeit(run(fused_swiglu))
    elem = torch.finfo(dtype).bits // 8
    # forward:  read a, b            write out         -> 3 tensor-equivalents
    # backward: read dout, a, b      write da, db      -> 5 tensor-equivalents
    # timeit() runs fwd+bwd together each iteration, so both legs count.
    # (Earlier version said "6" and only counted forward once plus a
    # miscounted backward -- that undercount is why the first run of this
    # benchmark reported >100% of peak bandwidth, which is physically
    # impossible and was the signal something here was wrong, not the
    # kernel being unrealistically fast.)
    moved = 8 * m * n * elem
    bw = bandwidth_gb_s(moved, fused_ms)

    return {
        "shape": f"{m}x{n}",
        "dtype": str(dtype).split(".")[-1],
        "eager_ms": ref_ms,
        "fused_ms": fused_ms,
        "speedup": ref_ms / fused_ms,
        "achieved_gb_s": bw,
        "roofline": roofline_note(bw, peak_bw),
    }


def bench_ce(m, h, v, dtype, device):
    """Cross entropy is judged on memory, not latency. Chunking trades a little
    time for a lot of headroom, and headroom converts to batch size."""
    hid = torch.randn(m, h, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(v, h, device=device, dtype=dtype, requires_grad=True)
    tgt = torch.randint(0, v, (m,), device=device)
    result = {"shape": f"M{m}xH{h}xV{v}", "dtype": str(dtype).split(".")[-1]}

    for name, fn in [
        ("naive", lambda: chunked_cross_entropy_ref(hid, w, tgt)),
        ("chunked", lambda: chunked_cross_entropy(hid, w, tgt, chunk_size=1024)),
    ]:
        hid.grad = w.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            ms = timeit(lambda: fn().backward(), iters=10, warmup=3)  # noqa: B023
            # -- fn is captured and called synchronously within this same
            # loop iteration (inside timeit, before the next iteration
            # rebinds it), so there is no late-binding closure bug here.
            result[f"{name}_ms"] = ms
            result[f"{name}_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        except torch.cuda.OutOfMemoryError:
            result[f"{name}_ms"] = None
            result[f"{name}_peak_mib"] = None
            result[f"{name}_oom"] = True

    if result.get("naive_peak_mib") and result.get("chunked_peak_mib"):
        result["memory_reduction"] = result["naive_peak_mib"] / result["chunked_peak_mib"]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/kernels.json")
    ap.add_argument("--peak-bw", type=float, default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    name = torch.cuda.get_device_name(0)
    peak = args.peak_bw or PEAK_BW_GB_S.get(name, 1000.0)
    dtype = getattr(torch, args.dtype)
    dev = "cuda"

    print(f"device: {name}   assumed peak bandwidth: {peak:.0f} GB/s\n")

    report = {"device": name, "peak_bw_gb_s": peak, "dtype": args.dtype,
              "adaln": [], "swiglu": [], "cross_entropy": []}

    print("=== fused adaLN-Zero + gated residual + RMSNorm ===")
    for b, t, n in [(8, 32, 384), (8, 128, 384), (16, 256, 768), (8, 512, 1024), (4, 1024, 1536)]:
        r = bench_adaln(b, t, n, dtype, dev, peak)
        report["adaln"].append(r)
        print(f"  {r['shape']:<22} eager {r['eager_ms']:7.3f} ms | "
              f"fused {r['fused_ms']:7.3f} ms | {r['speedup']:5.2f}x | {r['roofline']}")

    print("\n=== fused SwiGLU ===")
    for m, n in [(8192, 1536), (16384, 1536), (32768, 3072)]:
        r = bench_swiglu(m, n, dtype, dev, peak)
        report["swiglu"].append(r)
        print(f"  {r['shape']:<22} eager {r['eager_ms']:7.3f} ms | "
              f"fused {r['fused_ms']:7.3f} ms | {r['speedup']:5.2f}x | {r['roofline']}")

    print("\n=== chunked cross entropy (memory) ===")
    for m, h, v in [(4096, 384, 32000), (8192, 384, 32000), (16384, 384, 32000)]:
        r = bench_ce(m, h, v, dtype, dev)
        report["cross_entropy"].append(r)
        red = r.get("memory_reduction")
        naive = "OOM" if r.get("naive_oom") else f"{r.get('naive_peak_mib', 0):.0f} MiB"
        chunked = f"{r.get('chunked_peak_mib', 0):.0f} MiB"
        tail = f"{red:.2f}x less" if red else "(naive OOMed -- that is the result)"
        print(f"  {r['shape']:<22} naive {naive:>10} | chunked {chunked:>10} | {tail}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
