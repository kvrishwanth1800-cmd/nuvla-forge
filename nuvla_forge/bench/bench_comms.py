"""Communication benchmark for the interconnect you actually have.

    torchrun --nproc_per_node=2 -m nuvla_forge.bench.bench_comms

Why this gets its own benchmark
-------------------------------
Consumer Ada cards (RTX 4090 and friends) have no NVLink, and NVIDIA does not
expose peer-to-peer access between them over PCIe in the stock driver. NCCL
therefore cannot DMA device-to-device; it stages through host memory. An
all-reduce that would cost a few hundred microseconds on an NVLink pair costs
several times that here, and on a model whose gradients are tens of megabytes
that is a visible fraction of step time.

This is not a reason to apologise for the hardware. It is a reason to measure it,
because the fixes -- larger DDP buckets, suppressing the all-reduce on
accumulation micro-steps, overlapping the reduction with backward, reducing in
bf16 -- are worth several times more here than they are on a DGX, and they are
the same fixes that matter at scale when you outgrow a single node.

Run ``scripts/probe_hardware.sh`` first. If ``nvidia-smi topo -m`` reports PHB or
SYS between your GPUs and the P2P test shows no direct path, the numbers below
are your ceiling and the optimisations are your lever.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def timed_allreduce(numel, dtype, iters=50, warmup=10):
    x = torch.randn(numel, device="cuda", dtype=dtype)
    for _ in range(warmup):
        dist.all_reduce(x)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        dist.all_reduce(x)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def bus_bandwidth_gb_s(numel, elem_bytes, ms, world):
    """Ring all-reduce algorithmic bandwidth: 2(N-1)/N bytes per rank."""
    total = numel * elem_bytes
    factor = 2.0 * (world - 1) / world
    return total * factor / (ms * 1e-3) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/comms.json")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)

    if rank == 0:
        print(f"world={world}  device={torch.cuda.get_device_name(local)}")
        print(f"{'bytes':>12} {'fp32 ms':>10} {'bf16 ms':>10} "
              f"{'fp32 GB/s':>11} {'bf16 GB/s':>11}  {'bf16 saving':>11}")

    rows = []
    for mb in [0.25, 1, 4, 16, 64, 256]:
        numel32 = int(mb * 2**20 // 4)
        ms32 = timed_allreduce(numel32, torch.float32)
        ms16 = timed_allreduce(numel32, torch.bfloat16)   # same count, half the bytes
        bw32 = bus_bandwidth_gb_s(numel32, 4, ms32, world)
        bw16 = bus_bandwidth_gb_s(numel32, 2, ms16, world)

        rows.append(dict(size_mb=mb, fp32_ms=ms32, bf16_ms=ms16,
                         fp32_gb_s=bw32, bf16_gb_s=bw16, bf16_speedup=ms32 / ms16))
        if rank == 0:
            print(f"{mb:12.2f} {ms32:10.3f} {ms16:10.3f} "
                  f"{bw32:11.1f} {bw16:11.1f}  {ms32/ms16:10.2f}x")

    if rank == 0:
        report = {
            "world_size": world,
            "device": torch.cuda.get_device_name(local),
            "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
            "rows": rows,
            "note": (
                "Compare fp32 GB/s against the PCIe generation's practical ceiling "
                "(~25 GB/s for gen4 x16 one way). Landing well under it with a "
                "flat curve across sizes points at host staging, i.e. no P2P."
            ),
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
