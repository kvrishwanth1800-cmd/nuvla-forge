"""Dataloader throughput, isolated from the model.

    python -m nuvla_forge.bench.bench_dataloader --dataset drivelm --data-root data/drivelm

Measures the loader alone (samples/sec with no model attached) and then the
loader in situ (GPU idle fraction during a real step). Both matter and they
answer different questions: the first tells you the ceiling, the second tells you
whether you are hitting it.

The number to watch is `gpu_idle_pct`. Above ~10% you have a data problem, and
no amount of kernel work will help -- the GPU is already waiting.
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch

from ..data.adapters import SampleSpec, build_dataset
from ..data.loader import (CudaPrefetcher, LoaderConfig, build_fast_loader,
                           build_naive_loader, describe_environment, normalise_images)


def measure_loader(loader, n_batches, device, prefetch=False):
    if prefetch and device.type == "cuda":
        loader = CudaPrefetcher(loader, device)
    it = iter(loader)
    for _ in range(5):                      # warm the workers
        next(it, None)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    seen = 0
    for _ in range(n_batches):
        batch = next(it, None)
        if batch is None:
            break
        seen += batch["images"].shape[0]
    torch.cuda.synchronize() if device.type == "cuda" else None
    dt = time.perf_counter() - t0
    return dict(samples_per_sec=seen / dt, batches=n_batches, seconds=dt)


def measure_idle(loader, device, n_batches=30, prefetch=False):
    """GPU idle fraction: total wall time minus time the GPU was busy."""
    from ..model.nuvla import NuVLA, NuVLAConfig
    model = NuVLA(NuVLAConfig(fused=True)).to(device)
    if prefetch and device.type == "cuda":
        loader = CudaPrefetcher(loader, device)

    it = iter(loader)
    busy_ms, wall0 = 0.0, None
    for i in range(n_batches):
        batch = next(it, None)
        if batch is None:
            break
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        if i == 5:
            torch.cuda.synchronize(); wall0 = time.perf_counter(); busy_ms = 0.0
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        out = model(normalise_images(batch["images"]),
                    trajectory=batch["trajectory"],
                    text_tokens=batch["text_tokens"],
                    text_targets=batch["text_targets"])
        out["loss"].backward()
        e.record(); torch.cuda.synchronize()
        if wall0 is not None:
            busy_ms += s.elapsed_time(e)
    wall_ms = (time.perf_counter() - wall0) * 1000
    return dict(wall_ms=wall_ms, gpu_busy_ms=busy_ms,
                gpu_idle_pct=100.0 * (1 - busy_ms / wall_ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--out", default="reports/dataloader.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = {"spec": SampleSpec()}
    if args.data_root:
        kw["root"] = args.data_root
    ds = build_dataset(args.dataset, **kw)
    print(f"dataset {args.dataset}: {len(ds)} samples | env {describe_environment()}\n")

    report = {"dataset": args.dataset, "env": describe_environment(), "configs": []}

    configs = [
        ("naive (2 workers, no pin)", build_naive_loader, dict(num_workers=2), False),
        ("tuned (8 workers, pinned)", build_fast_loader, dict(num_workers=8, prefetch_factor=4), False),
        ("tuned + cuda prefetch",     build_fast_loader, dict(num_workers=8, prefetch_factor=4), True),
        ("tuned + 16 workers",        build_fast_loader, dict(num_workers=16, prefetch_factor=6), True),
    ]

    print(f"{'config':<30} {'samples/s':>11} {'gpu idle':>10}")
    for name, build, over, prefetch in configs:
        cfg = LoaderConfig(batch_size=args.batch_size, **over)
        try:
            thr = measure_loader(build(ds, cfg), args.batches, device, prefetch)
            idle = (measure_idle(build(ds, cfg), device, prefetch=prefetch)
                    if device.type == "cuda" else {"gpu_idle_pct": float("nan")})
            row = dict(config=name, **thr, **idle)
            print(f"{name:<30} {thr['samples_per_sec']:11.1f} {idle['gpu_idle_pct']:9.1f}%")
        except Exception as exc:
            row = dict(config=name, error=str(exc))
            print(f"{name:<30} failed: {exc}")
        report["configs"].append(row)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
