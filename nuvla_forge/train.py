"""Training entrypoint.

Run it twice -- once with everything off, once with everything on -- and the two
JSON reports are the README's headline table.

    torchrun --nproc_per_node=2 -m nuvla_forge.train --preset baseline
    torchrun --nproc_per_node=2 -m nuvla_forge.train --preset optimised

Every optimisation is an independent flag so you can bisect which one paid,
rather than shipping one commit that flips nine things at once and claiming the
whole speedup for the kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .data.adapters import SampleSpec, build_dataset
from .data.loader import (
    CudaPrefetcher,
    LoaderConfig,
    build_fast_loader,
    build_naive_loader,
    describe_environment,
    normalise_images,
)
from .model.nuvla import NuVLA, NuVLAConfig
from .profiling import StepTimer, make_profiler


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    return rank, world, local


def is_main(rank):
    return rank == 0


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["baseline", "optimised", "custom"], default="custom")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--data-root", default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=20,
                   help="excluded from timing; covers Triton JIT and cuDNN autotune")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)

    # the A/B switches
    p.add_argument("--fused-kernels", action="store_true")
    p.add_argument("--fast-loader", action="store_true")
    p.add_argument("--cuda-prefetch", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--ddp-bucket-mb", type=float, default=25.0)
    p.add_argument("--no-sync-accum", action="store_true",
                   help="skip the all-reduce on non-final accumulation micro-steps")
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--compile", action="store_true")

    p.add_argument("--profile", action="store_true")
    p.add_argument("--out", default="reports")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def apply_preset(args):
    if args.preset == "baseline":
        args.fused_kernels = False
        args.fast_loader = False
        args.cuda_prefetch = False
        args.bf16 = False
        args.no_sync_accum = False
        args.ddp_bucket_mb = 25.0
    elif args.preset == "optimised":
        args.fused_kernels = True
        args.fast_loader = True
        args.cuda_prefetch = True
        args.bf16 = True
        args.no_sync_accum = True
        args.ddp_bucket_mb = 100.0
    return args


def main():
    args = apply_preset(build_args())
    rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    tag = args.tag or args.preset

    if not args.fused_kernels:
        os.environ["NUVLA_FORGE_DISABLE_TRITON"] = "1"

    torch.manual_seed(1234 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    spec = SampleSpec()
    ds_kwargs = {"spec": spec}
    if args.data_root:
        ds_kwargs["root"] = args.data_root
    dataset = build_dataset(args.dataset, **ds_kwargs)

    lcfg = LoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    sampler = DistributedSampler(dataset) if world > 1 else None
    build = build_fast_loader if args.fast_loader else build_naive_loader
    loader = build(dataset, lcfg, sampler=sampler)
    if args.cuda_prefetch and device.type == "cuda":
        loader = CudaPrefetcher(loader, device)

    mcfg = NuVLAConfig(fused=args.fused_kernels, horizon=spec.horizon)
    model = NuVLA(mcfg).to(device)

    if args.grad_checkpoint:
        # Recompute the vision blocks in backward instead of storing their
        # activations. On 24 GB cards this is what buys batch size; it costs
        # roughly one extra forward pass through the encoder.
        from torch.utils.checkpoint import checkpoint

        def _wrap(block):
            inner = block.forward
            def fwd(*a, **kw):
                return checkpoint(inner, *a, use_reentrant=False, **kw)
            return fwd

        for blk in model.encoder.blocks:
            blk.forward = _wrap(blk)

    if args.compile:
        model = torch.compile(model)

    if world > 1:
        if args.fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            model = FSDP(model, device_id=local)
        else:
            model = DDP(
                model,
                device_ids=[local],
                bucket_cap_mb=args.ddp_bucket_mb,
                gradient_as_bucket_view=True,
            )

    # fused AdamW is a CUDA-only path; fall back cleanly so CPU smoke tests run.
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
        fused=(device.type == "cuda"),
    )
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    autocast = (
        torch.autocast("cuda", dtype=amp_dtype)
        if (args.bf16 and device.type == "cuda")
        else nullcontext()
    )

    timer = StepTimer(warmup=args.warmup_steps, device=device)
    prof = make_profiler(Path(args.out) / f"trace-{tag}-rank{rank}") if args.profile else nullcontext()

    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{tag}] {n_params/1e6:.1f}M params | world={world} | device={device}")
        print(f"[{tag}] env={describe_environment()}")

    step, done = 0, False
    with prof as p:
        while not done:
            if sampler is not None:
                sampler.set_epoch(step)
            for batch in loader:
                if step >= args.steps + args.warmup_steps:
                    done = True
                    break

                timer.start()
                batch = {
                    k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                images = normalise_images(batch["images"], dtype=amp_dtype)
                timer.mark("data")

                micro = args.grad_accum
                for m in range(micro):
                    # Suppressing the all-reduce on all but the last micro-step is
                    # the single cheapest comms win available, and on a box with
                    # no NVLink it is worth far more than it is on one with.
                    sync_ctx = (
                        model.no_sync()
                        if (args.no_sync_accum and world > 1 and not args.fsdp and m < micro - 1)
                        else nullcontext()
                    )
                    with sync_ctx, autocast:
                        out = model(
                            images,
                            trajectory=batch["trajectory"].to(amp_dtype),
                            text_tokens=batch["text_tokens"],
                            text_targets=batch["text_targets"],
                        )
                        loss = out["loss"] / micro
                    loss.backward()
                timer.mark("fwd_bwd")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                timer.mark("optim")
                timer.stop()

                if args.profile:
                    p.step()
                if is_main(rank) and step % 25 == 0:
                    print(f"  step {step:5d}  loss {out['loss'].item():.4f}  "
                          f"{timer.last_ms():.1f} ms")
                step += 1

    stats = timer.summary()
    stats.update(
        tag=tag,
        world_size=world,
        batch_size=args.batch_size,
        global_batch=args.batch_size * world * args.grad_accum,
        samples_per_sec=args.batch_size * world * args.grad_accum / (stats["step_ms"] / 1000),
        peak_mem_mib=(
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0
        ),
        flags={k: v for k, v in vars(args).items() if isinstance(v, (bool, int, float, str))},
        model=asdict(mcfg),
    )

    if is_main(rank):
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"train-{tag}.json"
        path.write_text(json.dumps(stats, indent=2, default=str))
        print(f"\n[{tag}] step {stats['step_ms']:.2f} ms | "
              f"{stats['samples_per_sec']:.1f} samples/s | "
              f"peak {stats['peak_mem_mib']:.0f} MiB")
        print(f"[{tag}] wrote {path}")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
