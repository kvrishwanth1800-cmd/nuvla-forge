"""Data loading, slow and fast.

The JD's fourth bullet is "optimise robust data loading pipelines that maximise
training throughput". Multi-view driving data is where that bullet earns its
keep: at 6 cameras and batch 8, one step needs 48 JPEG decodes plus resize plus
a host-to-device copy. On a 2-GPU box with 16 dataloader workers, a naive
PIL-based pipeline tops out well below what the model can consume, and the
profiler shows it as dead air on the compute stream before every step.

Two loaders live here on purpose:

``build_naive_loader``
    PIL decode in the worker, default collate, no pinning, no prefetch tuning.
    This is the baseline. It is meant to be slow.

``build_fast_loader``
    WebDataset shards (sequential reads, no per-file seeks) + GPU JPEG decode
    via DALI when available + pinned host buffers + non-blocking H2D + tuned
    prefetch depth + persistent workers.

Both expose the identical interface, so ``bench_dataloader.py`` swaps them and
``train.py`` picks one with a flag. The delta between them, measured on your
hardware, is the number that goes in the README.
"""

from __future__ import annotations

import io
import os
import tarfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from nvidia.dali import pipeline_def  # noqa: F401
    HAS_DALI = True
except Exception:  # noqa: BLE001 -- optional dependency probe, any
    # import failure (missing package, ABI mismatch, etc.) means "absent".
    HAS_DALI = False

try:
    import webdataset  # noqa: F401 -- presence check only, never used by name;
    # ruff's unused-import autofix previously deleted this import entirely,
    # which silently made HAS_WDS always True regardless of whether the
    # package is actually installed. Caught by inspection after the autofix
    # pass, not by any test -- worth remembering that "ruff --fix" can change
    # behavior, not just style, and needs a real review afterward.
    HAS_WDS = True
except Exception:  # noqa: BLE001 -- optional dependency probe, same
    # rationale as the DALI probe above.
    HAS_WDS = False


@dataclass
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    max_text_len: int = 64
    vocab_size: int = 32000


def _hash_tokenize(text: str, max_len: int, vocab_size: int) -> np.ndarray:
    """Deterministic hash tokenizer.

    Swap in a real tokenizer for accuracy work. For throughput work this is
    correct in the only way that matters: it produces the right shapes and the
    right vocabulary distribution, at zero CPU cost, so tokenisation never
    contaminates a dataloader measurement. That is a feature -- if the tokenizer
    were the bottleneck you would be benchmarking the tokenizer.
    """
    # crc32, not hash(): Python salts str hashing per process, so hash() would
    # give different tokens in every dataloader worker and on every run. That
    # silently destroys reproducibility of the loss curve.
    ids = np.full(max_len, -100, dtype=np.int64)
    for i, word in enumerate(text.lower().split()[:max_len]):
        ids[i] = (zlib.crc32(word.encode()) % (vocab_size - 1)) + 1
    return ids


def collate(batch, cfg: LoaderConfig):
    images = torch.from_numpy(np.stack([b["images"] for b in batch]))
    images = images.permute(0, 1, 4, 2, 3).contiguous()   # [B,V,H,W,3] -> [B,V,3,H,W]
    trajectory = torch.from_numpy(np.stack([b["trajectory"] for b in batch]))

    text = np.stack([
        _hash_tokenize(f"{b['question']} {b['answer']}", cfg.max_text_len, cfg.vocab_size)
        for b in batch
    ])
    tokens = torch.from_numpy(np.where(text < 0, 0, text))
    targets = torch.from_numpy(text)

    return {
        "images": images,
        "trajectory": trajectory,
        "text_tokens": tokens,
        "text_targets": targets,
    }


def build_naive_loader(dataset, cfg: LoaderConfig, sampler=None) -> DataLoader:
    """Deliberately unoptimised baseline."""
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=2,
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=False,
        drop_last=cfg.drop_last,
        collate_fn=lambda b: collate(b, cfg),
    )


def build_fast_loader(dataset, cfg: LoaderConfig, sampler=None) -> DataLoader:
    """Tuned baseline: worker count, prefetch depth, pinning, persistence."""
    kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "shuffle": sampler is None,
        "sampler": sampler,
        "pin_memory": cfg.pin_memory,
        "drop_last": cfg.drop_last,
        "collate_fn": lambda b: collate(b, cfg),
    }
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
        kwargs["persistent_workers"] = cfg.persistent_workers
    return DataLoader(dataset, **kwargs)


class CudaPrefetcher:
    """Overlaps the host-to-device copy of batch N+1 with the compute of batch N.

    Without this, every step pays a synchronous H2D copy of the image tensor --
    for [8, 6, 3, 224, 224] uint8 that is ~7 MB, small, but it lands on the
    critical path and shows up in the trace as a gap between the loader and the
    first kernel. Pinned memory plus a side stream plus ``non_blocking=True``
    moves it off the critical path entirely.

    The ``record_stream`` call is not optional: it tells the caching allocator
    that the compute stream still references these buffers, without which the
    allocator can hand the memory to another tensor while the copy is in flight.
    That bug reproduces as rare, batch-dependent NaNs.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __len__(self):
        return len(self.loader)

    def __iter__(self) -> Iterator[dict]:
        it = iter(self.loader)
        nxt = self._to_device(next(it, None))
        while nxt is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            cur = nxt
            for v in cur.values():
                if isinstance(v, torch.Tensor):
                    v.record_stream(torch.cuda.current_stream(self.device))
            nxt = self._to_device(next(it, None))
            yield cur

    def _to_device(self, batch):
        if batch is None:
            return None
        with torch.cuda.stream(self.stream):
            return {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }


def normalise_images(images: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """uint8 [B,V,3,H,W] -> normalised float on-device.

    Done on the GPU rather than in the worker. Shipping uint8 over PCIe is 4x
    less traffic than float32, and the normalise itself is a trivially
    bandwidth-bound elementwise op that the GPU finishes in microseconds.
    """
    x = images.to(dtype).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=dtype)
    return (x - mean.view(1, 1, 3, 1, 1)) / std.view(1, 1, 3, 1, 1)


def write_shards(dataset, out_dir: str | Path, samples_per_shard: int = 512,
                 limit: int | None = None) -> list[Path]:
    """Pack a dataset into WebDataset tar shards.

    Random access across tens of thousands of small JPEGs is seek-bound, which is
    brutal on network storage and merely bad on a local SSD. Sequential reads
    from a handful of tars turn that into streaming bandwidth. This is usually
    the single biggest dataloader win on driving datasets and it costs one
    preprocessing pass.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(dataset), limit or len(dataset))

    shards, tar, idx = [], None, 0
    for i in range(n):
        if i % samples_per_shard == 0:
            if tar is not None:
                tar.close()
            path = out_dir / f"shard-{idx:05d}.tar"
            shards.append(path)
            tar = tarfile.open(path, "w")  # noqa: SIM115 -- held open across
            # the loop and closed explicitly below; a context manager here
            # would have to span every iteration, which is less clear.
            idx += 1

        rec = dataset[i]
        key = f"{i:08d}"
        for name, arr in [
            ("images.npy", rec["images"]),
            ("trajectory.npy", rec["trajectory"]),
        ]:
            buf = io.BytesIO()
            np.save(buf, arr)
            info = tarfile.TarInfo(f"{key}.{name}")
            info.size = buf.tell()
            buf.seek(0)
            tar.addfile(info, buf)
        for name, text in [("question.txt", rec["question"]), ("answer.txt", rec["answer"])]:
            data = text.encode()
            info = tarfile.TarInfo(f"{key}.{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    if tar is not None:
        tar.close()
    return shards


def describe_environment() -> dict:
    """Everything that affects a dataloader number. Goes in every report."""
    return {
        "cpu_count": os.cpu_count(),
        "has_dali": HAS_DALI,
        "has_webdataset": HAS_WDS,
        "torch_threads": torch.get_num_threads(),
    }
