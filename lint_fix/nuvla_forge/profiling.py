"""Timing and profiling.

Two rules this module exists to enforce.

**Time with CUDA events, not ``time.time()``.** Kernel launches are asynchronous.
A wall-clock timer around a launch measures how long it took to *queue* the work,
which on a fast host is near zero and on a slow one is noise. Events are recorded
in the stream and measure the GPU.

**Throw away the warmup.** The first Triton call JIT-compiles, cuDNN autotunes its
algorithm on first sight of a shape, and the caching allocator has not reached
steady state. Including those steps understates every optimisation, sometimes by
enough to invert the comparison.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


class StepTimer:
    """Per-step and per-phase timing with CUDA events.

    Phases nest inside one step:  data -> fwd_bwd -> optim.  The phase split is
    what tells you whether you have a dataloader problem or a kernel problem, and
    they are different problems with different fixes.
    """

    def __init__(self, warmup: int = 20, device=None):
        self.warmup = warmup
        self.device = device
        self.cuda = torch.cuda.is_available() and (device is None or device.type == "cuda")
        self.n = 0
        self.steps: list[float] = []
        self.phases: dict[str, list[float]] = {}
        self._events: list[tuple[str, torch.cuda.Event]] = []
        self._last = 0.0

    def start(self):
        self.n += 1
        self._events = []
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._events.append(("__start__", e))

    def mark(self, name: str):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._events.append((name, e))

    def stop(self):
        if not self.cuda or not self._events:
            return
        torch.cuda.synchronize(self.device)
        if self.n <= self.warmup:
            return

        start_evt = self._events[0][1]
        prev = start_evt
        for name, evt in self._events[1:]:
            self.phases.setdefault(name, []).append(prev.elapsed_time(evt))
            prev = evt
        total = start_evt.elapsed_time(self._events[-1][1])
        self.steps.append(total)
        self._last = total

    def last_ms(self) -> float:
        return self._last

    def summary(self) -> dict:
        if not self.steps:
            return {"step_ms": float("nan"), "n_steps": 0}
        out = {
            "step_ms": statistics.median(self.steps),
            "step_ms_mean": statistics.fmean(self.steps),
            "step_ms_p10": _pct(self.steps, 10),
            "step_ms_p90": _pct(self.steps, 90),
            "n_steps": len(self.steps),
        }
        for name, vals in self.phases.items():
            out[f"{name}_ms"] = statistics.median(vals)
            out[f"{name}_pct"] = 100.0 * statistics.median(vals) / out["step_ms"]
        return out


def _pct(values, p):
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


def make_profiler(out_dir: str | Path, wait: int = 10, warmup: int = 10, active: int = 10):
    """PyTorch Profiler configured for a training loop.

    The schedule matters. Profiling every step produces a trace so large it is
    unopenable and so distorted it is useless. Wait, then warm up, then capture a
    short active window in steady state.

    Open the result with ``chrome://tracing`` or TensorBoard. The thing to look
    for first is gaps on the GPU stream -- that is the dataloader starving you,
    and no kernel work will fix it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(out_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,   # stacks balloon the trace; turn on only when hunting
    )


def bandwidth_gb_s(bytes_moved: int, ms: float) -> float:
    return bytes_moved / (ms * 1e-3) / 1e9


def roofline_note(achieved_gb_s: float, peak_gb_s: float) -> str:
    """Contextualise a bandwidth number.

    A kernel at 85% of peak HBM bandwidth is finished -- further work belongs
    elsewhere. A kernel at 20% has something structurally wrong (uncoalesced
    access, too few programs in flight, register spills). Reporting the ratio
    instead of the raw GB/s is what separates a benchmark from a brag.
    """
    pct = 100.0 * achieved_gb_s / peak_gb_s
    if pct > 80:
        verdict = "bandwidth-saturated; no headroom left in this kernel"
    elif pct > 50:
        verdict = "healthy; remaining gap is launch overhead and tails"
    elif pct > 25:
        verdict = "underutilised; check occupancy and coalescing"
    else:
        verdict = "poor; likely uncoalesced or occupancy-starved"
    return f"{achieved_gb_s:.0f} GB/s = {pct:.0f}% of {peak_gb_s:.0f} GB/s peak -- {verdict}"
