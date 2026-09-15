# nuvla-forge

**Training a VLA-style autonomous driving planner faster on two consumer GPUs.**

Custom Triton kernels, a profiled distributed training pipeline, and a data
loader rebuilt around what multi-view camera data actually costs — measured end
to end on 2× RTX 4090, a box with no NVLink and 24 GB per card.

---

## Why this model, why this hardware

Motional's nuVLA architecture takes multi-view, multi-frame camera images plus a
driving instruction, encodes them with a VLM backbone, and feeds the hidden
states into a **trajectory DiT** that generates the future ego plan. Reasoning
supervision trains the backbone; both modules optimise jointly.

That architecture has a specific, exploitable performance shape:

- **The DiT is conditioned with adaLN-Zero.** Six elementwise scale/shift/gate
  operations wrap every normalisation. In eager PyTorch the seam between two
  sub-blocks is roughly nine separate kernel launches, each making a full round
  trip to global memory, for arithmetic that needs two reads and two writes. The
  block is bandwidth bound, so those round trips *are* the cost.
- **Six camera views per sample.** At batch 8 that is 48 JPEG decodes per step.
  A naive loader cannot feed the GPU, and the profiler shows it as dead air
  before every step.
- **The reasoning head projects to a 32k vocabulary.** On a 24 GB card the logit
  tensor and its gradient are the allocation that decides your batch size.

And the hardware is the point, not an apology. Consumer Ada cards have no NVLink
and no peer-to-peer over PCIe, so NCCL stages every all-reduce through host
memory. Communication is a *visible* fraction of step time here. The fixes —
larger DDP buckets, suppressing the all-reduce on accumulation micro-steps,
overlapping reduction with backward, reducing in bf16 — are worth several times
more on this box than on a DGX, and they are the same fixes that matter when you
outgrow a single node.

Motional names safety and **cost efficiency** as its two competitive advantages.
This repo is the cost-efficiency half, taken literally.

---

## What's in here

### Three fused Triton kernels, each with forward *and* backward

| kernel | what it fuses | why |
|---|---|---|
| `fused_adaln_residual_rmsnorm` | gated residual add → RMSNorm → adaLN-Zero modulate | the DiT block seam; ~9 memory round trips collapsed to 2 |
| `fused_swiglu` | `silu(a) * b`, fwd and bwd | gated-MLP tail; recomputes the sigmoid instead of stashing it |
| `chunked_cross_entropy` | tiled logits + in-place online-softmax gradient | deletes the `[M, V]` logit tensor *and* its gradient |

The adaLN kernel is the one that matters. Its backward avoids both atomics
(non-deterministic, slow under contention) and the locking scheme from the
Triton layer-norm tutorial. Instead the grid is `(B × programs_per_sample,)`:
each program owns one sample and a strided slice of its rows, accumulates its
four parameter partials in registers, and writes them once to a small scratch
buffer that PyTorch reduces with two `.sum()` calls. Deterministic, lock-free,
and the scratch is `B·P·N` floats rather than `M·N`.

Every kernel has a pure-PyTorch reference in `kernels/reference.py`. The
reference is the test oracle, the benchmark baseline, *and* the CPU fallback — so
the whole repo runs without a GPU, which is what makes CI meaningful.

**The gradient math is finite-difference verified.** All six gradients of the
adaLN kernel, both SwiGLU gradients, and both chunked-CE gradients agree with
central differences to better than 1e-9 relative error. See `tests/`.

### Distributed training

DDP and FSDP paths, bucket-size sweep, `no_sync` accumulation, and
`bench_comms.py` — an all-reduce benchmark that measures your actual
interconnect rather than assuming one. It reports bus bandwidth against the PCIe
ceiling, so a flat curve across message sizes tells you immediately that you are
host-staging.

### Data pipeline

Two loaders behind one interface. `build_naive_loader` is deliberately bad
(2 workers, PIL decode, no pinning) and is the baseline. `build_fast_loader`
adds tuned worker counts, prefetch depth, pinned memory, persistent workers,
WebDataset shards for sequential reads, and optional DALI GPU decode.
`CudaPrefetcher` overlaps the H2D copy of batch N+1 with the compute of batch N.

Images cross PCIe as `uint8` and are normalised on device — 4× less traffic than
shipping float32, for an elementwise op the GPU finishes in microseconds.

### Profiling

CUDA-event timing with per-phase breakdown (`data` / `fwd_bwd` / `optim`),
warmup exclusion, PyTorch Profiler traces on a proper `wait/warmup/active`
schedule, and a roofline helper that reports achieved bandwidth **as a fraction
of the card's spec sheet**. A kernel at 85% of peak is finished. One at 20% has
something structurally wrong. Reporting the ratio instead of raw GB/s is what
separates a benchmark from a brag.

---

## Data

| dataset | access | use |
|---|---|---|
| `synthetic` | none needed | smoke tests and step-time benchmarks before any download finishes |
| `nuscenes-qa-mini` | ungated, one `load_dataset` call | smallest real-data path: 6-view RGB + lidar + QA, day/night splits |
| `drivelm` | ungated HF download | DriveLM-nuScenes reasoning QA over real driving scenes |
| `nureasoning` | **gated** — [request access](https://huggingface.co/datasets/qixuewei/nuReasoning) | the target: Motional fleet data, 105+ h of long-tail scenarios, 247k human-verified reasoning annotations |

All four sit behind one adapter interface, so switching is a config flag and
nothing above `data/adapters.py` changes. Development runs on the ungated
datasets; nuReasoning activates when access lands.

**On trajectory labels:** DriveLM and nuScenes-QA ship reasoning QA over nuScenes
keyframes but not ego futures, so the trajectory head trains against
kinematically plausible synthetic waypoints unless real ones are available. That
flag is surfaced in every report. A throughput number is hardware truth
regardless of label quality; an L2 planning number is not, and this repo does not
report one it hasn't earned.

---

## Running it

```bash
git clone <this repo> && cd nuvla-forge
bash scripts/setup_vast.sh          # installs, probes hardware, runs the tests
```

Nothing else runs until `pytest tests/ -q` is green.

```bash
bash scripts/probe_hardware.sh      # topology, P2P matrix, driver, NCCL
bash scripts/run_all_benchmarks.sh  # ~10 min: kernels, comms, loader, A/B

# or piecewise
python -m nuvla_forge.bench.bench_kernels
torchrun --nproc_per_node=2 -m nuvla_forge.bench.bench_comms
python -m nuvla_forge.bench.bench_dataloader --dataset drivelm --data-root data/drivelm

torchrun --nproc_per_node=2 -m nuvla_forge.train --preset baseline  --steps 200 --profile
torchrun --nproc_per_node=2 -m nuvla_forge.train --preset optimised --steps 200 --profile
python -m nuvla_forge.bench.report   # writes RESULTS.md
```

Every optimisation is an independent flag, so you can bisect which one paid
rather than flipping nine things in one commit and crediting the kernels.

---

## Results

See **[RESULTS.md](RESULTS.md)**, generated from `reports/*.json` by
`nuvla_forge.bench.report`. The generator cannot invent a number: if a benchmark
did not run, its row says so rather than quietly disappearing.

The table is empty until you run the sweep on your own hardware. That is
intentional. Benchmark numbers that were not measured on a named, probed machine
are decoration.

---

## Layout

```
nuvla_forge/
  kernels/      adaln_rmsnorm · swiglu · chunked_ce · reference · utils
  model/        dit (adaLN-Zero blocks, rectified flow) · nuvla (full model)
  data/         adapters (4 datasets, 1 interface) · loader (naive vs fast)
  bench/        bench_kernels · bench_comms · bench_dataloader · report
  train.py      DDP/FSDP entrypoint, A/B presets
  profiling.py  CUDA-event timing, profiler schedule, roofline
tests/          finite-difference-verified kernel correctness
scripts/        setup_vast · probe_hardware · run_all_benchmarks
```

---

## Honest limitations

- **All benchmarks in `RESULTS.md` use the `synthetic` dataset**, not real
  driving data. A real-data run (`nuscenes-qa-mini`) was attempted on the
  rented H100 instance and hit a Vast.ai storage constraint: the container's
  writable overlay filesystem was capped at 16GB regardless of the disk size
  requested at instance creation (the requested disk maps to a separate,
  read-only-to-this-container mount on that particular image/template), and
  the PyTorch/CUDA/Triton environment alone consumes 8.2GB of that, leaving
  no room for a multi-gigabyte dataset download. This is an infrastructure
  provisioning issue, not a code limitation — the dataset adapter, loader,
  and training loop all work identically regardless of which dataset backs
  them, and swapping to `nuscenes-qa-mini` or `drivelm` is one flag
  (`--dataset nuscenes-qa-mini`) on an instance with correctly-mounted
  storage. Kernel correctness and microbenchmarks are dataset-independent
  and unaffected by this.
- The model is small by design. This repo is about step time, not leaderboard
  rank, and a model that converges overnight on two consumer cards makes the
  optimisation work legible.
- The tokenizer is a deterministic hash. Correct shapes and vocabulary
  distribution at zero CPU cost, so tokenisation never contaminates a dataloader
  measurement. Swap in a real one for accuracy work.
- DALI is optional. Without it the loader falls back to a tuned torch pipeline,
  which is slower but works everywhere.
- No planning accuracy is claimed. See the note on trajectory labels above.

## License

MIT.
