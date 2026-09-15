"""Turn ``reports/*.json`` into ``RESULTS.md``.

Deliberately dumb: it reads what the benchmarks wrote and formats it. It cannot
invent a number, and if a benchmark did not run its row says so rather than
quietly disappearing. A results table with a visible gap is honest; one that
silently drops its failures is not.
"""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path("reports")


def load(name):
    p = REPORTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def fmt(v, spec=".2f", missing="—"):
    if v is None:
        return missing
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def section_hardware(lines):
    hw = REPORTS / "hardware.txt"
    lines += ["## Hardware", ""]
    if hw.exists():
        lines += ["```", hw.read_text().strip()[:2500], "```", ""]
    else:
        lines += ["_Not captured. Run `bash scripts/probe_hardware.sh`._", ""]


def section_kernels(lines):
    k = load("kernels.json")
    lines += ["## Kernel microbenchmarks", ""]
    if not k:
        lines += ["_Not run._", ""]
        return

    lines += [f"Device: `{k['device']}` · assumed peak bandwidth "
              f"{k['peak_bw_gb_s']:.0f} GB/s · dtype `{k['dtype']}`", ""]

    lines += ["### Fused adaLN-Zero + gated residual + RMSNorm (fwd+bwd)", "",
              "| shape | eager ms | fused ms | speedup | achieved GB/s | % of peak |",
              "|---|---|---|---|---|---|"]
    for r in k.get("adaln", []):
        pct = 100 * r["achieved_gb_s"] / k["peak_bw_gb_s"]
        lines.append(f"| `{r['shape']}` | {fmt(r['eager_ms'],'.3f')} | "
                     f"{fmt(r['fused_ms'],'.3f')} | **{fmt(r['speedup'])}x** | "
                     f"{fmt(r['achieved_gb_s'],'.0f')} | {fmt(pct,'.0f')}% |")

    lines += ["", "### Fused SwiGLU (fwd+bwd)", "",
              "| shape | eager ms | fused ms | speedup | achieved GB/s | % of peak |",
              "|---|---|---|---|---|---|"]
    for r in k.get("swiglu", []):
        pct = 100 * r["achieved_gb_s"] / k["peak_bw_gb_s"]
        lines.append(f"| `{r['shape']}` | {fmt(r['eager_ms'],'.3f')} | "
                     f"{fmt(r['fused_ms'],'.3f')} | **{fmt(r['speedup'])}x** | "
                     f"{fmt(r['achieved_gb_s'],'.0f')} | {fmt(pct,'.0f')}% |")

    lines += ["", "### Chunked cross entropy (peak memory)", "",
              "| shape | naive peak | chunked peak | reduction |", "|---|---|---|---|"]
    for r in k.get("cross_entropy", []):
        naive = "OOM" if r.get("naive_oom") else f"{fmt(r.get('naive_peak_mib'),'.0f')} MiB"
        lines.append(f"| `{r['shape']}` | {naive} | "
                     f"{fmt(r.get('chunked_peak_mib'),'.0f')} MiB | "
                     f"**{fmt(r.get('memory_reduction'))}x** |")
    lines.append("")


def section_comms(lines):
    c = load("comms.json")
    lines += ["## Collective communication", ""]
    if not c:
        lines += ["_Not run (single GPU, or `bench_comms` skipped)._", ""]
        return
    lines += [f"{c['world_size']} x `{c['device']}` · NCCL {c.get('nccl_version','?')}", "",
              "| all-reduce size | fp32 ms | bf16 ms | fp32 GB/s | bf16 speedup |",
              "|---|---|---|---|---|"]
    for r in c["rows"]:
        lines.append(f"| {r['size_mb']:g} MB | {fmt(r['fp32_ms'],'.3f')} | "
                     f"{fmt(r['bf16_ms'],'.3f')} | {fmt(r['fp32_gb_s'],'.1f')} | "
                     f"{fmt(r['bf16_speedup'])}x |")
    lines += ["", f"> {c.get('note','')}", ""]


def section_dataloader(lines):
    d = load("dataloader.json")
    lines += ["## Data pipeline", ""]
    if not d:
        lines += ["_Not run._", ""]
        return
    lines += [f"Dataset: `{d['dataset']}` · {d['env']['cpu_count']} CPUs · "
              f"DALI {'available' if d['env']['has_dali'] else 'absent'}", "",
              "| configuration | loader-only samples/s | GPU idle (in training loop) |",
              "|---|---|---|"]
    for r in d["configs"]:
        if "error" in r:
            lines.append(f"| {r['config']} | failed | — |")
        else:
            lines.append(f"| {r['config']} | {fmt(r['samples_per_sec'],'.1f')} | "
                         f"{fmt(r.get('gpu_idle_pct'),'.1f')}% |")
    lines += [
        "",
        "> **GPU idle** (measured with a real forward+backward pass in the loop) "
        "is the column that matters; above ~10% means the model is waiting on "
        "data and no kernel work will fix that.",
        "",
        "> **Loader-only samples/s** measures the loader alone, no compute in "
        "between batches. Note this column is *not* meaningful for the CUDA-"
        "prefetch rows: the prefetcher's entire purpose is overlapping the H2D "
        "copy with GPU compute, and with no compute here to overlap with, it "
        "can only add stream-management overhead for zero benefit. Judge the "
        "prefetcher's effect from the GPU-idle column instead, where it is "
        "actually exercised doing its job.",
        "",
    ]


def section_end_to_end(lines):
    base, opt = load("train-baseline.json"), load("train-optimised.json")
    lines += ["## End-to-end training step", ""]
    if not (base and opt):
        lines += ["_Both presets must run. "
                  "`torchrun --nproc_per_node=2 -m nuvla_forge.train --preset baseline|optimised`_", ""]
        return

    lines += [f"{base['world_size']} GPUs · global batch {base['global_batch']} · "
              f"{base['n_steps']} timed steps after warmup", "",
              "| metric | baseline | optimised | change |", "|---|---|---|---|"]

    rows = [
        ("step time (median)", "step_ms", "ms", True),
        ("samples / sec", "samples_per_sec", "", False),
        ("peak memory", "peak_mem_mib", "MiB", True),
        ("data phase", "data_ms", "ms", True),
        ("fwd+bwd phase", "fwd_bwd_ms", "ms", True),
        ("optimiser phase", "optim_ms", "ms", True),
    ]
    for label, key, unit, lower_better in rows:
        b, o = base.get(key), opt.get(key)
        if b is None or o is None:
            continue
        delta = (f"**{b/o:.2f}x faster**" if lower_better else f"**{o/b:.2f}x more**")
        lines.append(f"| {label} | {fmt(b)} {unit} | {fmt(o)} {unit} | {delta} |")
    lines.append("")


def main():
    lines = ["# Results", "",
             "Generated by `python -m nuvla_forge.bench.report` from the JSON in "
             "`reports/`. Every number here was measured on the hardware described "
             "below. Nothing is copied from a paper or estimated.", ""]
    section_hardware(lines)
    section_kernels(lines)
    section_comms(lines)
    section_dataloader(lines)
    section_end_to_end(lines)

    out = Path("RESULTS.md")
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
