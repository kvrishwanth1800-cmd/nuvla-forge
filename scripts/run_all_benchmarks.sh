#!/usr/bin/env bash
# Full benchmark sweep. ~10 minutes on 2x consumer GPUs.
set -euo pipefail
mkdir -p reports

echo "### hardware"
bash scripts/probe_hardware.sh reports/hardware.txt >/dev/null

echo "### kernel correctness"
pytest tests/ -q

echo "### kernel microbenchmarks"
python -m nuvla_forge.bench.bench_kernels --out reports/kernels.json

echo "### dataloader"
python -m nuvla_forge.bench.bench_dataloader --dataset "${DATASET:-synthetic}" --out reports/dataloader.json

NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
if [ "$NGPU" -gt 1 ]; then
  echo "### collective communication ($NGPU GPUs)"
  torchrun --nproc_per_node="$NGPU" -m nuvla_forge.bench.bench_comms --out reports/comms.json
else
  echo "### comms skipped (single GPU)"
fi

echo "### end-to-end A/B"
torchrun --nproc_per_node="$NGPU" -m nuvla_forge.train --preset baseline  --steps 150 --profile
torchrun --nproc_per_node="$NGPU" -m nuvla_forge.train --preset optimised --steps 150 --profile

python -m nuvla_forge.bench.report
echo "done -- see reports/ and RESULTS.md"
