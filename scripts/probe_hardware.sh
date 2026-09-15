#!/usr/bin/env bash
# Capture everything that a benchmark number depends on. Run this FIRST on a new
# box and paste the output into reports/hardware.txt -- every table in the README
# is meaningless without it.
set -uo pipefail
out="${1:-reports/hardware.txt}"
mkdir -p "$(dirname "$out")"

{
  echo "=== captured $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo; echo "--- GPUs ---"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv
  echo; echo "--- topology (look for NV# = NVLink, PHB/SYS = through the host) ---"
  nvidia-smi topo -m
  echo; echo "--- peer-to-peer access matrix ---"
  python3 - <<'PY'
import torch
n = torch.cuda.device_count()
print(f"visible GPUs: {n}")
for i in range(n):
    row = [("-" if i == j else str(torch.cuda.can_device_access_peer(i, j))) for j in range(n)]
    print(f"  gpu{i}: " + "  ".join(row))
print("\nIf every off-diagonal entry is False, NCCL stages through host memory.")
print("That is expected on consumer Ada cards and is the premise of bench_comms.py.")
PY
  echo; echo "--- CPU / memory / disk ---"
  lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\))"
  free -g | head -2
  df -h . | tail -1
  echo; echo "--- software ---"
  python3 -c "import torch,sys;print('python ',sys.version.split()[0]);print('torch  ',torch.__version__);print('cuda   ',torch.version.cuda);print('nccl   ','.'.join(map(str,torch.cuda.nccl.version())))"
  python3 -c "import triton;print('triton ',triton.__version__)" 2>/dev/null || echo "triton  NOT INSTALLED"
} 2>&1 | tee "$out"

echo; echo "wrote $out"
