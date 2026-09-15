#!/usr/bin/env bash
# Provision a Vast.ai box. Assumes a PyTorch + CUDA 12.x image.
set -euo pipefail

echo "==> system packages"
apt-get update -qq && apt-get install -y -qq git tmux htop nvtop unzip >/dev/null

echo "==> python packages"
pip install -q --upgrade pip
pip install -q "triton>=3.0" datasets pillow numpy pytest webdataset huggingface_hub
# DALI is optional; the loader detects it and falls back cleanly if absent.
pip install -q --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120 || \
  echo "    DALI unavailable, falling back to tuned torch loader"

echo "==> repo"
pip install -q -e .

echo "==> hardware probe"
bash scripts/probe_hardware.sh reports/hardware.txt

echo "==> kernel correctness (nothing else runs until this is green)"
pytest tests/ -q

cat <<'MSG'

Ready. Next:
  1. bash scripts/run_all_benchmarks.sh          # kernels, comms, dataloader
  2. Pull real data:
       huggingface-cli download OpenDriveLab/DriveLM --repo-type dataset --local-dir data/drivelm
  3. A/B the training run:
       torchrun --nproc_per_node=2 -m nuvla_forge.train --preset baseline  --steps 200
       torchrun --nproc_per_node=2 -m nuvla_forge.train --preset optimised --steps 200
  4. python -m nuvla_forge.bench.report        # fills the README table

MSG
