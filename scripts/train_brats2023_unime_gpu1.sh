#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

exec python -u -m scripts.train \
  --config configs/v21_brats2023_unime.yaml \
  --output outputs/BraTS2023_UniME \
  "$@"
