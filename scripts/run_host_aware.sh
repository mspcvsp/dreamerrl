#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$script_dir/uv_shell.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
cd "$script_dir/.."
uv sync --locked
uv run --locked python -u environments/cage2/host_aware_experiment.py "$@"
