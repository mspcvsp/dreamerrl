#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$script_dir/uv_shell.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export UV_PROJECT_ENVIRONMENT="$script_dir/../.venv-legacy"
cd "$script_dir/.."
uv sync --locked --check --no-default-groups --group legacy
exec uv run --locked --no-sync --no-default-groups --group legacy python -B -u environments/minihack/director_room5x5.py "$@"
