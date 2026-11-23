#!/usr/bin/env bash
# Run with bash --noprofile --norc so interactive Conda hooks cannot affect verification.
set -euo pipefail
if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do conda deactivate; done
fi
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL VIRTUAL_ENV
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD UV_PROJECT_ENVIRONMENT UV_PYTHON
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin"
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
uv sync --locked
uv run --locked python smoke.py "$@"
