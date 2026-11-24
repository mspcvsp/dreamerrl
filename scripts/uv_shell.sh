#!/usr/bin/env bash
# Source from tracked launchers: no external activation script is required.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL VIRTUAL_ENV
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD UV_PROJECT_ENVIRONMENT UV_PYTHON
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin"
unset UV_PYTHON_PREFERENCE
