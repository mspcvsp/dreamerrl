#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$project/../../scripts/uv_shell.sh"
cd "$project/../.."
uv run --no-project --managed-python --python 3.10.19 python scripts/baseline_uv.py verify "$@"
