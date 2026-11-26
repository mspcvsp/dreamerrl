#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$script_dir/uv_shell.sh"
cd "$script_dir/.."
uv python install 3.10.19
# Initialize the same pinned source and reuse the original native-wheel repair.
git submodule update --init -- environments/cage2/vendor/cage-challenge-2
revision="$(git -C environments/cage2/vendor/cage-challenge-2 rev-parse HEAD)"
[[ "$revision" == 26ce1c1253fa9e2e73f25e6a7f2da32860c11257 ]]
uv run --no-project --managed-python --python 3.10.19 python environments/cage2/prepare_native_wheel.py
uv sync --locked
if [[ "${1:-}" == --legacy ]]; then
    UV_PROJECT_ENVIRONMENT="$PWD/.venv-legacy" uv sync --locked --no-default-groups --group legacy
elif [[ $# -gt 0 ]]; then
    printf 'Usage: bash scripts/setup_uv.sh [--legacy]\n' >&2
    exit 2
fi
