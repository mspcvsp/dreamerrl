#!/usr/bin/env bash
# Initialize pinned sources and the deterministic native-wheel repair before locked sync.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
project="$(cd -- "$(dirname -- "$0")" && pwd)"
repository="$(git -C "$project" rev-parse --show-toplevel)"
git -C "$repository" submodule update --init -- environments/cage2/vendor/cage-challenge-2
revision="$(git -C "$project/vendor/cage-challenge-2" rev-parse HEAD)"
[[ "$revision" == 26ce1c1253fa9e2e73f25e6a7f2da32860c11257 ]]
cd "$project"
uv python install 3.10.19
uv run --no-project --managed-python --python 3.10.19 python prepare_native_wheel.py
uv sync --locked
