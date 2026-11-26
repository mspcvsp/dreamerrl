#!/usr/bin/env bash
# Build the LinkedIn-only output and preview. The Scout workflow stays separate.
set -euo pipefail
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$source_dir/../.." && pwd)"
repo="$(dirname "$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)")"
out="$(realpath -m "${1:?Supply a fresh publication output directory under runs/}")"
case "$out" in "$repo"/runs/*|"$root"/runs/*) ;; *) echo "Use a fresh directory under runs/" >&2; exit 2 ;; esac
[[ ! -e "$out" ]] || { echo "Choose a fresh output directory" >&2; exit 2; }
project="${PUBLICATION_UV_PROJECT:-$repo}"
export UV_PROJECT_ENVIRONMENT="${PUBLICATION_ENVIRONMENT:-$project/.venv}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR="$out/qa/matplotlib"
mkdir -p "$out/qa" "$out/linkedin"
uv run --project "$project" --locked --no-sync python "$root/docs/publication_figures/generate_figures.py" "$out" --audience linkedin
python3 "$source_dir/prepare_article.py" "$source_dir/article.md" "$out/linkedin/article.md"
python3 "$source_dir/preview.py" "$out/linkedin/article.md" "$out/linkedin/preview.html"
font="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["typography"])' "$out/figure_provenance.json")"
uv run --project "$project" --locked --no-sync python "$root/docs/presentation/preflight.py" > "$out/font_resolution.json"
python3 "$root/docs/presentation/manifest.py" "$root" "$out" "$font"
