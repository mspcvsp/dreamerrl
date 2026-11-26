#!/usr/bin/env bash
# Usage: bash docs/presentation/build.sh /absolute/path/to/new-output
set -euo pipefail
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$source_dir/../.." && pwd)"
repo="$(dirname "$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)")"
out="$(realpath -m "${1:?Supply an output directory under runs/}")"
case "$out" in "$repo"/runs/*|"$root"/runs/*) ;; *) echo "Use a publication directory under runs/" >&2; exit 2 ;; esac
[[ ! -e "$out/build_manifest.json" ]] || { echo "Choose a fresh publication output" >&2; exit 2; }
mkdir -p "$out"/{linkedin,presentation,qa}
for cmd in pandoc xelatex kpsewhich pdftoppm pdfinfo pdffonts fc-match rsvg-convert python3 uv; do command -v "$cmd" >/dev/null; done
kpsewhich beamer.cls >/dev/null
palette="$root/docs/publication_figures/palette.json"
font="$(python3 - "$palette" <<'PY'
from pathlib import Path
import json, subprocess, sys
p=json.loads(Path(sys.argv[1]).read_text())
for name in [p["preferred_font"],p["fallback_font"]]:
    if name in subprocess.check_output(["fc-match","--format=%{family}",name],text=True).split(","):
        print(name)
        break
else:
    raise SystemExit("Neither publication font is installed")
PY
)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR="$out/qa/matplotlib"
uv_project="${PUBLICATION_UV_PROJECT:-$repo}"
export UV_PROJECT_ENVIRONMENT="${PUBLICATION_ENVIRONMENT:-$uv_project/.venv}"
uv run --project "$uv_project" --locked --no-sync python "$root/docs/publication_figures/generate_figures.py" "$out"
python3 "$root/docs/linkedin/prepare_article.py" "$root/docs/linkedin/article.md" "$out/linkedin/article.md"
cp "$source_dir/"{slides.md,presenter_guide.md} "$out/presentation/"
work="$(mktemp -d /tmp/dreamer-publication.XXXXXX)"
trap 'rm -rf -- "$work"' EXIT
cd "$work"
ln -s "$out/presentation/figures" figures
cp "$source_dir/slides.md" .
python3 - "$palette" "$font" "$source_dir/theme.tex" <<'PY'
import json, sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
colors="\n".join(r"\definecolor{Pub"+k.title()+r"}{HTML}{"+p[k][1:]+"}" for k in ["navy","teal","mist","orange","ivory"])
Path("theme-build.tex").write_text(colors+"\n"+r"\newcommand{\PublicationFont}{"+sys.argv[2]+"}\n"+Path(sys.argv[3]).read_text())
PY
sed 's/\.png)/.pdf)/g' slides.md > slides-vector.md
pandoc slides-vector.md -f markdown-implicit_figures -s -t beamer --slide-level=2 -V aspectratio=169 -V fontsize=12pt -H theme-build.tex -o scout_presentation.tex
python3 "$source_dir/prepare_beamer.py" scout_presentation.tex
for pass in 1 2; do xelatex -halt-on-error -interaction=nonstopmode scout_presentation.tex > "$out/qa/xelatex-$pass.log"; done
python3 "$source_dir/prepare_powerpoint.py" slides.md slides-powerpoint.md
pandoc slides-powerpoint.md -f markdown-implicit_figures -t pptx --slide-level=2 --reference-doc="$source_dir/reference.pptx" -o scout-pandoc.pptx
python3 "$source_dir/style_pptx.py" scout-pandoc.pptx "$out/presentation/scout_presentation.pptx" "$font"
cp scout_presentation.pdf "$out/presentation/"
if command -v libreoffice >/dev/null; then
    mkdir -p "$out/qa/powerpoint"
    libreoffice "-env:UserInstallation=file://$work/lo-profile" --headless --convert-to pdf --outdir "$out/qa/powerpoint" "$out/presentation/scout_presentation.pptx" > "$out/qa/libreoffice.log"
fi
python3 "$source_dir/manifest.py" "$root" "$out" "$font"
