"""Record tool versions, source and built publication hashes."""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def version(command):
    if not shutil.which(command[0]):
        return "unavailable"
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return completed.stdout.strip().splitlines()[0]


root, output = map(Path, sys.argv[1:3])
font = sys.argv[3]
data = {}
for audience in ("linkedin","presentation"):
    files = [p for p in (output/audience/"figures").glob("*")]
    files += list((output/audience).glob("*.md"))
    files += [p for p in (output/audience).glob("scout_presentation.*") if p.suffix in (".pdf",".pptx")]
    data.update({str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
manifest = dict(
    source_revision=version(["git","-C",str(root),"rev-parse","HEAD"]),
    platform=platform.machine(), typography=font, figure_dpi=320,
    font_resolution=json.loads((output/"font_resolution.json").read_text()),
    palette_sha256=hashlib.sha256((root/"docs/publication_figures/palette.json").read_bytes()).hexdigest(),
    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ("linkedin","presentation","publication_figures")
                   for p in sorted((root/"docs"/folder).iterdir()) if p.is_file()},
    tools={k[0]: version(k) for k in (["pandoc","--version"],["xelatex","--version"],["pdftoppm","-v"],
                                    ["libreoffice","--version"],["rsvg-convert","--version"],["fc-match",font])},
    beamer=version(["kpsewhich","beamer.cls"]),
    outputs_sha256={**data, "figure_provenance.json": hashlib.sha256((output/"figure_provenance.json").read_bytes()).hexdigest()},
)
(output/"build_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
