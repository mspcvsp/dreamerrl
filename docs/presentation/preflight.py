"""Check Python dependencies and fingerprint the font faces used by publishing."""
import hashlib
import json
from pathlib import Path
import subprocess

try:
    import matplotlib
    from matplotlib.font_manager import FontProperties, findfont
    import PIL
except ImportError as exc:
    raise SystemExit(f"Publication dependency unavailable: {exc}. Provision the locked uv environment "
                     "with Matplotlib and Pillow as described in docs/presentation/README.md.") from exc

palette = json.loads((Path(__file__).parent.parent / "publication_figures/palette.json").read_text())


def match(family, style):
    return subprocess.check_output(
        ["fc-match", "--format=%{family}\n%{style}\n%{postscriptname}\n%{file}",
         f"{family}:style={style}"], text=True).splitlines()


def fingerprint(path):
    path = Path(path)
    return dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


for family in (palette["preferred_font"], palette["fallback_font"]):
    if family in match(family, "Regular")[0].split(","):
        try:
            faces = {}
            for style in ("Regular", "Bold"):
                name, resolved_style, postscript, file = match(family, style)
                mpl = findfont(FontProperties(family=family, weight=style.lower()), fallback_to_default=False)
                faces[style.lower()] = {
                    "fontconfig": dict(family=name, style=resolved_style, postscript=postscript, **fingerprint(file)),
                    "matplotlib": fingerprint(mpl),
                }
        except (ValueError, OSError) as exc:
            raise SystemExit(f"Font {family} could not be resolved by both renderers: {exc}. "
                             "Verify the font installation and refresh Fontconfig and Matplotlib font caches.") from exc
        print(json.dumps(dict(family=family, faces=faces, matplotlib=matplotlib.__version__, pillow=PIL.__version__), indent=2))
        break
else:
    raise SystemExit("Install Inter (preferred) or DejaVu Sans and refresh the Fontconfig cache.")
