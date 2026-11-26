"""Create a standalone local article preview with selectable display widths."""
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).parent
STYLE = json.loads((HERE.parent / "publication_figures/linkedin_style.json").read_text())
PALETTE = json.loads((HERE.parent / "publication_figures/palette.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("article", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    widths = STYLE["preview_widths"]
    preview = subprocess.check_output(["pandoc", "--from=markdown-implicit_figures", "--to=html5",
                                       str(args.article)], text=True)
    buttons = "".join(f'<a href="?width={w}">{w} px</a>' for w in widths)
    args.output.write_text(f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local LinkedIn article preview</title>
<style>
html {{background: {PALETTE["ivory"]}; color: {PALETTE["navy"]}; font: 17px/1.55 Inter, "DejaVu Sans", sans-serif;}}
body {{margin: 0; padding: 16px;}}
nav,article {{margin: 0 auto; max-width: min(100%,var(--width,600px));}}
nav {{display:flex;flex-wrap:wrap;gap:12px;font-size:.85rem;}}
a {{color: {PALETTE["dark_teal"]};}}
h1,h2 {{line-height: 1.15;}}
img {{display:block;width:100%;height:auto;}}
body.figures article > :not(.figure):not(.caption) {{display:none;}}
</style>
</head>
<body>
<nav>{buttons}<span>Local layout, approximate display widths</span></nav>
<article>{preview}</article>
<script>
const options = new URLSearchParams(location.search);
const widths = {json.dumps(widths)};
const width = Number(options.get("width"));
if (widths.includes(width)) document.documentElement.style.setProperty("--width", width + "px");
if (options.get("figures")==="1") document.body.classList.add("figures");
document.querySelectorAll("article > p").forEach(p => {{
  if (p.querySelector("img")) {{
    p.classList.add("figure");
    const following = p.nextElementSibling;
    if (following && following.textContent.startsWith("Figure ")) following.classList.add("caption");
  }}
}});
window.addEventListener("load", () => {{
  const measure = document.createElement("script");
  measure.type = "application/json";
  measure.id = "preview-measure";
  measure.textContent = JSON.stringify(Array.from(document.images, img => {{
    const rect = img.getBoundingClientRect();
    return {{name: img.getAttribute("src"), loaded: img.complete && img.naturalWidth > 0,
             x: rect.x, y: rect.y, width: rect.width, height: rect.height}};
  }}));
  document.body.appendChild(measure);
}});
</script>
</body></html>
""")


if __name__ == "__main__":
    main()
