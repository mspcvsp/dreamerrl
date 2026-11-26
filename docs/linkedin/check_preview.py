"""Capture LinkedIn preview proofs in Chromium at all configured widths."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "presentation"))
from render_qa import montage  # noqa: E402

STYLE = json.loads((HERE.parent / "publication_figures/linkedin_style.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--chromium",default="chromium")
    args=parser.parse_args()
    root=args.output.resolve()
    qa=root/"qa"
    findings={}
    with TemporaryDirectory(prefix="linkedin-browser-",dir=qa) as profile:
        for width in STYLE["preview_widths"]:
            height=8*width+1800
            url=(root/"linkedin/preview.html").as_uri()+f"?width={width}&figures=1"
            cmd=[args.chromium,"--headless","--disable-gpu","--disable-background-networking",
                 "--force-device-scale-factor=1",f"--user-data-dir={profile}",
                 f"--window-size={max(520, width+64)},{height}"]
            process=subprocess.run([*cmd,"--dump-dom",url],capture_output=True,text=True,check=True,timeout=45)
            match=re.search(r'<script[^>]+id="preview-measure">([^<]+)</script>',process.stdout)
            if not match:
                raise ValueError("Preview image measurements were unavailable in Chromium")
            figures=json.loads(match[1])
            expected={f"figures/{x.name}" for x in (root/"linkedin/figures").glob("*.png")}
            assert {f["name"] for f in figures}==expected and len(figures)==len(expected)
            assert all(f["loaded"] and f["width"]==width and f["y"]+f["height"]<height for f in figures)
            screenshot=qa/f"browser-{width}.png"
            subprocess.run([*cmd,f"--screenshot={screenshot}",url],capture_output=True,check=True,timeout=45)
            frames=qa/f"browser-figures-{width}"
            frames.mkdir(exist_ok=True)
            files=[]
            with Image.open(screenshot) as image:
                for f in figures:
                    name=frames/Path(f["name"]).name
                    x,y,w,h=[round(f[k]) for k in ("x","y","width","height")]
                    image.crop((x,y,x+w,y+h)).save(name)
                    files.append(name)
            montage(files,qa/f"browser-contact-{width}",2,(width,round(width*.75)))
            findings[str(width)]=figures
    (qa/"preview_checks.json").write_text(json.dumps(findings,indent=2)+"\n")


if __name__=="__main__":
    main()
