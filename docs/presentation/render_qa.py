"""Render every page and article image for manual visual inspection."""
import argparse
import json
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps

NS_XHTML = "{http://www.w3.org/1999/xhtml}"
HEADINGS = Path(__file__).with_name("slides.md")


def montage(files, output, columns, thumb, padding=18):
    for group, start in enumerate(range(0,len(files),columns*3),1):
        paths=files[start:start+columns*3]
        width,height=thumb
        rows=(len(paths)+columns-1)//columns
        sheet=Image.new("RGB",(columns*(width+padding)+padding, rows*(height+34)+padding),"#DCE4E7")
        draw=ImageDraw.Draw(sheet)
        for i,path in enumerate(paths):
            with Image.open(path) as img:
                img=ImageOps.contain(img.convert("RGB"),thumb,Image.Resampling.LANCZOS)
            x=padding+(i%columns)*(width+padding)
            y=padding+(i//columns)*(height+34)
            sheet.paste(img,(x+(width-img.width)//2,y))
            draw.text((x,y+height+4),path.stem,fill="#13344C")
        sheet.save(output.with_name(f"{output.name}-{group}.png"))


def render_pdf(pdf, folder, headings):
    folder.mkdir(parents=True,exist_ok=True)
    subprocess.run(["pdftoppm","-scale-to-x","1280","-scale-to-y","720","-png",str(pdf),str(folder/"slide")],check=True)
    pages=sorted(folder.glob("slide-*.png"))
    if len(pages)!=len(headings):
        raise ValueError(f"Unexpected page count in {pdf}: {len(pages)}")
    text=subprocess.check_output(["pdftotext",str(pdf),"-"],text=True).split("\f")
    for page,title in zip(text,headings):
        if " ".join(title.split()) not in " ".join(page.split()):
            raise ValueError(f"Missing rendered title {title!r} in {pdf}")
    bbox=subprocess.check_output(["pdftotext","-bbox",str(pdf),"-"])
    xml=ET.fromstring(bbox)
    count=0
    for page in xml.iter(f"{NS_XHTML}page"):
        for word in page.iter(f"{NS_XHTML}word"):
            count+=1
            a=word.attrib
            if not(0 <= float(a["xMin"]) < float(a["xMax"]) <= float(page.attrib["width"]) and
                   0 <= float(a["yMin"]) < float(a["yMax"]) <= float(page.attrib["height"])):
                raise ValueError(f"Off-page text {word.text!r} in {pdf}")
    montage(pages,folder.parent/f"{folder.name}-contact",2,(640,360))
    return dict(pages=len(pages),words_within_page=count)


def editable(pptx, n):
    ns="{http://schemas.openxmlformats.org/drawingml/2006/main}"
    with ZipFile(pptx) as deck:
        for i in range(1,n+1):
            slide=ET.fromstring(deck.read(f"ppt/slides/slide{i}.xml"))
            if not any(t.text for t in slide.iter(ns+"t")):
                raise ValueError(f"No editable text on slide {i}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    root=args.output.resolve()
    qa=root/"qa"
    titles=["Learning Through Imagination"]+re.findall(r"^## (.+)$",HEADINGS.read_text(),re.M)
    checks={"beamer":render_pdf(root/"presentation/scout_presentation.pdf",qa/"beamer-pages",titles)}
    powerpoint=qa/"powerpoint/scout_presentation.pdf"
    checks["powerpoint"]=render_pdf(powerpoint,qa/"powerpoint-pages",titles) if powerpoint.exists() else "LibreOffice unavailable"
    editable(root/"presentation/scout_presentation.pptx",len(titles))
    checks["editable_slide_headings"]=len(titles)
    graphics=sorted((root/"linkedin/figures").glob("*.png"))
    for width in (400,800):
        resized=qa/f"linkedin-{width}"
        resized.mkdir(exist_ok=True)
        for path in graphics:
            with Image.open(path) as image:
                height=round(image.height*width/image.width)
                image.resize((width,height),Image.Resampling.LANCZOS).save(resized/path.name)
        montage(sorted(resized.glob("*.png")),qa/f"linkedin-{width}-contact",2 if width==400 else 1,(width,width+25))
    checks["linkedin_figures"]=len(graphics)
    checks["linkedin_review_widths_px"]=[400,800]
    (qa/"automated_checks.json").write_text(json.dumps(checks,indent=2)+"\n")


if __name__=="__main__":
    main()
