"""Apply the shared palette and editable 16:9 layout to Pandoc PowerPoint files."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re
import xml.etree.ElementTree as ET

HERE = Path(__file__).parent
P = json.loads((HERE.parent / "publication_figures/palette.json").read_text())
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
for prefix, url in NS.items():
    ET.register_namespace(prefix, url)


def tag(name):
    prefix, local = name.split(":")
    return f"{{{NS[prefix]}}}{local}"


def add(parent, name, **attributes):
    return ET.SubElement(parent, tag(name), attributes)


def get(parent, name):
    found = parent.find(name, NS)
    return found if found is not None else add(parent, name)


def rgb(parent, color):
    return add(parent, "a:srgbClr", val=color.lstrip("#"))


def theme(root, font):
    for scheme in root.findall(".//a:clrScheme", NS):
        for item, color in zip(scheme, (P["navy"], P["ivory"], P["navy"], P["ivory"],
                                       P["teal"], P["mist"], P["orange"], P["dark_teal"],
                                       P["dark_orange"], P["navy"], P["mist"], P["mist"])):
            item.clear()
            rgb(item, color)
    for latin in root.findall(".//a:fontScheme/*/a:latin", NS):
        latin.set("typeface", font)


def emu(inches):
    return str(round(inches*914400))


def place(shape, x, y, width, height):
    node = get(get(shape, "p:spPr"), "a:xfrm")
    get(node, "a:off").attrib.update(x=emu(x), y=emu(y))
    get(node, "a:ext").attrib.update(cx=emu(width), cy=emu(height))


def text_style(box, size, color, font, bold=False):
    body = get(box, "a:bodyPr")
    body.attrib.update(lIns="0", rIns="0", tIns="0", bIns="0", anchor="t")
    for old in list(body):
        if old.tag.split("}")[-1].endswith("Autofit"):
            body.remove(old)
    add(body, "a:noAutofit")

    def properties(node, linked=False):
        node.set("sz", str(size*100))
        node.set("b", "1" if bold else "0")
        fill = node.find("a:solidFill", NS)
        if fill is not None:
            node.remove(fill)
        fill = ET.Element(tag("a:solidFill"))
        node.insert(0, fill)
        rgb(fill, P["mist"] if linked and color == P["ivory"] else color)
        get(node, "a:latin").set("typeface", font)

    for para in box.findall("a:p", NS):
        pp = para.find("a:pPr", NS)
        if pp is None:
            pp = ET.Element(tag("a:pPr"))
            para.insert(0, pp)
        pp.set("algn", "l")
        properties(get(pp,"a:defRPr"))
        for run in para.findall("a:r",NS):
            rpr = run.find("a:rPr",NS)
            if rpr is None:
                rpr = ET.Element(tag("a:rPr"))
                run.insert(0,rpr)
            properties(rpr, rpr.find("a:hlinkClick",NS) is not None)
        for rpr in para.findall("a:endParaRPr",NS):
            properties(rpr)


def slide(root, number, font, alt):
    dark = number in (1,7,15)
    root.set("showMasterSp", "0")
    main = root.find("p:cSld", NS)
    bg = main.find("p:bg", NS)
    if bg is not None:
        main.remove(bg)
    bg = ET.Element(tag("p:bg"))
    main.insert(0,bg)
    bp = add(bg,"p:bgPr")
    rgb(add(bp,"a:solidFill"), P["navy"] if dark else P["ivory"])
    add(bp,"a:effectLst")
    tree = main.find("p:spTree",NS)
    pictures = tree.findall("p:pic", NS)
    text = [(shape, shape.find("p:txBody",NS)) for shape in tree.findall("p:sp",NS)]
    text = [(s, b) for s, b in text if b is not None and any(
        (t.text or "").strip() for t in b.findall(".//a:t",NS))]
    if not text:
        raise ValueError(f"Slide {number} has no editable heading")
    color = P["ivory"] if dark else P["navy"]
    title, body = text[0], text[1:]
    place(title[0], .72, 2.05 if number == 1 else .51, 11.85, 1.2 if number == 1 else .8)
    text_style(title[1], 42 if number == 1 else 34, color, font, True)
    for i, (shape, words) in enumerate(body):
        if number == 1:
            y, width, height, sz = 3.36+i*.65, 11.85, .78, 25
        elif pictures:
            y, width, height, sz = 6.16+i*.47, 11.85, .86, 23
        else:
            y, width, height, sz = 1.82+i*1.9, 11.6, 4.7, 28
        place(shape, .72, y, width, height)
        text_style(words, sz, P["mist"] if number == 1 else color, font)
    for picture in pictures:
        place(picture, 1.02, 1.49, 11.28, 11.28*3.7/9.6)
        description = picture.find("p:nvPicPr/p:cNvPr", NS)
        if description is not None and number in alt:
            description.set("descr", alt[number])


def rewrite(source, target, font, reference):
    markdown = (HERE / "slides.md").read_text()
    sections = re.split(r"^## ", markdown, flags=re.M)[1:]
    alt = {i: match[1] for i, section in enumerate(sections, 2)
           if (match := re.search(r"!\[([^\]]+)\]", section))}
    with ZipFile(source) as source_zip, ZipFile(target, "w", ZIP_DEFLATED) as output:
        for name in source_zip.namelist():
            data = source_zip.read(name)
            slide_number = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
            if re.fullmatch(r"ppt/theme/theme\d+\.xml", name):
                root = ET.fromstring(data)
                theme(root, font)
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            elif re.fullmatch(r"ppt/presentation\.xml", name):
                root = ET.fromstring(data)
                get(root,"p:sldSz").attrib.update(cx=emu(13+1/3),cy=emu(7.5),type="screen16x9")
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            elif slide_number and not reference:
                root = ET.fromstring(data)
                slide(root, int(slide_number[1]), font, alt)
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            output.writestr(name,data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("target",type=Path)
    parser.add_argument("font")
    parser.add_argument("--reference",action="store_true")
    args=parser.parse_args()
    rewrite(args.source,args.target,args.font,args.reference)


if __name__=="__main__":
    main()
