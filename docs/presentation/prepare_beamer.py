"""Apply the dark slide theme to the same title, activity, and closing frames."""
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text().replace(r"\frame{\titlepage}", r"\begin{frame}\titlepage\end{frame}")
frames = re.compile(r"\\begin\{frame\}.*?\\end\{frame\}", re.S)
counter = 0


def dark(found):
    global counter
    counter += 1
    body = found.group()
    if counter in (9, 15):
        body = re.sub(r"(^\\begin\{frame\}\{[^}]*\})", r"\1\n\\color{PubIvory}", body, count=1)
    if counter == 15:
        body, links = re.subn(r"(\\href\{[^{}]+\})\{([^{}]+)\}", r"\1{\\textcolor{PubMist}{\2}}", body)
        if links != 3:
            raise ValueError("Expected three readable closing-slide links")
    return body


text = frames.sub(dark, text)
if counter != 15:
    raise ValueError(f"Expected 15 frames but Pandoc emitted {counter}")
path.write_text(text)
