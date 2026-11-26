"""Insert local image links at the draft's explicit placements."""
from pathlib import Path
import re
import sys

source, output = map(Path, sys.argv[1:3])
text = source.read_text()
cover = re.compile(r"\*\*Cover\*\* · `(\w+\.png)`\n\n\*\*Alt text\*\* · ([^\n]+)")
text, count = cover.subn(r"![\2](figures/\1)", text)
figure = re.compile(r"(\*\*Figure \d+ · `(\w+\.png)`\.\*\*[^\n]+)\n\n\*\*Alt text:\*\* ([^\n]+)")
text, figures = figure.subn(r"![\3](figures/\2)\n\n\1", text)
if (count, figures) != (1, 7):
    raise ValueError("Expected one cover and seven article placements")
output.write_text(text)
