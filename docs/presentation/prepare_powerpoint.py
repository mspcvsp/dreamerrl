"""Keep Pandoc picture-caption layouts and the teaching line on one PPTX slide."""
from pathlib import Path
import re
import sys

source, target = map(Path, sys.argv[1:3])
markdown = source.read_text()
pattern = r"^!\[[^\]]*\](\([^)]+\)\{[^}]+\})\n\n([^\n]+)$"
result, count = re.subn(pattern, lambda match: f"![{match[2]}]{match[1]}", markdown, flags=re.M)
if count != markdown.count("!["):
    raise ValueError("Each image slide must have one editable teaching line")
target.write_text(result)
