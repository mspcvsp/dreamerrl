"""Artifact writing and reporting utilities for frozen studies."""
import json
from pathlib import Path


def write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)
