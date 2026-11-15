"""Append-only, flushed metrics independent of any terminal dashboard."""

import json
from datetime import datetime, timezone
from pathlib import Path


class JSONLLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            self._repair_tail()
        self.fp = self.path.open("a", encoding="utf-8")

    def _repair_tail(self):
        """A killed process may leave an incomplete last record; retain its valid prefix."""
        with self.path.open("r+b") as file:
            file.seek(-1, 2)
            if file.read(1) == b"\n":
                return
            end = file.tell()
            start, position = 0, end
            while position:
                size = min(position, 4096)
                position -= size
                file.seek(position)
                block = file.read(size)
                newline = block.rfind(b"\n")
                if newline >= 0:
                    start = position + newline + 1
                    break
            file.seek(start)
            tail = file.read()
            try:
                valid = isinstance(json.loads(tail), dict)
            except (ValueError, UnicodeDecodeError):
                valid = False
            if valid:
                file.seek(0, 2)
                file.write(b"\n")
            else:
                file.truncate(start)

    def log(self, record: dict):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        # Serialize first: invalid metrics must not leave a partial JSON line.
        line = json.dumps(record, allow_nan=False)
        self.fp.write(line + "\n")
        self.fp.flush()

    def close(self):
        self.fp.close()
