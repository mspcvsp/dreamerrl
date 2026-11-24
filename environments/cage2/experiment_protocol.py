"""Standard-library access to the tracked original CAGE2 execution contract."""

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MANIFEST = PROJECT / "reproducibility/original_baseline.json"


def original_protocol():
    return json.loads(MANIFEST.read_text())


def original_schedule():
    execution = original_protocol()["execution"]
    schedule = execution["fixed_original_schedule"]
    return schedule["final_update"], [row["update"] for row in schedule["evaluation_checkpoints"]]


def verify_original_inputs(source, config):
    """Reject wrong starting bytes/config before restoring a trusted checkpoint."""
    protocol = original_protocol()
    digest = hashlib.sha256()
    with Path(source).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != protocol["source_checkpoint"]["sha256"]:
        raise ValueError("Starting checkpoint does not match the original update-1 fingerprint")
    if config is not None:
        expected = json.loads((MANIFEST.parent / protocol["configuration"]["file"]).read_text())
        actual = json.loads(json.dumps(config))  # Dataclass tuples serialize as arrays in historical JSON.
        expected.pop("log")
        actual.pop("log")
        if actual != expected:
            raise ValueError("Restored scientific configuration differs from the original baseline")
