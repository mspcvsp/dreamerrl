"""File-based thermal shutdown handshake for an explicitly selected run."""

import json
import os
import tempfile
from pathlib import Path

REQUEST = "thermal_stop_request.json"
STATUS = "thermal_stop_status.json"


class ThermalStop(Exception):
    """A requested thermal shutdown has checkpointed successfully."""


def read_record(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def publish(path: Path, record: dict) -> None:
    """Publish a complete record once, even if two watchdogs request a stop."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream)
            stream.close()
            os.link(temporary, path)
        finally:
            temporary.unlink()


def checkpoint_if_requested(run: Path, trainer) -> None:
    request = read_record(run / REQUEST)
    if request is None:
        return
    checkpoint = run / "latest.pt"
    status = dict(
        request_id=request["request_id"],
        reason=request["reason"],
        update=trainer.completed_updates,
        env_steps=trainer.total_env_steps,
    )
    try:
        trainer.save_checkpoint(checkpoint)
    except Exception as error:
        publish(run / STATUS, dict(status, checkpoint_status="failed", error=repr(error)))
        raise
    publish(run / STATUS, dict(status, checkpoint_status="saved", checkpoint=str(checkpoint)))
    raise ThermalStop(f'{run}: {request["reason"]}; saved {checkpoint}')
