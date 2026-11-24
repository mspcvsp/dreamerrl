"""Detached pilot supervision with an absolute four-hour child-process limit."""

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from host_experiment_support import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    command = [
        "uv",
        "run",
        "--locked",
        "python",
        "-u",
        str(ROOT / "environments/cage2/host_pilot.py"),
        "--manifest",
        str(args.manifest.resolve()),
        "--run-dir",
        str(run),
    ]
    start = time.monotonic()
    with (run / "console.log").open("a", buffering=1) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = json.loads((run / "launch.json").read_text())
        launch.update(pid=process.pid, training_command=command)
        (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        print("Detached training PID", process.pid, "log", run / "console.log", flush=True)
        try:
            result = process.wait(timeout=14400)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            result = 124
        launch.update(
            exit_code=result,
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - start,
        )
        (run / "exit_code").write_text(str(result) + "\n")
        (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    raise SystemExit(result)


if __name__ == "__main__":
    main()
