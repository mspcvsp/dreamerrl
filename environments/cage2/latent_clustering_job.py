"""Detached, 30-minute offline clustering supervision."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.worker:
        launch = json.loads((run / "launch.json").read_text())
        started = time.monotonic()
        with (run / "console.log").open("a", buffering=1) as log:
            command = [
                "uv",
                "run",
                "--locked",
                "python",
                "-u",
                str(ROOT / "environments/cage2/latent_clustering.py"),
                "--run-dir",
                str(run),
            ]
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            launch.update(pid=process.pid, analysis_command=command)
            (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
            launch.update(
                exit_code=code,
                elapsed_seconds=time.monotonic() - started,
                end_time_utc=datetime.now(timezone.utc).isoformat(),
            )
            (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
            (run / "exit_code").write_text(str(code) + "\n")
        raise SystemExit(code)
    for name in ("latent_clustering.py", "latent_clustering_job.py", "latent_clustering_config.json"):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "environments/cage2/" + name], check=True, capture_output=True
        )
    session = "cage2-latent-clustering"
    if run.exists() or subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        parser.error("Output directory or tmux session already exists")
    run.mkdir(parents=True)
    command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "bash",
        str(ROOT / "scripts/run_latent_clustering.sh"),
        "--worker",
        "--run-dir",
        str(run),
    ]
    (run / "launch.json").write_text(
        json.dumps(
            dict(
                tmux_session=session,
                command=command,
                start_time_utc=datetime.now(timezone.utc).isoformat(),
                time_limit_seconds=1800,
                exit_code=None,
            ),
            indent=2,
        )
        + "\n"
    )
    (run / "reproduce_command.txt").write_text(
        shlex.join(
            [
                "uv",
                "run",
                "--locked",
                "python",
                "environments/cage2/latent_clustering_job.py",
                "--run-dir",
                str(run) + "_rerun",
            ]
        )
        + "\n"
    )
    subprocess.run(command, check=True)
    print("Launched", session, "log", run / "console.log")


if __name__ == "__main__":
    main()
