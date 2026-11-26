"""Detached, bounded analysis supervision. Never invokes a Dreamer training entry point."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((ROOT / "environments/cage2/latent_analysis_config.json").read_text())
    if args.worker:
        command = [
            "uv",
            "run",
            "--locked",
            "python",
            "-u",
            str(ROOT / "environments/cage2/latent_analysis.py"),
            "--run-dir",
            str(run),
        ]
        started = time.monotonic()
        with (run / "console.log").open("a", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            launch = json.loads((run / "launch.json").read_text())
            launch.update(pid=process.pid, analysis_command=command)
            (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
            print("Analysis PID", process.pid, "log", run / "console.log", flush=True)
            try:
                result = process.wait(timeout=config["max_wall_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result = 124
            launch.update(
                exit_code=result,
                elapsed_seconds=time.monotonic() - started,
                end_time_utc=datetime.now(timezone.utc).isoformat(),
            )
            (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
            (run / "exit_code").write_text(str(result) + "\n")
            if result != 0:
                (run / "analysis_failed.txt").write_text(
                    "Analysis stopped with exit code "
                    + str(result)
                    + ". Inspect console.log; no automatic retry or training was launched.\n"
                )
        raise SystemExit(result)
    for path in (
        "environments/cage2/latent_analysis.py",
        "environments/cage2/latent_probe.py",
        "environments/cage2/latent_analysis_config.json",
        "environments/cage2/latent_analysis_job.py",
        "scripts/run_latent_analysis.sh",
    ):
        subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, check=True, capture_output=True)
    if run.exists():
        parser.error("Use a new analysis directory")
    session = "cage2-latent-analysis"
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        parser.error("Analysis session already exists")
    command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "bash",
        str(ROOT / "scripts/run_latent_analysis.sh"),
        "--worker",
        "--run-dir",
        str(run),
    ]
    run.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc)
    (run / "launch.json").write_text(
        json.dumps(
            dict(
                tmux_session=session,
                launch_command=command,
                start_time_utc=now.isoformat(),
                start_time_new_york=now.astimezone(ZoneInfo("America/New_York")).isoformat(),
                exit_code=None,
                time_limit_seconds=config["max_wall_seconds"],
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
                str(ROOT / "environments/cage2/latent_analysis_job.py"),
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
