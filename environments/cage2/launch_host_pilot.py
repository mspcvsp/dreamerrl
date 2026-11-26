"""Start the one authorized pilot in a named detached tmux session."""

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from host_experiment_support import ROOT, verify_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = verify_gate(args.manifest)
    run = args.run_dir.resolve()
    session = "cage2-host-aware-v1"
    if run.exists():
        parser.error("Use a fresh run directory")
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        parser.error("Pilot session already exists; refusing duplicate training")
    command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "bash",
        str(ROOT / "scripts/run_host_pilot.sh"),
        "--manifest",
        str(args.manifest.resolve()),
        "--run-dir",
        str(run),
    ]
    now = datetime.now(timezone.utc)
    run.mkdir(parents=True, exist_ok=False)
    (run / "launch.json").write_text(
        json.dumps(
            dict(
                tmux_session=session,
                launch_command=command,
                start_time_utc=now.isoformat(),
                start_time_new_york=now.astimezone(ZoneInfo("America/New_York")).isoformat(),
                source_commit=manifest["source_commit"],
                source_tag=manifest["source_tag"],
                exit_code=None,
                hard_wall_seconds=14400,
                console_log=str(run / "console.log"),
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
                "--project",
                str(ROOT),
                "python",
                str(ROOT / "environments/cage2/launch_host_pilot.py"),
                "--manifest",
                str(args.manifest.resolve()),
                "--run-dir",
                str(run) + "_rerun",
            ]
        )
        + "\n"
    )
    subprocess.run(command, check=True)
    print("Launched", session, "console:", run / "console.log")


if __name__ == "__main__":
    main()
