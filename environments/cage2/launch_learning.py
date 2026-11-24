"""Launch a tracked CAGE2 entry point in tmux; write only generated logs under runs/."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def launch_command(run, session, source, schedule):
    training = [
        "bash",
        "--noprofile",
        "--norc",
        str(PROJECT / "run_learning_clean.sh"),
        "--run-dir",
        str(run),
        "--schedule",
        schedule,
    ]
    if source is not None:
        training.extend(["--source", str(source)])
    job = Path(str(run) + "_job")
    # Only shell syntax is literal; all paths/arguments are shell-quoted individually.
    script = (
        shlex.join(training) + " > " + shlex.quote(str(job / "console.log")) + " 2>&1; "
        "result=$?; printf '%s\\n' \"$result\" > " + shlex.quote(str(job / "exit_code")) + '; exit "$result"'
    )
    return ["tmux", "new-session", "-d", "-s", session, "bash", "-c", script]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--schedule", choices=["original", "measured"], default="original")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    source = args.source.resolve() if args.source else None
    command = launch_command(run, args.session, source, args.schedule)
    if args.dry_run:
        print(shlex.join(command))
        return
    if run.exists() or Path(str(run) + "_job").exists():
        parser.error("Use a new run and job directory; existing artifacts will not be overwritten")
    if subprocess.run(["tmux", "has-session", "-t", args.session], capture_output=True).returncode == 0:
        parser.error("The tmux session already exists")
    job = Path(str(run) + "_job")
    job.mkdir(parents=True, exist_ok=False)
    (job / "launch.json").write_text(json.dumps(dict(command=command, schedule=args.schedule), indent=2) + "\n")
    subprocess.run(command, check=True)
    print(f"Session: {args.session}\nLog: {job / 'console.log'}")


if __name__ == "__main__":
    main()
