"""Supervise a bounded queue of independent CAGE2 host-aware training runs."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from thermal_stop import REQUEST, STATUS, publish, read_record
from thermal_watchdog import POLL_SECONDS, READY, TRIP

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
CHECKPOINT_EXIT = 75
TERM_GRACE = 2
HEARTBEAT_TIMEOUT = 45


def spawn(command, log_path):
    with log_path.open("x") as log:
        return subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )


def signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def stop_group(process):
    if process.poll() is None:
        signal_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=TERM_GRACE)
        except subprocess.TimeoutExpired:
            pass
    signal_group(process, signal.SIGKILL)
    return process.wait()


def last_progress(run):
    progress = {}
    try:
        with (run / "metrics.jsonl").open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    progress = {key: row[key] for key in ("update", "env_steps")}
                except (ValueError, KeyError, TypeError):
                    continue
    except FileNotFoundError:
        pass
    return progress


@dataclass
class Job:
    run: Path
    command: list[str]
    manifest: Path
    arm: str
    seed: int
    process: subprocess.Popen | None = None
    started: float | None = None
    request_id: str | None = None
    forced: bool = False
    pid: int | None = None

    def request_checkpoint(self, reason):
        if self.request_id is not None or (self.run / "complete.json").exists():
            return
        request = dict(request_id=uuid4().hex, reason=reason)
        try:
            publish(self.run / REQUEST, request)
        except FileNotFoundError:
            return  # Launcher has yet to create the run directory.
        except FileExistsError:
            request = read_record(self.run / REQUEST)
        if request is not None:
            self.request_id = request["request_id"]

    def finish(self, ended, exit_code):
        try:
            return self.read_result(ended, exit_code)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return self.record(
                "failed", exit_code=exit_code, error=f"Unable to read worker result: {error}",
                runtime_seconds=ended - self.started if self.started is not None else None,
            )

    def read_result(self, ended, exit_code):
        completed = read_record(self.run / "complete.json")
        checkpoint = read_record(self.run / STATUS)
        if checkpoint and checkpoint["request_id"] != self.request_id:
            checkpoint = None
        if completed and exit_code == 0:
            state = "completed"
            checkpoint = dict(checkpoint_status="saved", checkpoint=completed["checkpoint"])
        elif self.forced:
            state = "shutdown_timeout"
        elif exit_code == CHECKPOINT_EXIT and checkpoint and checkpoint["checkpoint_status"] == "saved":
            state = "checkpointed"
        else:
            state = "failed"
        if checkpoint is None and (self.request_id or self.forced):
            checkpoint = dict(checkpoint_status="unconfirmed")
        progress = completed or checkpoint or {}
        if "update" not in progress:
            progress = last_progress(self.run)
        return self.record(
            state,
            exit_code=exit_code,
            runtime_seconds=ended - self.started if self.started is not None else None,
            checkpoint=checkpoint,
            updates=progress.get("update", 0),
            env_steps=progress.get("env_steps", 0),
        )

    def record(self, state, **details):
        return dict(
            state=state, manifest=str(self.manifest), arm=self.arm, seed=self.seed, run=str(self.run),
            pid=self.pid, command=self.command, console_log=str(self.run.parent.parent / "console.log"), **details,
        )


def supervise(root, jobs, watchdog_command, concurrency, readiness_timeout, checkpoint_timeout, poll_seconds=0.2):
    """Run queued jobs and return the durable summary and supervisor exit code."""
    queue = deque(jobs)
    active = []
    results = {}
    watchdog = None
    interrupted = None
    reason = first_launch = last_exit = None
    deadline = math.inf
    ready = False
    error = None
    watchdog_exit = None
    watchdog_log = root / "thermal.jsonl"
    heartbeat = None
    started = last_heartbeat = time.monotonic()

    def request_shutdown(signum, frame):
        nonlocal interrupted
        if interrupted is None:
            interrupted = signum

    publish(root / "execution_plan.json", dict(
        supervisor_pid=os.getpid(), concurrency=concurrency, watchdog_command=watchdog_command,
        readiness_timeout_seconds=readiness_timeout, checkpoint_timeout_seconds=checkpoint_timeout,
        jobs=[job.record("queued") for _, job in jobs],
    ))
    previous = {sig: signal.signal(sig, request_shutdown) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        watchdog = spawn(watchdog_command, root / "watchdog_console.log")
        while active or queue:
            now = time.monotonic()
            for index, job, process in active[:]:
                exit_code = process.poll()
                if exit_code is not None:
                    signal_group(process, signal.SIGKILL)  # Clean up any descendants left by the worker.
                    results[index] = job.finish(now, exit_code)
                    last_exit = now
                    active.remove((index, job, process))

            if reason is None:
                watchdog_exit = watchdog.poll()
                trip = read_record(root / TRIP)
                if watchdog_log.exists():
                    stamp = watchdog_log.stat().st_mtime_ns
                    if stamp != heartbeat:
                        heartbeat, last_heartbeat = stamp, now
                ready = read_record(root / READY) is not None
                if trip is not None:
                    reason = trip
                elif interrupted is not None:
                    reason = dict(type="supervisor_signal", signal=signal.Signals(interrupted).name)
                elif watchdog_exit is not None:
                    reason = dict(type="watchdog_failure", exit_code=watchdog_exit)
                elif not ready and now - started >= readiness_timeout:
                    reason = dict(type="watchdog_readiness_timeout")
                elif ready and now - last_heartbeat >= HEARTBEAT_TIMEOUT:
                    reason = dict(type="watchdog_heartbeat_timeout")
                if reason is not None:
                    deadline = now + checkpoint_timeout
                    print("SHUTDOWN", json.dumps(reason), flush=True)

            if reason is not None:
                while queue:
                    index, job = queue.popleft()
                    results[index] = job.record("skipped")
                for _, job, process in active:
                    job.request_checkpoint(reason)
                    if now >= deadline and not job.forced:
                        job.forced = True
                        signal_group(process, signal.SIGTERM)
                    if now >= deadline + TERM_GRACE:
                        signal_group(process, signal.SIGKILL)
            elif ready and queue and len(active) < concurrency:
                index, job = queue.popleft()
                job.started = time.monotonic()
                try:
                    process = spawn(job.command, job.run.parent.parent / "console.log")
                except OSError as exc:
                    results[index] = job.record("launch_failed", error=str(exc))
                else:
                    first_launch = job.started if first_launch is None else first_launch
                    job.pid = process.pid
                    active.append((index, job, process))
                    print("START", index, process.pid, job.run, flush=True)
                continue

            if active or queue:
                time.sleep(poll_seconds)
    except Exception as exc:
        error = repr(exc)
        reason = reason or dict(type="supervisor_error")
        for _, job, _ in active:
            try:
                job.request_checkpoint(reason)
            except (OSError, ValueError):
                pass
        deadline = time.monotonic() + checkpoint_timeout
        for _, _, process in active:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
    finally:
        try:
            for index, job, process in active:
                job.forced |= process.poll() is None
                exit_code = stop_group(process)
                last_exit = time.monotonic()
                results[index] = job.finish(last_exit, exit_code)
        finally:
            try:
                if watchdog is not None:
                    watchdog_exit = stop_group(watchdog)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

    ordered = [results.get(index) or job.record("skipped") for index, job in jobs]
    seconds = last_exit - first_launch if first_launch is not None and last_exit is not None else 0
    updates = sum(row.get("updates", 0) for row in ordered)
    env_steps = sum(row.get("env_steps", 0) for row in ordered)
    failed = error is not None or any(
        row["state"] in ("failed", "launch_failed", "shutdown_timeout") for row in ordered
    )
    if failed:
        exit_code = 1
    elif reason is None:
        exit_code = 0
    elif reason["type"] == "sustained_gpu_temperature":
        exit_code = CHECKPOINT_EXIT
    elif reason["type"] == "supervisor_signal":
        exit_code = 128 + int(signal.Signals[str(reason["signal"])])
    else:
        exit_code = 1
    summary = dict(
        exit_code=exit_code, shutdown_reason=reason, error=error, concurrency=concurrency,
        runtime_seconds=time.monotonic() - started, watchdog_ready=ready, watchdog_exit_code=watchdog_exit,
        watchdog_command=watchdog_command,
        aggregate=dict(
            training_wall_seconds=seconds, updates=updates, env_steps=env_steps,
            updates_per_second=updates / seconds if seconds else None,
            env_steps_per_second=env_steps / seconds if seconds else None,
        ),
        jobs=ordered,
    )
    publish(root / "execution_summary.json", summary)
    return summary, exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job", action="append", nargs=3, metavar=("MANIFEST", "ARM", "SEED"), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--warn-c", type=float, required=True)
    parser.add_argument("--stop-c", type=float, required=True)
    parser.add_argument("--sustain-seconds", type=int, default=30)
    parser.add_argument("--readiness-timeout", type=int, default=60)
    parser.add_argument("--checkpoint-timeout", type=int, default=300)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to(ROOT / "runs") or root.exists():
        parser.error("--output-root must be a fresh directory under this checkout's runs/")
    if min(args.concurrency, args.readiness_timeout, args.checkpoint_timeout) < 1:
        parser.error("concurrency and timeouts must be positive")
    if not 0 < args.warn_c < args.stop_c < math.inf or args.sustain_seconds < POLL_SECONDS:
        parser.error("require 0 < warn-c < stop-c and sustain-seconds >= 10")

    jobs = []
    for index, (manifest, arm, seed_string) in enumerate(args.job, 1):
        manifest_path = Path(manifest).resolve()
        if not manifest_path.is_file() or manifest_path.is_relative_to(ROOT / "runs"):
            parser.error("each manifest must exist outside the checkout's runs/")
        try:
            seed = int(seed_string)
        except ValueError:
            parser.error("job seeds must be integers")
        if arm not in ("flat", "cage2_host"):
            parser.error("job arm must be flat or cage2_host")
        job_root = root / f"job_{index:03d}"
        output = job_root / "results"
        command = [
            sys.executable, "-u", str(PROJECT / "host_aware_experiment.py"), "--manifest", str(manifest_path),
            "--output-root", str(output), "--arm", arm, "--seed", str(seed),
        ]
        jobs.append((index, Job(output / f"{arm}_seed{seed}", command, manifest_path, arm, seed)))

    root.mkdir(parents=True, exist_ok=False)
    for _, job in jobs:
        job.run.parent.parent.mkdir()
    watchdog_command = [
        sys.executable, "-u", str(PROJECT / "thermal_watchdog.py"),
        "--supervisor-control", str(root), "--log", str(root / "thermal.jsonl"),
        "--gpu", args.gpu, "--warn-c", str(args.warn_c), "--stop-c", str(args.stop_c),
        "--sustain-seconds", str(args.sustain_seconds),
    ]
    _, exit_code = supervise(
        root, jobs, watchdog_command, args.concurrency, args.readiness_timeout, args.checkpoint_timeout
    )
    print("SUMMARY", root / "execution_summary.json", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
