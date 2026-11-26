"""Exercise real subprocess supervision with mock workers and synthetic temperatures."""

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import experiment_supervisor as supervisor
import pytest

WORKER = """
import json, os, signal, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
from thermal_stop import ThermalStop, checkpoint_if_requested

root, run = map(Path, sys.argv[2:4])
mode = sys.argv[4]
assert (root / 'watchdog_ready.json').exists()
run.mkdir(parents=True)
if mode == 'hang':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
(run / 'pids.json').write_text(json.dumps([os.getpid(), child.pid]))
started = time.monotonic()
(run / 'started').touch()
(run / 'metrics.jsonl').write_text('{"update": 3, "env_steps": 90}\\n')

trainer = SimpleNamespace(
    completed_updates=3, total_env_steps=90, save_checkpoint=lambda p: p.write_text('fresh checkpoint'),
)
while time.monotonic() - started < 0.35 or mode in ('wait', 'hang'):
    if mode != 'hang':
        try:
            checkpoint_if_requested(run, trainer)
        except ThermalStop:
            sys.exit(75)
    time.sleep(0.01)
if mode == 'fail':
    sys.exit(7)
(run / 'timeline.json').write_text(json.dumps([started, time.monotonic()]))
trainer.save_checkpoint(run / 'latest.pt')
(run / 'complete.json').write_text(json.dumps(dict(update=3, env_steps=90, checkpoint=str(run / 'latest.pt'))))
"""

WATCHDOG = """
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
import thermal_watchdog as watchdog

root = Path(sys.argv[2])
mode = sys.argv[3]
watchdog.POLL_SECONDS = 0.01
args = SimpleNamespace(
    supervisor_control=root, run=[], warn_c=70, stop_c=80, sustain_seconds=0.03, gpu='0',
)

def sample():
    active = len(list(root.glob('job_*/results/*/started')))
    if mode == 'stalled':
        __import__('time').sleep(60)
    if mode == 'silent':
        return {'temperature_c': 82}
    if active and mode == 'crash':
        sys.exit(9)
    if active and mode == 'missing':
        return {'temperature_c': None}
    hot = (mode == 'hot' and active) or (mode == 'hot_pair' and active >= 2) or (mode == 'hot_three' and active >= 3)
    return {'temperature_c': 82 if hot else 45}

def log(event, **data):
    with (root / 'thermal.jsonl').open('a') as stream:
        print(event, file=stream)

(root / 'watchdog_pid').write_text(str(__import__('os').getpid()))
sys.exit(watchdog.monitor(args, sample, log))
"""


def prepare(root, modes):
    jobs = []
    for index, mode in enumerate(modes, 1):
        job_root = root / f"job_{index}"
        job_root.mkdir()
        run = job_root / "results" / f"flat_seed{index}"
        command = [sys.executable, "-c", WORKER, str(supervisor.PROJECT), str(root), str(run), mode]
        jobs.append((index, supervisor.Job(run, command, Path("mock.json"), "flat", index)))
    return jobs


def execute(root, jobs, limit, telemetry="ok", timeout=0.35, ready_timeout: float = 2, wall_seconds=None):
    command = [sys.executable, "-c", WATCHDOG, str(supervisor.PROJECT), str(root), telemetry]
    summary, code = supervisor.supervise(
        root, jobs, command, limit, ready_timeout, timeout, poll_seconds=0.01, wall_seconds=wall_seconds
    )
    assert json.loads((root / "execution_summary.json").read_text()) == summary
    pids = [int((root / "watchdog_pid").read_text())]
    for path in root.glob("job_*/results/*/pids.json"):
        pids.extend(json.loads(path.read_text()))
    for pid in pids:
        for _ in range(100):
            try:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"PID {pid} is still running")
    return summary, code


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_concurrency_throughput_and_isolation(tmp_path, limit):
    jobs = prepare(tmp_path, ["ok"] * 4)
    summary, code = execute(tmp_path, jobs, limit)
    assert code == 0
    assert summary["watchdog_ready"]
    assert summary["aggregate"]["updates"] == 12
    assert summary["aggregate"]["env_steps"] == 360
    assert summary["aggregate"]["updates_per_second"] > 0
    assert {row["state"] for row in summary["jobs"]} == {"completed"}
    events = []
    for _, job in jobs:
        start, end = json.loads((job.run / "timeline.json").read_text())
        assert (job.run / "latest.pt").exists()
        assert (job.run.parent.parent / "console.log").exists()
        events.extend(((start, 1), (end, -1)))
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    assert peak == limit


def test_worker_failure_does_not_prevent_other_jobs(tmp_path):
    jobs = prepare(tmp_path, ["fail", "ok", "ok"])
    summary, code = execute(tmp_path, jobs, 2)
    assert code == 1
    assert [row["state"] for row in summary["jobs"]] == ["failed", "completed", "completed"]
    assert summary["jobs"][0]["exit_code"] == 7


@pytest.mark.parametrize(
    "telemetry,reason,exit_code", [
        ("hot", "sustained_gpu_temperature", 75),
        ("crash", "watchdog_failure", 1),
        ("missing", "watchdog_failure", 1),
    ],
)
def test_shutdown_checkpoint_and_queue_cancellation(tmp_path, telemetry, reason, exit_code):
    jobs = prepare(tmp_path, ["wait", "wait"])
    summary, code = execute(tmp_path, jobs, 1, telemetry, timeout=2)
    assert code == exit_code
    assert summary["shutdown_reason"]["type"] == reason
    assert [row["state"] for row in summary["jobs"]] == ["checkpointed", "skipped"]
    checkpoint = summary["jobs"][0]["checkpoint"]
    assert checkpoint["checkpoint_status"] == "saved"
    assert checkpoint["reason"] == summary["shutdown_reason"]
    assert not jobs[1][1].run.exists()


def test_thermal_trip_checkpoints_both_active_workers(tmp_path):
    jobs = prepare(tmp_path, ["wait", "wait", "wait"])
    summary, code = execute(tmp_path, jobs, 2, "hot_pair", timeout=2)
    assert code == 75
    assert [job["state"] for job in summary["jobs"]] == ["checkpointed", "checkpointed", "skipped"]
    assert summary["aggregate"]["updates"] == 6


def test_readiness_timeout_never_launches_worker(tmp_path):
    jobs = prepare(tmp_path, ["ok"])
    summary, code = execute(tmp_path, jobs, 1, "stalled", ready_timeout=0.5)
    assert code == 1
    assert summary["shutdown_reason"]["type"] == "watchdog_readiness_timeout"
    assert summary["jobs"][0]["state"] == "skipped"


def test_checkpoint_timeout_cleans_worker_and_descendants(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "TERM_GRACE", 0.1)
    jobs = prepare(tmp_path, ["hang"])
    summary, code = execute(tmp_path, jobs, 1, "hot", timeout=0.12)
    assert code == 1
    row = summary["jobs"][0]
    assert row["state"] == "shutdown_timeout"
    assert row["exit_code"] == -signal.SIGKILL
    assert row["checkpoint"]["checkpoint_status"] == "unconfirmed"
    assert (jobs[0][1].run / "thermal_stop_request.json").exists()


def test_hot_start_never_launches_worker(tmp_path):
    jobs = prepare(tmp_path, ["ok"])
    summary, code = execute(tmp_path, jobs, 1, "silent")
    assert code == 75
    assert not summary["watchdog_ready"]
    assert summary["jobs"][0]["state"] == "skipped"


def test_supervisor_signal_requests_checkpoint(tmp_path):
    jobs = prepare(tmp_path, ["wait", "wait"])
    cancelled = threading.Event()

    def interrupt():
        while not cancelled.wait(0.01):
            if (jobs[0][1].run / "started").exists():
                os.kill(os.getpid(), signal.SIGTERM)
                return

    thread = threading.Thread(target=interrupt)
    thread.start()
    try:
        summary, code = execute(tmp_path, jobs, 1, timeout=2)
    finally:
        cancelled.set()
        thread.join()
    assert code == 143
    assert summary["shutdown_reason"]["type"] == "supervisor_signal"
    assert [row["state"] for row in summary["jobs"]] == ["checkpointed", "skipped"]


@pytest.mark.parametrize(("telemetry", "wall_seconds", "reason", "status"), [
    ("hot_three", None, "sustained_gpu_temperature", 75),
    ("ok", 1, "wall_limit", 1),
])
def test_three_workers_share_watchdog_and_checkpoint(tmp_path, telemetry, wall_seconds, reason, status):
    jobs = prepare(tmp_path, ["wait"] * 3)
    summary, code = execute(tmp_path, jobs, 3, telemetry, timeout=2, wall_seconds=wall_seconds)
    assert code == status
    assert summary["shutdown_reason"]["type"] == reason
    assert summary["watchdog_ready"]
    assert [job["state"] for job in summary["jobs"]] == ["checkpointed"] * 3
    assert len({job["console_log"] for job in summary["jobs"]}) == 3
    assert len({job["checkpoint"]["request_id"] for job in summary["jobs"]}) == 3
