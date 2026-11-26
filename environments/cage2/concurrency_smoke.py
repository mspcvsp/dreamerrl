"""Detached supervisor smoke test using real hierarchical Dreamer on POPGym."""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiment_supervisor import Job, supervise
from thermal_stop import ThermalStop, checkpoint_if_requested, publish
from thermal_watchdog import NvidiaTelemetry, monitor

PROJECT = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve()
DURATION = 185
SEEDS = (101, 202, 303)


def command(*args):
    return [sys.executable, "-u", str(SOURCE), *map(str, args)]


def worker(run, seed):
    import torch

    from dreamerrl.tests.functional.test_hierarchical_actor import small_config
    from dreamerrl.training.trainer import DreamerTrainer

    deadline = time.monotonic() + DURATION
    torch.set_num_threads(2)
    run.mkdir(parents=True)
    cfg = small_config(run / "tensorboard")
    cfg.env.seed = cfg.train.seed = seed
    cfg.train.cuda = True
    publish(run / "resolved_config.json", cfg.to_dict())
    trainer = DreamerTrainer(cfg)
    checkpoint = run / "latest.pt"
    def digest(state):
        return hashlib.sha256(bytes(state.tolist())).hexdigest()

    initial_rng = digest(torch.get_rng_state())
    try:
        with (run / "metrics.jsonl").open("x", buffering=1) as stream:
            def update(record):
                credit = record["credit_assignment"]
                stream.write(json.dumps(dict(
                    seed=seed, update=record["update"], env_steps=record["env_steps"],
                    manager_actor_loss=credit["manager_actor_loss"], worker_actor_loss=credit["worker_actor_loss"],
                    manager_decisions=credit["manager_decisions"], worker_decisions=credit["worker_decisions"],
                ), allow_nan=False) + "\n")
                checkpoint_if_requested(run, trainer)
                return time.monotonic() < deadline

            trainer.train(100_000, checkpoint_path=checkpoint, checkpoint_every=100_000, on_update=update)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["version"] == 5 and payload["config"].train.seed == payload["config"].env.seed == seed
        assert payload["rng"]["cuda"] and payload["completed_updates"] == trainer.completed_updates
        publish(run / "complete.json", dict(
            checkpoint=str(checkpoint), update=trainer.completed_updates, env_steps=trainer.total_env_steps,
            seed=seed, pid=os.getpid(), initial_rng=initial_rng, final_rng=digest(payload["rng"]["torch"]),
            final_cuda_rng=digest(payload["rng"]["cuda"][0]), device=str(trainer.device),
        ))
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def watchdog(root):
    publish(root / "watchdog_identity.json", {"pid": os.getpid()})
    query = NvidiaTelemetry("0")

    def sample():
        values = query()
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        values["swap_used_kib"] = int(memory["SwapTotal"].split()[0]) - int(memory["SwapFree"].split()[0])
        for line in Path("/proc/vmstat").read_text().splitlines():
            key, value = line.split()
            if key in ("pswpin", "pswpout"):
                values[key] = int(value)
        return values

    args = SimpleNamespace(
        supervisor_control=root, run=[], gpu="0", warn_c=75, stop_c=80, sustain_seconds=30,
    )
    with (root / "thermal.jsonl").open("x", buffering=1) as stream:
        def log(event, **data):
            stream.write(json.dumps(dict(event=event, timestamp=time.time(), **data)) + "\n")

        return monitor(args, sample, log)


def group_exited(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def phase(root, limit):
    root.mkdir()
    jobs = []
    for index, seed in enumerate(SEEDS[:limit], 1):
        run = root / f"job_{index:03d}" / "results" / f"popgym_seed{seed}"
        run.parent.parent.mkdir()
        jobs.append((index, Job(run, command("worker", run, seed), SOURCE, "popgym_hierarchical", seed)))
    # Run exactly one full wave at each limit.
    summary, code = supervise(root, jobs, command("watchdog", root), limit, 45, 15)
    workers = []
    for row in summary["jobs"]:
        run = Path(row["run"])
        if row["state"] != "completed":
            continue
        completed = json.loads((run / "complete.json").read_text())
        metrics = rows(run / "metrics.jsonl")
        assert all(metric["seed"] == row["seed"] for metric in metrics)
        assert [metric["update"] for metric in metrics] == list(range(1, row["updates"] + 1))
        assert metrics[-1]["env_steps"] == row["env_steps"]
        assert all(metric[key] > 0 for metric in metrics for key in ("manager_decisions", "worker_decisions"))
        assert completed["seed"] == row["seed"] and completed["device"].startswith("cuda")
        assert Path(row["console_log"]).is_file() and Path(completed["checkpoint"]).is_file()
        assert group_exited(row["pid"])
        workers.append(dict(
            **completed, runtime_seconds=row["runtime_seconds"],
            updates_per_second=row["updates"] / row["runtime_seconds"],
            env_steps_per_second=row["env_steps"] / row["runtime_seconds"],
            manager_actor_loss_mean=statistics.mean(metric["manager_actor_loss"] for metric in metrics),
            worker_actor_loss_mean=statistics.mean(metric["worker_actor_loss"] for metric in metrics),
        ))
    identity = json.loads((root / "watchdog_identity.json").read_text())
    assert group_exited(identity["pid"]) and summary["watchdog_ready"]
    assert code == 0 and len(workers) == limit, summary
    assert len({worker["checkpoint"] for worker in workers}) == limit
    for key in ("initial_rng", "final_rng", "final_cuda_rng"):
        assert len({worker[key] for worker in workers}) == limit
    aggregate = summary["aggregate"]
    for key, tally in (("updates", "update"), ("env_steps", "env_steps")):
        assert sum(worker[tally] for worker in workers) == aggregate[key]
        assert math.isclose(aggregate[key] / aggregate["training_wall_seconds"], aggregate[f"{key}_per_second"])
    ready_ns = (root / "watchdog_ready.json").stat().st_mtime_ns
    assert all(ready_ns <= Path(row["console_log"]).stat().st_mtime_ns for row in summary["jobs"])
    samples = [row for row in rows(root / "thermal.jsonl") if row["event"] == "sample"]
    # Exclude the readiness sample taken before workers launch.
    samples = samples[1:]
    assert len(samples) >= 2
    resources: dict[str, Any] = {"samples": len(samples)}
    for key in ("temperature_c", "gpu_utilization_pct", "host_memory_available_kib", "swap_used_kib"):
        values = [row[key] for row in samples if row.get(key) is not None]
        assert values, key
        resources[key] = dict(min=min(values), mean=statistics.mean(values), max=max(values))
    for key in ("pswpin", "pswpout"):
        resources[key + "_delta"] = samples[-1][key] - samples[0][key]
    resources["thermal_throttling_samples"] = sum(
        bool(row["sw_thermal_slowdown"] or row["hw_thermal_slowdown"]) for row in samples
    )
    return dict(concurrency=limit, result="pass", workers=workers, aggregate=aggregate, resources=resources,
                watchdog_pid=identity["pid"])


def run_smoke(root):
    started = time.monotonic()
    report: dict[str, Any] = dict(result="fail", phases=[], phase_target_seconds=DURATION, seeds=SEEDS)
    try:
        for limit in range(1, 4):
            print(f"START PHASE concurrency={limit}", flush=True)
            result = phase(root / f"concurrency_{limit}", limit)
            report["phases"].append(result)
            print(f"PASS PHASE concurrency={limit} throughput={result['aggregate']}", flush=True)
        tests = [
            sys.executable, "-B", "-m", "pytest", "-q", "--assert=plain", "--rootdir=.", "--confcutdir=.",
            "--import-mode=importlib", "--noconftest", "-c", "/dev/null", "-o", f"pythonpath={PROJECT}",
            "-p", "no:cacheprovider", "--basetemp", str(root / "pytest"),
            str(PROJECT / "test_experiment_supervisor.py"), str(PROJECT / "test_thermal_watchdog.py"),
        ]
        proc = subprocess.run(tests, capture_output=True, text=True, timeout=30, check=False)
        (root / "mock_shutdown_tests.log").write_text(proc.stdout + proc.stderr)
        report["mock_shutdown_exit_code"] = proc.returncode
        assert proc.returncode == 0, proc.stdout + proc.stderr
        eligible = [entry for entry in report["phases"]
                    if not entry["resources"]["thermal_throttling_samples"]
                    and not entry["resources"]["pswpin_delta"] and not entry["resources"]["pswpout_delta"]]
        if eligible:
            best = max(entry["aggregate"]["updates_per_second"] for entry in eligible)
            report["recommended_concurrency"] = min(
                entry["concurrency"] for entry in eligible if entry["aggregate"]["updates_per_second"] >= 0.9 * best
            )
        report["readiness"] = (
            "Passed unattended supervisor and checkpoint smoke on lightweight GPU POPGym. "
            "Production-scale memory, thermal headroom and throughput require separate measurement."
        )
        report["result"] = "pass"
        return 0
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
        return 1
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        publish(root / "final_report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "watchdog"))
    parser.add_argument("root", type=Path)
    parser.add_argument("seed", nargs="?", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        try:
            worker(args.root, args.seed)
        except ThermalStop as error:
            print(error, flush=True)
            return 75
        return 0
    if args.mode == "watchdog":
        return watchdog(args.root)
    return run_smoke(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
