"""Five-minute one/two-process CAGE2 throughput benchmark; never saves research checkpoints."""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(run, seed):
    import torch
    from host_experiment_support import initialize_paired, load_config

    from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
    from dreamerrl.training.trainer import DreamerTrainer

    torch.set_num_threads(2)
    cfg = load_config("cage2_host", seed, run)
    write(run / "resolved_config.json", asdict(cfg))
    trainer = DreamerTrainer(cfg)
    assert trainer.device.type == "cuda"
    assert isinstance(trainer.env, Cage2BlueEnv)
    initialize_paired(trainer)
    for _ in range(16):
        trainer.collect_env_steps()
    trainer.env.journal.clear()
    (run / "ready").write_text(str(time.monotonic()))
    while not (run.parent / "go.json").exists():
        time.sleep(0.05)
    schedule = json.loads((run.parent / "go.json").read_text())
    with (run / "updates.jsonl").open("w", buffering=1) as stream:

        def record(row):
            now = time.monotonic()
            stream.write(
                json.dumps(
                    dict(
                        time=now,
                        update=row["update"],
                        cadence=row["cadence"],
                        losses=[row["world_loss"], row["actor_loss"], row["critic_loss"]],
                    )
                )
                + "\n"
            )
            if row["update"] % 50 == 0:
                print("PROGRESS", run.name, row["update"], flush=True)
            return now < schedule["end"]

        trainer.train(total_updates=100000, on_update=record)
    write(
        run / "complete.json",
        dict(
            updates=trainer.completed_updates,
            env_steps=trainer.total_env_steps,
            optimizer_steps=trainer._optimizer_steps,
        ),
    )
    trainer.env.venv.close()
    trainer.tb.close()


def telemetry(pids):
    cpu = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    swap = {
        line.split()[0]: int(line.split()[1])
        for line in Path("/proc/vmstat").read_text().splitlines()
        if line.startswith(("pswpin ", "pswpout "))
    }
    ticks = 0
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().split(") ")[1].split()
            ticks += int(stat[11]) + int(stat[12])
        except FileNotFoundError:
            pass
    raw = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], text=True
        )
        .strip()
        .split(",")
    )
    return dict(
        time=time.monotonic(),
        cpu_total=sum(cpu),
        cpu_idle=cpu[3] + cpu[4],
        worker_ticks=ticks,
        available_gib=memory["MemAvailable"] / 1024**2,
        swap_used_gib=(memory["SwapTotal"] - memory["SwapFree"]) / 1024**2,
        **swap,
        gpu_percent=float(raw[0]) if raw[0].strip().isdigit() else None,
        gpu_temperature_c=float(raw[1]) if raw[1].strip().isdigit() else None,
    )


def summarize(phase, end):
    import numpy as np

    workers = sorted(phase.glob("worker*"))
    records = [[json.loads(line) for line in (path / "updates.jsonl").read_text().splitlines()] for path in workers]
    assert all(len(rows) > 5 for rows in records)
    start = max(rows[4]["time"] for rows in records)
    selected = [[row for row in rows if start < row["time"] <= end] for rows in records]
    count = sum(map(len, selected))
    samples = [json.loads(line) for line in (phase / "telemetry.jsonl").read_text().splitlines()]
    samples = [row for row in samples if start <= row["time"] <= end]
    assert len(samples) > 2
    cpu, worker_cpu = [], []
    for a, b in zip(samples, samples[1:]):
        cpu.append(100 * (1 - (b["cpu_idle"] - a["cpu_idle"]) / max(1, b["cpu_total"] - a["cpu_total"])))
        worker_cpu.append(
            100 * (b["worker_ticks"] - a["worker_ticks"]) / os.sysconf("SC_CLK_TCK") / (b["time"] - a["time"])
        )
    durations = {
        name: float(np.mean([row["cadence"][name] for rows in selected for row in rows]))
        for name in ["collection_seconds", "world_model_seconds", "actor_critic_seconds", "update_seconds"]
    }
    result: dict[str, Any] = dict(
        processes=len(workers),
        measured_seconds=end - start,
        measured_updates=count,
        updates_per_worker=list(map(len, selected)),
        aggregate_updates_per_second=count / (end - start),
        per_update_mean_seconds=durations,
        host_cpu_mean_percent=float(np.mean(cpu)),
        workers_cpu_mean_percent_one_core_100=float(np.mean(worker_cpu)),
        available_gib_min=min(row["available_gib"] for row in samples),
        swap_used_gib_max=max(row["swap_used_gib"] for row in samples),
        swap_in_mib=(samples[-1]["pswpin"] - samples[0]["pswpin"]) * os.sysconf("SC_PAGE_SIZE") / 2**20,
        swap_out_mib=(samples[-1]["pswpout"] - samples[0]["pswpout"]) * os.sysconf("SC_PAGE_SIZE") / 2**20,
    )
    for name in ["gpu_percent", "gpu_temperature_c"]:
        values = [row[name] for row in samples if row[name] is not None]
        result[name + "_mean"] = float(np.mean(values)) if values else None
        result[name + "_max"] = max(values) if values else None
    write(phase / "summary.json", result)
    return result


def benchmark(run):
    import hashlib

    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 300
    write(
        run / "manifest.json",
        dict(
            source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            uv_lock_sha256=hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
            start_time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            settings=(
                "host-aware pilot config unchanged; fresh weights/optimizers/replay; "
                "16 initial episodes; 2 CPU threads/worker"
            ),
            warmup="first 5 completed updates per worker excluded; shared steady-state measurement window",
            schedule="one process then two; nominal 135 seconds/phase; total hard cap 300 seconds",
            limitations=(
                "single short ordered trial; no evaluation/checkpoint I/O; "
                "no independent repeats; CUDA phase synchronization retained"
            ),
        ),
    )
    (run / "source.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], text=True))
    (run / "rerun_command.txt").write_text(
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "uv run --locked python environments/cage2/concurrency_benchmark.py --run-dir " + str(run) + "_rerun\n"
    )
    summaries = []
    for number in [1, 2]:
        phase = run / f"processes_{number}"
        phase.mkdir()
        processes = []
        logs = []
        try:
            for seed in range(number):
                output = phase / f"worker{seed}"
                output.mkdir()
                log = (output / "console.log").open("w")
                logs.append(log)
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--seed",
                            str(seed),
                            "--run-dir",
                            str(output),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=ROOT,
                    )
                )
            while not all((phase / f"worker{seed}" / "ready").exists() for seed in range(number)):
                if time.monotonic() > deadline - 30 or any(p.poll() is not None for p in processes):
                    raise RuntimeError("Worker startup failed or exceeded budget")
                time.sleep(0.1)
            end = min(time.monotonic() + 135, deadline - 3)
            write(phase / "go.json", dict(end=end))
            print("PHASE_STARTED", number, "seconds", end - time.monotonic(), flush=True)
            with (phase / "telemetry.jsonl").open("w", buffering=1) as stream:
                while time.monotonic() < end:
                    stream.write(json.dumps(telemetry([p.pid for p in processes])) + "\n")
                    if time.monotonic() < end and any(p.poll() is not None for p in processes):
                        raise RuntimeError("Worker stopped before phase deadline")
                    time.sleep(1)
            for p in processes:
                p.wait(timeout=max(1, deadline - time.monotonic()))
                assert p.returncode == 0
            summaries.append(summarize(phase, end))
            print("PHASE_COMPLETE", json.dumps(summaries[-1]), flush=True)
        finally:
            for p in processes:
                if p.poll() is None:
                    p.kill()
                    p.wait()
            for log in logs:
                log.close()
    one, two = summaries
    speedup = two["aggregate_updates_per_second"] / one["aggregate_updates_per_second"]
    sequential = 2 * 16899 / one["aggregate_updates_per_second"] / 3600
    concurrent = 2 * 16899 / two["aggregate_updates_per_second"] / 3600
    result = dict(
        phases=summaries,
        aggregate_speedup=speedup,
        sequential_two_runs_hours=sequential,
        concurrent_two_runs_hours=concurrent,
        elapsed_seconds=time.monotonic() - started,
        recommendation="Parallel worthwhile for throughput" if speedup > 1.1 else "Prefer sequential execution",
        extrapolation=(
            "Two 16899-update runs, compute/collection only; evaluation, startup/checkpoint overhead excluded"
        ),
    )
    write(run / "results.json", result)
    (run / "final_report.txt").write_text(json.dumps(result, indent=2) + "\n")
    print("BENCHMARK_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    worker(args.run_dir.resolve(), args.seed) if args.worker else benchmark(args.run_dir.resolve())
