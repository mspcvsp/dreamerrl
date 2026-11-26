"""Run the offline hierarchy audit and regression checks, and publish a final report."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from thermal_stop import publish

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]


def run_logged(command, log, timeout):
    print("START", log.name, flush=True)
    with log.open("x", buffering=1) as stream:
        proc = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    print("EXIT", log.name, proc.returncode, flush=True)
    return proc.returncode


def estimate_overhead():
    import torch
    from torch.utils.benchmark import Timer

    from dreamerrl.training.core.hierarchical_update import goal_diagnostics

    torch.set_num_threads(2)
    ids = torch.zeros((15, 512), dtype=torch.long)
    weights = torch.ones_like(ids, dtype=torch.float32)
    embeddings = torch.ones((8, 16))
    scope: dict[str, Any] = dict(
        goal_diagnostics=goal_diagnostics, ids=ids, weights=weights, embeddings=embeddings, json=json,
    )
    reducer = "goal_diagnostics(ids, weights, 4, embeddings)"
    cost = Timer(reducer, globals=scope, num_threads=2).blocked_autorange(min_run_time=0.2)
    payload = {key: value.item() for key, value in goal_diagnostics(ids, weights, 4, embeddings).items()}
    scope["payload"] = payload
    serial = Timer("json.dumps(payload)", globals=scope).blocked_autorange(min_run_time=0.2)
    return dict(
        imagined_shape=[15, 512], goals=8, cpu_reducer_microseconds=cost.median * 1e6,
        json_encode_microseconds=serial.median * 1e6, added_goal_json_bytes=len(json.dumps(payload).encode()),
        scope="CPU microbenchmark. Timing excludes GPU synchronization, TensorBoard I/O, and full-update overhead.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.is_relative_to(ROOT / "runs") or not root.is_dir():
        parser.error("--output must be an existing launch directory under runs/")
    if (root / "final_report.json").exists():
        parser.error("This launch directory already contains a final report")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    started = time.monotonic()
    report: dict[str, Any] = dict(result="fail", regressions=None, probes=[], overhead=None)
    try:
        tests = [
            "environments/cage2/test_hierarchy_diagnostics.py",
            "environments/cage2/test_latent_probe.py",
            "environments/cage2/test_experiment_supervisor.py",
            "environments/cage2/test_thermal_watchdog.py",
            "environments/cage2/test_concurrency_workflow.py",
            "dreamerrl/tests/functional/test_hierarchical_actor.py",
            "dreamerrl/tests/functional/test_hierarchical_amp.py",
        ]
        pytest = [
            sys.executable, "-B", "-m", "pytest", "-q", "--assert=plain", "--rootdir=.", "--confcutdir=.",
            "--import-mode=importlib", "--noconftest", "-c", "/dev/null", "-o", f"pythonpath={PROJECT}",
            "-p", "no:cacheprovider", "--basetemp", str(root / "pytest"), "-k", "not historical", *tests,
        ]
        log = root / "regressions.log"
        status = run_logged(pytest, log, 600)
        report["regressions"] = dict(exit_status=status, log=str(log), tail=log.read_text().splitlines()[-6:])
        for index, checkpoint in enumerate(args.checkpoint, 1):
            destination = root / f"probe_{index:02d}"
            log = root / f"probe_{index:02d}.log"
            command = [
                sys.executable, "-B", str(PROJECT / "hierarchy_diagnostics.py"),
                "--checkpoint", str(checkpoint.resolve()), "--output", str(destination), "--episodes", "60",
            ]
            probe_status = run_logged(command, log, 180)
            path = destination / "report.json"
            probe: dict[str, Any] = dict(exit_status=probe_status, checkpoint=str(checkpoint), log=str(log))
            if probe_status == 0:
                data = json.loads(path.read_text())
                probe.update(
                    report=str(path), training_seed=data["training_seed"], split_pairs=data["split_pairs"],
                    goal_usage=data["goal_usage"], evaluation_returns=data["evaluation_returns"]["mean"],
                    added_goal_intervals=[row["added_goal_interval"] for row in data["components"]],
                )
            report["probes"].append(probe)
        report["overhead"] = estimate_overhead()
        passed = status == 0 and all(probe["exit_status"] == 0 for probe in report["probes"])
        if passed:
            report["result"] = "pass"
            report["readiness"] = (
                "Observation and offline analysis checks passed on synthetic and POPGym workloads. "
                "Learning stability and production GPU overhead remain to be measured on formal runs."
            )
        return int(not passed)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
        print(error, flush=True)
        return 1
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        publish(root / "final_report.json", report)
        print("REPORT", root / "final_report.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
