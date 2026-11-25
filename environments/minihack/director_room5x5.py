"""Seal, supervise and report the opt-in MiniHack Director comparison."""

import argparse
import importlib.metadata
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAGE = ROOT / "environments/cage2"
sys.path.insert(0, str(CAGE))

from experiment_supervisor import CHECKPOINT_EXIT, Job, supervise
from host_experiment_support import git, sha256
from thermal_stop import ThermalStop, checkpoint_if_requested
from thermal_watchdog import NvidiaTelemetry
from dreamerrl.scripts.run_minihack import main as train_minihack
from dreamerrl.training.run_monitor import write_json

TAG = "dreamerv3-director-room5x5-v1"
CONFIG = Path(__file__).with_suffix(".json")


def snapshot():
    if git("status", "--porcelain", root=ROOT):
        raise ValueError("Experiment source checkout is dirty")
    revision = git("rev-parse", "HEAD", root=ROOT)
    if git("cat-file", "-t", TAG, root=ROOT) != "tag" or git("rev-parse", TAG + "^{}", root=ROOT) != revision:
        raise ValueError("Experiment source must match the annotated tag")
    fingerprints = {
        name: sha256(ROOT / name)
        for name in git("ls-files", root=ROOT).splitlines()
        if (ROOT / name).is_file()
    }
    vendor = CAGE / "vendor/cage-challenge-2"
    vendor_revision = git("rev-parse", "HEAD:environments/cage2/vendor/cage-challenge-2", root=ROOT)
    if git("rev-parse", "HEAD", root=vendor) != vendor_revision or git("status", "--porcelain", root=vendor):
        raise ValueError("CybORG dependency checkout differs from the pinned gitlink")
    import CybORG
    import dreamerrl

    if not Path(CybORG.__file__).resolve().is_relative_to(vendor) or not Path(
        dreamerrl.__file__
    ).resolve().is_relative_to(ROOT):
        raise ValueError("Imported dependencies or project code are outside the tagged checkout")
    return dict(
        schema_version=1, source_tag=TAG, tag_object=git("rev-parse", TAG, root=ROOT), source_commit=revision,
        source_fingerprints=fingerprints, execution=json.loads(CONFIG.read_text()),
        checkpoint_provenance="Three fresh initializations. No input checkpoint.",
        dependencies=dict(
            python=sys.version, uv=subprocess.check_output(["uv", "--version"], text=True).strip(),
            package_versions=sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
            vendor_revision=vendor_revision, lock_sha256=sha256(ROOT / "uv.lock"),
            native_wheels={p.name: sha256(p) for p in sorted((CAGE / "wheels").glob("*spark*.whl"))},
        ),
    )


def verify_manifest(path, require_approval):
    manifest = json.loads(path.read_text())
    checkout = Path(git("rev-parse", "--show-toplevel", root=path.parent))
    relative = path.relative_to(checkout)
    committed = subprocess.check_output(["git", "-C", str(checkout), "show", f"HEAD:{relative}"])
    if committed != path.read_bytes():
        raise ValueError("Execution manifest must match its review commit")
    approval = manifest.pop("launch_approval")
    if require_approval and not approval:
        raise ValueError("Record explicit formal training approval in the reviewed execution manifest")
    if manifest != snapshot():
        raise ValueError("Sealed source, dependency or execution fingerprints differ")
    return manifest["execution"]


def worker(run, arm, seed, spec):
    args = [
        "--env", spec["environment"], "--actor-mode", arm, "--seed", str(seed),
        "--run-dir", str(run), "--dashboard", "off", "--require-cuda", "--amp",
    ]
    for field in (
        "updates", "num_envs", "collect_steps", "episode_steps", "batch_size", "seq_len", "burn_in",
        "imagination_horizon", "random_exploration_steps", "warmup_updates", "eval_every",
        "eval_episodes", "eval_seed_start", "checkpoint_every",
    ):
        args += ["--" + field.replace("_", "-"), str(spec[field])]
    try:
        train_minihack(argv=args, on_update=checkpoint_if_requested)
    except ThermalStop:
        return CHECKPOINT_EXIT
    checkpoint = run / "latest.pt"
    end = json.loads((run / "evaluation.json").read_text())
    if end["update"] != spec["updates"]:
        raise RuntimeError("Worker exited before the matched evaluation budget")
    write_json(run / "complete.json", dict(
        update=end["update"], env_steps=end["env_steps"], checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
    ))
    return 0


def report(root, summary):
    import matplotlib
    from dreamerrl.eval.posterior_diagnostics import diagnose_training_replay
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {}
    metrics = {}
    fig, ax = plt.subplots()
    for job in summary["jobs"]:
        arm = job["arm"]
        run = Path(job["run"])
        log = run / "metrics.jsonl"
        entries = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        curve = {r["env_steps"]: r for r in entries if r["event"] == "evaluation"}
        curves[arm] = curve
        train = [r for r in entries if r["event"] == "train"]
        diagnostics = {}
        for row in train:
            for key, value in row["credit_assignment"].items():
                if value is not None:
                    diagnostics.setdefault(key, []).append(value)
        metrics[arm] = dict(
            state=job["state"], training_seed=job["seed"], final_env_steps=job.get("env_steps", 0),
            mean_training_metrics={k: statistics.mean(v) for k, v in diagnostics.items()},
            goal_decision_counts={k: sum(v) for k, v in diagnostics.items() if k.endswith("_decisions") and k.startswith("goal_")},
            goal_optimizer_updates=sum(diagnostics.get("goal_optimizer_steps", [])),
            optimizer_cadence=train[-1]["cadence_total"] if train else {},
            top_allocated_cuda_mib=max((r["cuda_memory_mb"] or 0 for r in train), default=0),
            runtime_seconds=job.get("runtime_seconds"),
            env_steps_per_second=job.get("env_steps", 0) / job["runtime_seconds"] if job.get("runtime_seconds") else None,
        )
        checkpoint = run / "latest.pt"
        if checkpoint.is_file() and train:
            try:
                metrics[arm]["offline_posterior"] = diagnose_training_replay(checkpoint, batches=4, batch_size=4)
            except Exception as error:
                metrics[arm]["offline_posterior_error"] = repr(error)
        if curve:
            ax.plot(sorted(curve), [curve[n]["trained_mean"] for n in sorted(curve)], label=arm)
    if any(curves.values()):
        ax.legend()
    ax.set(xlabel="Environment transitions", ylabel="Mean greedy evaluation return")
    fig.tight_layout()
    fig.savefig(root / "learning_curves.png", dpi=150)
    plt.close(fig)
    common = max(set.intersection(*(set(rows) for rows in curves.values())), default=None) if curves else None
    matched = {}
    for arm, rows in curves.items():
        if common is not None:
            row = rows[common]
            returns = row["trained_returns"]
            matched[arm] = dict(
                mean=statistics.mean(returns), episode_sd=statistics.stdev(returns), episodes=len(returns),
                goal_completion_rate=row.get("goal_completion_rate"), random_mean=row["random_mean"],
            )
    recommendation = (
        "Repeat Room 5x5 with independent training seeds before progressing to Corridor R2. "
        "Check sampled behavior and goal semantics against matched controls."
    )
    write_json(root / "final_report.json", dict(
        common_env_steps=common, matched=matched, arms=metrics, curves=curves,
        recommendation=recommendation, supervisor=summary,
    ))
    lines = [
        "# Director-inspired Room 5x5", "", f"Latest shared evaluation budget: {common} environment transitions.", "",
        "| Arm | Mean task return | Episode SD | Episodes | Goal completion rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, row in matched.items():
        completion = row["goal_completion_rate"]
        label = "—" if completion is None else f"{completion:.1%}"
        lines.append(f"| {arm} | {row['mean']:.3f} | {row['episode_sd']:.3f} | {row['episodes']} | {label} |")
    lines += [
        "", "![Learning curves](learning_curves.png)", "",
        "Episode SD measures variation across test episodes for one fitted policy. Each arm uses one training seed.",
        "Per-update goal and optimizer metrics are in the arm JSONL logs. The supervisor records GPU and host telemetry in thermal.jsonl.",
        "", recommendation, "",
    ]
    (root / "final_report.md").write_text("\n".join(lines))


def launch(manifest, output):
    spec = verify_manifest(manifest, require_approval=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Formal AMP runs require CUDA")
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Run output must be inside the tagged checkout's ignored runs directory")
    telemetry = NvidiaTelemetry(spec["gpu"])()
    if not 0 < (telemetry["temperature_c"] or 0) < spec["warn_c"]:
        raise RuntimeError("GPU temperature is unavailable or above the launch threshold")
    available = telemetry["host_memory_available_kib"] or 0
    concurrency = min(
        len(spec["arms"]), available // (spec["minimum_available_memory_gib_per_arm"] * 2**20),
        max(1, len(os.sched_getaffinity(0)) // 2),
    )
    gpu_free = telemetry["gpu_memory_free_mib"]
    if gpu_free is not None:
        concurrency = min(concurrency, gpu_free // (spec["minimum_available_memory_gib_per_arm"] * 1024))
    if concurrency < 1 or shutil.disk_usage(ROOT).free < spec["minimum_available_disk_gib"] * 2**30:
        raise RuntimeError("Insufficient memory or disk space")
    output.mkdir(parents=True)
    shutil.copyfile(manifest, output / "execution_manifest.json")
    write_json(output / "preflight.json", dict(telemetry=telemetry, concurrency=concurrency))
    jobs = []
    for index, (arm, seed) in enumerate(spec["arms"].items(), 1):
        job_root = output / f"job_{index:03d}"
        run = job_root / "results" / f"{arm}_seed{seed}"
        job_root.mkdir()
        command = [sys.executable, "-u", str(Path(__file__)), "worker", arm, str(run), str(manifest)]
        jobs.append((index, Job(run, command, manifest, arm, seed)))
    watchdog = [
        sys.executable, "-u", str(CAGE / "thermal_watchdog.py"), "--supervisor-control", str(output),
        "--log", str(output / "thermal.jsonl"), "--gpu", spec["gpu"],
        "--warn-c", str(spec["warn_c"]), "--stop-c", str(spec["stop_c"]),
        "--sustain-seconds", str(spec["sustain_seconds"]),
    ]
    summary, status = supervise(
        output, jobs, watchdog, concurrency, readiness_timeout=60, checkpoint_timeout=300,
        wall_seconds=spec["max_wall_seconds"],
    )
    report(output, summary)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "check", "launch"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        if name == "launch":
            command.add_argument("output", type=Path)
    command = commands.add_parser("worker")
    command.add_argument("arm")
    command.add_argument("output", type=Path)
    command.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "seal":
        with args.manifest.open("x") as stream:
            json.dump(dict(**snapshot(), launch_approval=None), stream, indent=2)
            stream.write("\n")
    elif args.command == "check":
        verify_manifest(args.manifest.resolve(), require_approval=False)
        print("Verified sealed Room 5x5 manifest")
    elif args.command == "launch":
        return launch(args.manifest.resolve(), args.output.resolve())
    else:
        spec = verify_manifest(args.manifest.resolve(), require_approval=True)
        return worker(args.output, args.arm, spec["arms"][args.arm], spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
