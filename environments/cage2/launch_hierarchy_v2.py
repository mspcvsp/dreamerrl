"""Gate, supervise, and report the three-seed hierarchical CAGE2 experiment."""

import argparse
import json
import math
import statistics
from pathlib import Path

import experiment_supervisor
from host_experiment_support import HIERARCHY_CONFIG, experiment_spec, sha256, verify_gate
from thermal_stop import publish


def training_metrics(path):
    totals, counts = {}, {}
    last = {}
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                last = row
                values = {
                    **row.get("credit_assignment", {}),
                    **{key: value for key, value in row.items() if key.endswith(("_loss", "_grad_norm"))},
                }
                for key, value in values.items():
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        totals[key] = totals.get(key, 0) + value
                        counts[key] = counts.get(key, 0) + 1
    occupancy = {key: value for key, value in totals.items() if key.startswith("goal_") and key.endswith("_decisions")}
    n = sum(occupancy.values())
    return dict(
        finite_update_means={key: value / counts[key] for key, value in totals.items()},
        goal_decision_counts=occupancy,
        pooled_goal_entropy=-sum((count / n) * math.log(count / n) for count in occupancy.values() if count > 0),
        last_optimizer_cadence=last.get("cadence_total"),
    )


def report(root, protocol, spec):
    execution = json.loads((root / "execution_summary.json").read_text())
    reference = protocol["reference_return"]
    results, matched = [], []
    for job in execution["jobs"]:
        run = Path(job["run"])
        curve = sorted(
            (json.loads(path.read_text()) for path in run.glob("development_*.json") if path.stem[12:].isdigit()),
            key=lambda row: row["update"],
        )
        checkpoint = job.get("checkpoint") or {}
        path = run / "latest.pt"
        at_budget = (
            job["state"] == "completed" and job["env_steps"] == spec["final_env_steps"]
            and checkpoint.get("checkpoint_status") == "saved" and path.is_file()
            and bool(curve) and curve[-1]["update"] == spec["final_update"]
        )
        score = curve[-1]["mean_return"] if at_budget else None
        if score is not None:
            matched.append(score)
        seconds = job.get("runtime_seconds")
        result = dict(
            job=job, evaluation_curve=curve, matched_budget=at_budget,
            final_matched_return=score, delta_vs_historical=None if score is None else score - reference,
            updates_per_second=job.get("updates", 0) / seconds if seconds else None,
            env_steps_per_second=job.get("env_steps", 0) / seconds if seconds else None,
            checkpoint_bytes=path.stat().st_size if path.is_file() else None,
            checkpoint_sha256=sha256(path) if path.is_file() else None,
            final_save_acknowledged=checkpoint.get("checkpoint_status") == "saved",
            diagnostics=training_metrics(run / "metrics.jsonl"),
        )
        publish(run.parent.parent / "final_report.json", result)
        results.append(result)
    samples = {}
    thermal_log = root / "thermal.jsonl"
    if thermal_log.is_file():
        with thermal_log.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("event") == "sample":
                    for key, value in row.items():
                        if isinstance(value, (int, float, bool)):
                            samples.setdefault(key, []).append(value)
    publish(root / "final_report.json", dict(
        execution=execution, runs=results, historical_reference=reference,
        historical_reference_checkpoint_sha256=protocol["reference_checkpoint_sha256"],
        matched_seeds=len(matched), matched_mean=statistics.mean(matched) if matched else None,
        matched_sample_sd=statistics.stdev(matched) if len(matched) > 1 else None,
        matched_range=[min(matched), max(matched)] if matched else None,
        thermal={key: dict(min=min(values), mean=statistics.mean(values), max=max(values), samples=len(values))
                 for key, values in samples.items()},
    ))


def supervisor_args(manifest, root, protocol):
    arguments = ["--output-root", str(root)]
    for key in ("concurrency", "warn_c", "stop_c", "sustain_seconds", "wall_seconds", "checkpoint_timeout"):
        arguments.extend(("--" + key.replace("_", "-"), str(protocol[key])))
    for seed in protocol["training_seeds"]:
        arguments.extend(("--job", str(manifest), "cage2_host", str(seed)))
    return arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    manifest_path, root = args.manifest.resolve(), args.output_root.resolve()
    try:
        manifest = verify_gate(manifest_path, require_approval=not args.check_only)
        if manifest.get("hierarchy_config_sha256") != sha256(HIERARCHY_CONFIG):
            raise ValueError("A sealed hierarchy-v2 manifest is required")
        spec = experiment_spec(manifest)
        protocol = manifest["execution"]
    except ValueError as error:
        parser.error(str(error))
    arguments = supervisor_args(manifest_path, root, protocol)
    if args.check_only:
        plan = dict(source=manifest["source_commit"], protocol=protocol, supervisor_args=arguments)
        print(json.dumps(plan, indent=2))
        return 0
    code = experiment_supervisor.main(arguments)
    report(root, protocol, spec)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
