"""One exploratory fixed-budget host encoder pilot; never launches another experiment."""

import argparse
import csv
import json
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from host_aware_experiment import episode_interval
from host_experiment_support import ROOT, initialize_paired, load_config, sha256, verify_gate
from learner_cycle import optimizer_steps
from learning_curve import evaluate, write

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
from dreamerrl.training.trainer import DreamerTrainer

PILOT = ROOT / "environments/cage2/host_pilot_v1_config.json"


def score(report, baseline, update, env_steps):
    assert [r["seed"] for r in report["episodes"]] == [r["seed"] for r in baseline["episodes"]]
    differences = [a["total_return"] - b["total_return"] for a, b in zip(report["episodes"], baseline["episodes"])]
    return dict(
        update=update,
        env_steps=env_steps,
        mean_return=report["mean_return"],
        historical_mean=baseline["mean_return"],
        historical_paired_change=float(np.mean(differences)),
        conditional_episode_ci95=episode_interval(differences),
        per_seed_differences=differences,
        episode_return_sd=float(np.std([r["total_return"] for r in report["episodes"]], ddof=1)),
    )


def save_reports(run, rows, final):
    with (run / "scorecard.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "update",
                "environment_steps",
                "mean_return",
                "historical_mean",
                "paired_change",
                "episode_ci_low",
                "episode_ci_high",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["update"],
                    row["env_steps"],
                    row["mean_return"],
                    row["historical_mean"],
                    row["historical_paired_change"],
                    *row["conditional_episode_ci95"],
                ]
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([r["env_steps"] for r in rows], [r["mean_return"] for r in rows], marker="o", label="Host pilot seed 0")
    axis.plot(
        [r["env_steps"] for r in rows], [r["historical_mean"] for r in rows], marker="o", label="Historical flat seed 0"
    )
    axis.set(xlabel="Environment interactions", ylabel="Mean 30-step Blue return")
    axis.legend()
    figure.tight_layout()
    figure.savefig(run / "learning_curve.png", dpi=150)
    plt.close(figure)
    text = [
        f"Host-aware CAGE2 pilot: {final['stop_reason']}",
        f"Update {final['update']}; interactions {final['env_steps']}; mean return {final['mean_return']:.4f}.",
        "Historical final reference: -35.39 at 507450 interactions.",
        f"Elapsed: {final['elapsed_seconds']:.1f}s. Matched final budget: {final['matched_final_budget']}.",
        "Exploratory one-training-seed historical comparison; initialization and collection trajectories differ.",
        "Episode bootstrap intervals are conditional on these policies, not uncertainty across training runs.",
        "No held-out evaluation or leaderboard ranking claim. Stop for manual review; no further experiment queued.",
    ]
    (run / "final_report.txt").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = verify_gate(args.manifest)
    spec = json.loads(PILOT.read_text())
    assert manifest["pilot_config_sha256"] == sha256(PILOT) and manifest["execution"] == spec
    assert spec["arm"] == "cage2_host" and spec["training_seed"] == 0 and not spec["heldout_evaluation"]
    for entry in manifest["historical_evaluations"].values():
        assert sha256(entry["path"]) == entry["sha256"]
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GB10"
    torch.set_num_threads(2)
    run = args.run_dir.resolve()
    assert run.is_dir() and not (run / "latest.pt").exists() and not (run / "manifest.json").exists()
    started = time.monotonic()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    cfg = load_config("cage2_host", 0, run)
    trainer = DreamerTrainer(cfg)
    env = trainer.env
    assert isinstance(env, Cage2BlueEnv)
    rows = []
    recorded = set()
    wall_stop = False
    try:
        initial_hashes = initialize_paired(trainer)
        write(run / "manifest.json", manifest)
        write(run / "resolved_config.json", asdict(cfg))
        write(
            run / "initialization.json",
            dict(
                common_module_hashes=initial_hashes,
                source_checkpoint=None,
                training_seed=0,
                optimizer_state="fresh",
                replay="fresh",
                collection_rng_seed=123456,
            ),
        )
        write(
            run / "parameters.json",
            {
                name: sum(p.numel() for p in module.parameters())
                for name, module in [
                    ("encoder", trainer.world.encoder),
                    ("world", trainer.world),
                    ("actor", trainer.actor),
                    ("critic", trainer.critic),
                ]
            },
        )
        for _ in range(spec["initial_replay_episodes"]):
            trainer.collect_env_steps()
        assert trainer.total_env_steps == 480
        env.journal.clear()

        def evaluation(update, final=False):
            if update in recorded:
                return
            verify_gate(args.manifest)
            trainer.save_checkpoint(run / f"update_{update}.pt")
            report = evaluate(trainer, "actor", spec["development_seeds"])
            write(run / f"evaluation_{update}.json", report)
            baseline_update = str(update) if str(update) in manifest["historical_evaluations"] else "16899"
            historical = json.loads(Path(manifest["historical_evaluations"][baseline_update]["path"]).read_text())
            row = score(report, historical, update, trainer.total_env_steps)
            row["historical_budget_matched"] = str(update) in manifest["historical_evaluations"]
            rows.append(row)
            recorded.add(update)
            write(run / "learning_curve.json", rows)
            print("EVALUATION", update, trainer.total_env_steps, report["mean_return"], flush=True)

        with (run / "metrics.jsonl").open("a", buffering=1) as metrics:

            def monitor(record):
                nonlocal wall_stop
                update = trainer.completed_updates
                assert all(
                    math.isfinite(record[key])
                    for key in ("world_loss", "reward_loss", "recon_loss", "actor_loss", "critic_loss")
                )
                record["pilot_elapsed_seconds"] = time.monotonic() - started
                metrics.write(json.dumps(record, default=str) + "\n")
                assert not ({row["seed"] for row in env.journal} & set(spec["development_seeds"]))
                env.journal.clear()
                if update % 100 == 0:
                    verify_gate(args.manifest)
                if update in spec["evaluation_updates"]:
                    evaluation(update)
                elapsed = time.monotonic() - started
                if update == 1 or update % 10 == 0 or update == 25:
                    progress = dict(
                        update=update,
                        env_steps=trainer.total_env_steps,
                        elapsed_seconds=elapsed,
                        updates_per_second=update / max(elapsed, 1e-9),
                        optimizer_steps=optimizer_steps(trainer),
                        losses={
                            key: record[key]
                            for key in ("world_loss", "reward_loss", "recon_loss", "actor_loss", "critic_loss")
                        },
                    )
                    write(run / "progress.json", progress)
                    print("PROGRESS", json.dumps(progress), flush=True)
                wall_stop = stop_requested or elapsed >= spec["max_wall_seconds"] - spec["finalization_reserve_seconds"]
                return not wall_stop

            def calibration(record):
                return monitor(record) and trainer.completed_updates < spec["calibration_end_update"]

            trainer.train(
                spec["calibration_scheduler_target"],
                checkpoint_path=run / "latest.pt",
                checkpoint_every=spec["checkpoint_every"],
                on_update=calibration,
            )
            if not wall_stop:
                trainer.train(
                    spec["final_update"],
                    checkpoint_path=run / "latest.pt",
                    checkpoint_every=spec["checkpoint_every"],
                    on_update=monitor,
                )
        trainer.save_checkpoint(run / "latest.pt")
        evaluation(trainer.completed_updates, final=True)
        final = dict(
            update=trainer.completed_updates,
            env_steps=trainer.total_env_steps,
            mean_return=rows[-1]["mean_return"],
            elapsed_seconds=time.monotonic() - started,
            stop_reason="fixed_budget_complete"
            if trainer.completed_updates == spec["final_update"]
            else "wall_limit_or_stop",
            matched_final_budget=trainer.total_env_steps == spec["final_env_steps"],
            optimizer_steps=optimizer_steps(trainer),
            checkpoint=str(run / "latest.pt"),
            checkpoint_sha256=sha256(run / "latest.pt"),
            historical_final_reference=-35.39,
            comparisons=spec["comparison_limits"],
            training_seed=0,
            source_commit=manifest["source_commit"],
            source_tag=manifest["source_tag"],
        )
        write(run / "final_summary.json", final)
        save_reports(run, rows, final)
        write(run / "complete.json", dict(success=True, **final))
        print("PILOT_COMPLETE", json.dumps(final), flush=True)
    finally:
        env.close()
        trainer.tb.close()


if __name__ == "__main__":
    main()
