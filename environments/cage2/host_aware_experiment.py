"""Gated fixed-interaction, three-seed flat/host encoder ablation; no historical weights."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from host_experiment_support import CONFIG, ROOT, experiment_spec, initialize_paired, load_config, sha256, verify_gate
from learning_curve import evaluate, write
from thermal_stop import ThermalStop, checkpoint_if_requested

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
from dreamerrl.training.trainer import DreamerTrainer


def episode_interval(differences):
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(20261124)
    samples = rng.integers(len(values), size=(20000, len(values)))
    return np.quantile(values[samples].mean(1), [0.025, 0.975]).tolist()


def run_arm(manifest_path, root, arm, seed):
    manifest = verify_gate(manifest_path)
    spec = experiment_spec(manifest)
    run = root / f"{arm}_seed{seed}"
    run.mkdir(parents=True, exist_ok=False)
    cfg = load_config(arm, seed, run, spec)
    trainer = DreamerTrainer(cfg)
    env = trainer.env
    assert isinstance(env, Cage2BlueEnv)
    started = time.perf_counter()
    try:
        assert trainer.device.type == "cuda" and trainer.completed_updates == 0
        common_hashes = initialize_paired(trainer)
        write(run / "manifest.json", manifest)
        write(run / "resolved_config.json", asdict(cfg))
        write(
            run / "initialization.json",
            dict(
                common_module_hashes=common_hashes,
                seed=seed,
                collection_rng_seed=seed + 123456,
                source_checkpoint=None,
                optimizer_states="fresh",
                replay="fresh",
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
            checkpoint_if_requested(run, trainer)
            trainer.collect_env_steps()
        assert trainer.total_env_steps == 480
        env.journal.clear()
        curve = []
        targets = {(step - 480) // 30: step for step in spec["evaluation_env_steps"]}
        assert all(480 + 30 * update == step for update, step in targets.items())
        with (run / "metrics.jsonl").open("a", buffering=1) as metrics:

            def callback(record):
                update = trainer.completed_updates
                metrics.write(json.dumps(record, default=str) + "\n")
                assert not (
                    {row["seed"] for row in env.journal} & set(spec["development_seeds"] + spec["heldout_seeds"])
                )
                env.journal.clear()
                checkpoint_if_requested(run, trainer)
                if update % 100 == 0 or update in targets:
                    verify_gate(manifest_path)
                if update in targets:
                    assert trainer.total_env_steps == targets[update]
                    trainer.save_checkpoint(run / f"update_{update}.pt")
                    report = evaluate(trainer, "actor", spec["development_seeds"])
                    checkpoint_if_requested(run, trainer)
                    if cfg.ac.hierarchical:
                        report.update(training_seed=seed, arm=arm, update=update)
                        trainer.tb.add_scalar("evaluation/development_return", report["mean_return"], update)
                    write(run / f"development_{update}.json", report)
                    curve.append(
                        dict(
                            update=update,
                            env_steps=trainer.total_env_steps,
                            mean_return=report["mean_return"],
                            episode_return_sd=float(np.std([r["total_return"] for r in report["episodes"]], ddof=1)),
                        )
                    )
                    write(run / "development_curve.json", curve)
                    print("DEVELOPMENT", arm, seed, update, trainer.total_env_steps, report["mean_return"], flush=True)
                if update % 25 == 0:
                    print("PROGRESS", arm, seed, update, trainer.total_env_steps, flush=True)
                return update < spec["calibration_end_update"] if update <= spec["calibration_end_update"] else None

            # Same production training algorithm and two original LR horizons.
            checkpoint_if_requested(run, trainer)
            trainer.train(
                spec["calibration_scheduler_target"],
                checkpoint_path=run / "latest.pt",
                checkpoint_every=100,
                on_update=callback,
            )

            # The phase-1 stop callback must not stop the second phase at its first update.
            def main_callback(record):
                callback(record)

            checkpoint_if_requested(run, trainer)
            trainer.train(
                spec["final_update"], checkpoint_path=run / "latest.pt", checkpoint_every=100, on_update=main_callback
            )
        checkpoint_if_requested(run, trainer)
        assert trainer.total_env_steps == spec["final_env_steps"]
        verify_gate(manifest_path)
        write(
            run / "complete.json",
            dict(
                arm=arm,
                training_seed=seed,
                update=trainer.completed_updates,
                env_steps=trainer.total_env_steps,
                elapsed_seconds=time.perf_counter() - started,
                primary_development_return=curve[-1]["mean_return"],
                development_auc=float(np.trapz([r["mean_return"] for r in curve], [r["env_steps"] for r in curve])),
                checkpoint=str(run / "latest.pt"),
                config_sha256=sha256(CONFIG),
                manifest_sha256=sha256(manifest_path),
                **(dict(checkpoint_status="saved", termination_reason="fixed_budget_complete",
                         hierarchy_config_sha256=manifest["hierarchy_config_sha256"])
                   if cfg.ac.hierarchical else {}),
            ),
        )
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
    torch.cuda.empty_cache()


def evaluate_heldout(manifest_path, root):
    """Final models only, after every preregistered training arm is complete."""
    verify_gate(manifest_path)
    spec = json.loads(CONFIG.read_text())
    for item in spec["sequential_run_order"]:
        completed = json.loads((root / f"{item['arm']}_seed{item['seed']}" / "complete.json").read_text())
        assert completed["update"] == spec["final_update"] and completed["env_steps"] == spec["final_env_steps"]
    for item in spec["sequential_run_order"]:
        run = root / f"{item['arm']}_seed{item['seed']}"
        if (run / "heldout.json").exists():
            raise ValueError("Refuse to overwrite a held-out evaluation")
        trainer = DreamerTrainer.from_checkpoint(run / "latest.pt", cuda=True, log_dir=run / "heldout_logs")
        try:
            report = evaluate(trainer, "actor", spec["heldout_seeds"])
            write(run / "heldout.json", report)
        finally:
            trainer.env.venv.close()
            trainer.tb.close()
        del trainer
        torch.cuda.empty_cache()


def report_comparison(root):
    spec = json.loads(CONFIG.read_text())
    results = []
    for seed in spec["training_seeds"]:
        pair = {}
        for arm in spec["arms"]:
            run = root / f"{arm}_seed{seed}"
            pair[arm] = json.loads((run / "heldout.json").read_text())
        flat, host = pair["flat"]["episodes"], pair["cage2_host"]["episodes"]
        learning = {arm: json.loads((root / f"{arm}_seed{seed}" / "complete.json").read_text()) for arm in spec["arms"]}
        assert [r["seed"] for r in flat] == [r["seed"] for r in host] == spec["heldout_seeds"]
        differences = [h["total_return"] - f["total_return"] for f, h in zip(flat, host)]
        results.append(
            dict(
                training_seed=seed,
                flat_mean=pair["flat"]["mean_return"],
                host_mean=pair["cage2_host"]["mean_return"],
                paired_change=float(np.mean(differences)),
                conditional_episode_ci95=episode_interval(differences),
                per_episode_differences=differences,
                learning_curve_auc={arm: value["development_auc"] for arm, value in learning.items()},
                training_wall_seconds={arm: value["elapsed_seconds"] for arm, value in learning.items()},
                heldout_evaluation_seconds={arm: value["seconds"] for arm, value in pair.items()},
            )
        )
    changes = np.asarray([r["paired_change"] for r in results])
    halfwidth = 4.302652729911275 * float(changes.std(ddof=1)) / np.sqrt(3)
    write(
        root / "comparison.json",
        dict(
            training_seed_pairs=results,
            mean_paired_change=float(changes.mean()),
            training_seed_t_ci95=[float(changes.mean() - halfwidth), float(changes.mean() + halfwidth)],
            warning=(
                "Three independent training seeds; episode intervals are conditional on fitted agents. "
                "No leaderboard ranking claim."
            ),
            historical_reference=spec["historical_reference"],
        ),
    )


def execution_order(spec, arm=None, seed=None):
    """Select one existing arm/seed or retain the full preregistered order."""
    if (arm is None) != (seed is None):
        raise ValueError("--arm and --seed must be supplied together")
    if arm is None:
        return spec["sequential_run_order"]
    item = {"arm": arm, "seed": seed}
    if item not in spec["sequential_run_order"]:
        raise ValueError("Arm/seed must belong to the configured execution schedule")
    return [item]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--arm", choices=["flat", "cage2_host"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        manifest = verify_gate(args.manifest, require_approval=not args.check_only)
        spec = experiment_spec(manifest)
        if manifest.get("hierarchy_config_sha256") and args.arm != "cage2_host":
            raise ValueError("Hierarchical execution requires a single explicit cage2_host seed")
        order = execution_order(spec, args.arm, args.seed)
    except ValueError as error:
        parser.error(str(error))
    if args.check_only:
        print(
            json.dumps(
                dict(
                    gate_passed=True,
                    source_tag=manifest["source_tag"],
                    source_commit=manifest["source_commit"],
                    execution_order=order,
                ),
                indent=2,
            )
        )
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GB10"
    torch.set_num_threads(2)
    root = args.output_root.resolve()
    if not root.is_relative_to(ROOT / "runs"):
        parser.error("Generated experiment artifacts must be under this checkout's runs/ directory")
    root.mkdir(parents=True, exist_ok=False)
    write(root / "execution_selection.json", dict(execution_order=order, single_arm=args.arm is not None))
    for item in order:
        try:
            run_arm(args.manifest, root, item["arm"], item["seed"])
        except ThermalStop as stop:
            print(stop, flush=True)
            raise SystemExit(75) from None
    if args.arm is None:
        evaluate_heldout(args.manifest, root)
        report_comparison(root)
    print("EXPERIMENT_COMPLETE", root, flush=True)


if __name__ == "__main__":
    main()
