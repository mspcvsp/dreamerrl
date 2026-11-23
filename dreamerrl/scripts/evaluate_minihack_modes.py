"""Diagnose action sampling versus latent sampling on fixed held-out levels."""

import argparse
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Any

from dreamerrl.eval.minihack_eval import compare_returns, evaluate_minihack
from dreamerrl.training.run_monitor import write_json
from dreamerrl.training.trainer import DreamerTrainer

MODES = [(False, False), (True, False), (False, True), (True, True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000000)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 2 or min(args.seed_start, args.sampling_seed) < 0:
        parser.error("At least two episodes and nonnegative seeds are required")
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    report: dict[str, Any] = dict(evaluation_seeds=seeds, sampling_seed=args.sampling_seed, checkpoints=[])
    for path in args.checkpoint:
        with TemporaryDirectory(prefix="dreamerrl-modes-") as log_dir:
            trainer = DreamerTrainer.from_checkpoint(path, cuda=False if args.cpu else None, log_dir=log_dir)
            try:
                validation = (
                    trainer.run_state.get("minihack_monitor", {})
                    .get("baselines", {})
                    .get("spec", {})
                    .get("evaluation_seeds", [])
                )
                if set(seeds) & set(validation):
                    raise ValueError("Diagnostic test seeds must not overlap validation seeds")
                entry: dict[str, Any] = dict(
                    checkpoint=str(path.resolve()),
                    train_seed=trainer.cfg.train.seed,
                    update=trainer.completed_updates,
                    env_steps=trainer.total_env_steps,
                    device=str(trainer.device),
                    modes=[],
                )
                random_returns = evaluate_minihack(trainer.cfg, trainer.world, trainer.actor, seeds, random_policy=True)
                entry["random_returns"] = random_returns
                entry["random_mean"] = mean(random_returns)
                for actions, latents in MODES:
                    returns = evaluate_minihack(
                        trainer.cfg,
                        trainer.world,
                        trainer.actor,
                        seeds,
                        sample_actions=actions,
                        sample_latents=latents,
                        sampling_seed=args.sampling_seed,
                    )
                    mode = dict(
                        actions="sampled" if actions else "greedy",
                        latents="sampled" if latents else "mode",
                        returns=returns,
                        mean=mean(returns),
                        vs_random=compare_returns(returns, random_returns),
                    )
                    entry["modes"].append(mode)
                    print(
                        f"seed={entry['train_seed']} actions={mode['actions']} "
                        f"latents={mode['latents']} mean={mode['mean']:.4f}",
                        flush=True,
                    )
                report["checkpoints"].append(entry)
                write_json(args.output, report)
            finally:
                trainer.env.venv.close()
                trainer.tb.close()


if __name__ == "__main__":
    main()
