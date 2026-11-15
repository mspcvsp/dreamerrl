"""Compare a saved policy with its initial policy and random actions on fresh test seeds."""

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from dreamerrl.eval.minihack_eval import compare_returns, evaluate_minihack
from dreamerrl.training.run_monitor import write_json
from dreamerrl.training.trainer import DreamerTrainer


def evaluate_checkpoints(
    checkpoint, initial_checkpoint, *, episodes=100, seed_start=1000000, cpu=False
) -> dict[str, Any]:
    if episodes < 2 or seed_start < 0:
        raise ValueError("Final evaluation needs at least two episodes and a nonnegative seed")
    trainer = DreamerTrainer.from_checkpoint(checkpoint, cuda=False if cpu else None)
    initial = None
    try:
        initial = DreamerTrainer.from_checkpoint(initial_checkpoint, cuda=trainer.device.type == "cuda")
        if initial.completed_updates != 0:
            raise ValueError("The untrained checkpoint must be from update zero")
        if (
            trainer.cfg.env != initial.cfg.env
            or trainer.cfg.world != initial.cfg.world
            or trainer.cfg.ac != initial.cfg.ac
        ):
            raise ValueError("Trained and initial checkpoints must use matching environment/model configurations")
        seeds = list(range(seed_start, seed_start + episodes))
        validation = (
            trainer.run_state.get("minihack_monitor", {})
            .get("baselines", {})
            .get("spec", {})
            .get("evaluation_seeds", [])
        )
        if set(seeds) & set(validation):
            raise ValueError("Final test seeds must not overlap the run's validation seeds")
        trained = evaluate_minihack(trainer.cfg, trainer.world, trainer.actor, seeds)
        untrained = evaluate_minihack(initial.cfg, initial.world, initial.actor, seeds)
        random_returns = evaluate_minihack(trainer.cfg, trainer.world, trainer.actor, seeds, random_policy=True)
        return dict(
            update=trainer.completed_updates,
            env_steps=trainer.total_env_steps,
            train_seed=trainer.cfg.train.seed,
            evaluation_seeds=seeds,
            trained_returns=trained,
            untrained_returns=untrained,
            random_returns=random_returns,
            trained_mean=mean(trained),
            untrained_mean=mean(untrained),
            random_mean=mean(random_returns),
            vs_random=compare_returns(trained, random_returns),
            vs_untrained=compare_returns(trained, untrained),
            interval_method="paired normal approximation, descriptive; confirm across independent training seeds",
        )
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
        if initial is not None:
            initial.env.venv.close()
            initial.tb.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_checkpoints(
        args.checkpoint, args.initial_checkpoint, episodes=args.episodes, seed_start=args.seed_start, cpu=args.cpu
    )
    path = args.output or args.checkpoint.parent / "test_evaluation.json"
    write_json(path, result)
    print(
        f"Test mean: trained={result['trained_mean']:.4f} "
        f"untrained={result['untrained_mean']:.4f} random={result['random_mean']:.4f}"
    )
    print(f"Paired differences: random={result['vs_random']} untrained={result['vs_untrained']}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
