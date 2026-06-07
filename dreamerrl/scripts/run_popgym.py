import argparse

from dreamerrl.eval.popgym_eval import (
    aggregate_popgym_results,
    evaluate_popgym,
    train_popgym_seed,
)
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="popgym-RepeatFirstEasy-v0")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    results = []

    for seed in args.seeds:
        cfg = make_default_config()
        cfg.env.env_id = args.env
        cfg.env.seed = seed
        cfg.train.seed = seed
        cfg.train.enable_wandb = False

        trainer = DreamerTrainer(cfg)
        metrics = train_popgym_seed(trainer, steps=args.steps)
        results.append(metrics)

    summary = aggregate_popgym_results(results)

    print("\n=== Training Summary ===")
    print("World Model CV:", summary["wm_cv"])
    print("Actor CV:", summary["actor_cv"])
    print("Critic CV:", summary["critic_cv"])
    print("Action KL:", summary["action_kl"])
    print("Mean Return:", summary["mean_return"])

    # Optional deterministic evaluation
    env = trainer.env
    eval_stats = evaluate_popgym(env, trainer.world, trainer.actor, episodes=args.episodes)
    print("\n=== Deterministic Evaluation ===")
    print(eval_stats)


if __name__ == "__main__":
    main()
