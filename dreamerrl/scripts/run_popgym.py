"""Optional discrete-cue PopGym training and held-out evaluation."""

import argparse

import torch

from dreamerrl.env.env_factory import make_env
from dreamerrl.eval.popgym_eval import aggregate_popgym_results, evaluate_popgym, train_popgym_seed
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="popgym-RepeatFirstEasy-v0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    results = []
    for seed in args.seeds:
        cfg = make_default_config()
        cfg.env.env_id, cfg.env.seed, cfg.train.seed = args.env, seed, seed
        cfg.world.disable_aux_losses = True
        cfg.train.cuda = torch.cuda.is_available() and not args.cpu
        cfg.train.use_amp = False
        cfg.train.enable_fused_adamw_kernels = False
        cfg.train.freeze_actor_critic_steps = 0
        cfg.train.collect_steps = cfg.env.max_episode_steps
        cfg.train.seq_len = min(20, cfg.env.max_episode_steps)
        cfg.log.enable_wandb = False
        cfg.log.run_name = f"{args.env}_seed{seed}"
        trainer = DreamerTrainer(cfg)
        evaluation_env = make_env(cfg.env, trainer.device)
        try:
            untrained = evaluate_popgym(
                evaluation_env,
                trainer.world,
                trainer.actor,
                args.episodes,
                task_horizon=cfg.env.max_episode_steps if cfg.env.finite_horizon else None,
            )
            metrics = train_popgym_seed(trainer, steps=args.steps)
            results.append(metrics)
            trained = evaluate_popgym(
                evaluation_env,
                trainer.world,
                trainer.actor,
                args.episodes,
                task_horizon=cfg.env.max_episode_steps if cfg.env.finite_horizon else None,
            )
            print(
                f"seed={seed} env_steps={metrics['env_steps']} "
                f"untrained_return={untrained['mean']:.4f} trained_return={trained['mean']:.4f}"
            )
        finally:
            evaluation_env.venv.close()
            trainer.env.venv.close()
            trainer.tb.close()
    print("Training summary (native rewards):", aggregate_popgym_results(results))


if __name__ == "__main__":
    main()
