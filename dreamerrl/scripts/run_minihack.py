import argparse

import torch

from dreamerrl.models.aux_objectives import AUX_OBJECTIVES
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


def main():
    parser = argparse.ArgumentParser(description="MiniHack baseline and single-objective ablation")
    parser.add_argument("--env", default="MiniHack-Room-5x5-v0")
    parser.add_argument("--aux", choices=["none", "novelty"], default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    cfg = make_default_config()

    # -----------------------------------------------------
    # Environment: MiniHack symbolic mode
    # -----------------------------------------------------
    cfg.env.env_id = args.env
    cfg.env.num_envs = 8
    cfg.env.max_episode_steps = 200

    cfg.env.parallel = False
    cfg.env.deterministic = False
    cfg.env.seed = args.seed

    # -----------------------------------------------------
    # World Model: establish the baseline before auxiliary ablations.
    # -----------------------------------------------------
    cfg.world.disable_aux_losses = args.aux == "none"
    cfg.world.aux_objectives = [] if args.aux == "none" else [AUX_OBJECTIVES[args.aux]]

    # DreamerV3 latent sizes (moderate)
    cfg.world.deter_size = 128
    cfg.world.stoch_size = 32
    cfg.world.num_classes = 32
    cfg.world.hidden_size = 256
    cfg.world.imagination_horizon = 5

    # -----------------------------------------------------
    # Training hyperparameters
    # -----------------------------------------------------
    cfg.train.seed = args.seed
    cfg.train.cuda = torch.cuda.is_available() and not args.cpu
    cfg.train.batch_size = 16
    cfg.train.collect_steps = 250
    cfg.train.seq_len = 50

    cfg.train.model_lr = 3e-4
    cfg.train.actor_lr = 3e-4
    cfg.train.critic_lr = 3e-4

    cfg.train.random_exploration_steps = 1000
    cfg.train.freeze_actor_critic_steps = 0

    # -----------------------------------------------------
    # Logging (TensorBoard)
    # -----------------------------------------------------
    cfg.log.enable_wandb = False
    cfg.log.tb_logdir = "./logs"
    cfg.log.run_name = f"{args.env}_{args.aux}_seed{args.seed}"

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------
    trainer = DreamerTrainer(cfg)

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    try:
        trainer.train(total_updates=args.updates)
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


if __name__ == "__main__":
    main()
