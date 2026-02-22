import copy

import tyro
from dreamer.config import DreamerConfig
from dreamer.seed import set_global_seeds
from dreamer.trainer import DreamerTrainer

import wandb


def run_single_seed(base_cfg: DreamerConfig, total_updates: int, seed: int, wandb_project: str, wandb_group: str):

    cfg = copy.deepcopy(base_cfg)
    cfg.train.seed = seed

    set_global_seeds(seed)
    cfg.init_run_name()

    wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=cfg.log.run_name,
        config=cfg.__dict__,
    )

    trainer = DreamerTrainer(cfg)
    trainer.train(total_updates)

    wandb.finish()


def main(
    cfg: DreamerConfig,
    total_updates: int = 2000,
    seeds: list[int] = [0],
    wandb_project: str = "dreamer",
    wandb_group: str = "popgym",
):
    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        run_single_seed(cfg, total_updates, seed, wandb_project, wandb_group)


if __name__ == "__main__":
    tyro.cli(main)
