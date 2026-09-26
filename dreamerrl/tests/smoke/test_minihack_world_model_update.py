import pytest
import torch

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


@pytest.mark.minihack_smoke
def test_minihack_world_model_update():
    cfg = make_default_config()

    cfg.env.env_id = "MiniHack-Room-5x5-v0"
    cfg.env.num_envs = 4
    cfg.env.max_episode_steps = 50

    cfg.train.cuda = False
    cfg.train.collect_steps = 60

    trainer = DreamerTrainer(cfg)

    trainer.collect_env_steps()

    for i, ep in enumerate(trainer.replay.episodes):
        print(f"ep={i:2d} ", f"len={ep['obs'].shape[0]:2d} ", f"dones={ep['done'].sum().item():.0f}")

    batch = trainer.replay.sample(
        batch_size=cfg.train.batch_size,
        seed=cfg.train.seed,
    )

    metrics = trainer.update_world_model(
        batch,
        update_idx=0,
    )

    assert torch.isfinite(metrics.total_loss)
