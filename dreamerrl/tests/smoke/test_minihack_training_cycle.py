import pytest
import torch

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


@pytest.mark.parametrize("parallel", [False, True])
def test_minihack_collect_replay_and_both_updates(parallel):
    cfg = make_default_config()
    cfg.env.env_id = "MiniHack-Room-5x5-v0"
    cfg.env.num_envs = 2
    cfg.env.max_episode_steps = 6
    cfg.env.parallel = parallel
    cfg.world.disable_aux_losses = True
    cfg.world.deter_size = 16
    cfg.world.num_classes = 3
    cfg.world.stoch_size = 4
    cfg.world.hidden_size = 16
    cfg.world.imagination_horizon = 3
    cfg.ac.actor_hidden = cfg.ac.critic_hidden = 16
    cfg.train.cuda = False
    cfg.train.use_amp = False
    cfg.train.batch_size = 2
    cfg.train.collect_steps = 12
    cfg.train.seq_len = 4
    cfg.train.freeze_actor_critic_steps = 0
    cfg.log.enable_wandb = False
    trainer = DreamerTrainer(cfg)
    try:
        for update in range(2):
            trainer.collect_env_steps()
            assert trainer.world_state.h.grad_fn is None
            batch = trainer.replay.sample(2)
            assert batch["obs"].shape == (2, 4, trainer.env.obs_dim)
            metrics = trainer.update_world_model(batch, update)
            assert torch.isfinite(metrics.total_loss)
            before = [p.detach().clone() for p in trainer.actor.parameters()]
            actor_loss, critic_loss = trainer.update_actor_critic(batch, update)
            assert torch.isfinite(torch.tensor([actor_loss, critic_loss])).all()
            assert any(not torch.equal(old, new) for old, new in zip(before, trainer.actor.parameters()))
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
