import pytest
import torch

from dreamerrl.training.replay_buffer import DreamerReplayBuffer
from dreamerrl.training.test_trainer import TestDreamerTrainer


@pytest.mark.trainer
def test_world_model_training_step(world_model, device):
    B = 1
    obs_dim = world_model.flat_obs_dim
    action_dim = 4

    buffer = DreamerReplayBuffer(
        num_envs=B,
        obs_dim=obs_dim,
        capacity_episodes=10,
        device=device,
    )

    for t in range(6):
        buffer.add(
            state=torch.randn(obs_dim, device=device),
            action=torch.randint(0, action_dim, (1,), device=device),
            reward=torch.randn((), device=device),
            is_first=torch.tensor(t == 0, device=device),
            is_last=torch.tensor(t == 5, device=device),
            is_terminal=torch.tensor(False, device=device),
        )

    trainer = TestDreamerTrainer(
        world_model=world_model,
        actor=None,
        critic=None,
        replay_buffer=buffer,
        device=device,
    )

    out = trainer.world_model_training_step(batch_size=2, seq_len=3)

    assert "loss" in out
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
