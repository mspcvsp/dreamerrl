import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.training.replay_buffer import DreamerReplayBuffer
from dreamerrl.training.test_trainer import _TestDreamerTrainer


@pytest.mark.trainer
def test_imagination_rollout(world_model, device):
    B = 4
    H = 5

    deter = world_model.deter_size
    stoch = world_model.stoch_size
    action_dim = 4

    actor = Actor(deter, stoch, hidden_size=128, action_dim=action_dim).to(device)
    critic = ValueHead(deter, stoch, hidden_size=128).to(device)

    dummy_buffer = DreamerReplayBuffer(
        num_envs=1,
        obs_dim=world_model.flat_obs_dim,
        capacity_episodes=1,
        device=device,
    )

    trainer = _TestDreamerTrainer(
        world_model=world_model,
        actor=actor,
        critic=critic,
        replay_buffer=dummy_buffer,
        device=device,
    )

    state = world_model.init_state(B)
    rollout = trainer.imagination_rollout(state, horizon=H)

    assert rollout["h"].shape == (H, B, deter)
    assert rollout["z"].shape == (H, B, stoch)
    assert rollout["value"].shape == (H, B, 1)
    assert rollout["action"].shape == (H, B)
    assert torch.isfinite(rollout["value"]).all()
