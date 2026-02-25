import pytest
import torch

from dreamerrl.training.replay_buffer import DreamerReplayBuffer


@pytest.mark.infra
def test_replay_buffer_overflow(device):
    num_envs = 1
    obs_dim = 8
    action_dim = 4
    capacity = 3  # small to force overwrite

    buffer = DreamerReplayBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        capacity_episodes=capacity,
        device=device,
    )

    # Add 5 episodes → only last 3 should remain
    for ep in range(5):
        for t in range(3):
            buffer.add(
                state=torch.full((obs_dim,), float(ep), device=device),
                action=torch.randint(0, action_dim, (1,), device=device),
                reward=torch.randn((), device=device),
                is_first=torch.tensor(t == 0, device=device),
                is_last=torch.tensor(t == 2, device=device),
                is_terminal=torch.tensor(False, device=device),
            )

    assert buffer.num_episodes == 3

    # Episodes should be ep=2,3,4
    ep0 = buffer.get_episode(0)["state"][0, 0].item()
    ep1 = buffer.get_episode(1)["state"][0, 0].item()
    ep2 = buffer.get_episode(2)["state"][0, 0].item()

    assert (ep0, ep1, ep2) == (2.0, 3.0, 4.0)
