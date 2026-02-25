import pytest
import torch

from dreamerrl.training.replay_buffer import DreamerReplayBuffer


@pytest.mark.infra
def test_episode_construction(device):
    B = 1
    obs_dim = 8
    action_dim = 4
    capacity = 10

    buffer = DreamerReplayBuffer(
        num_envs=B,
        obs_dim=obs_dim,
        capacity_episodes=capacity,
        device=device,
    )

    # Episode 1: length 3
    for t in range(3):
        buffer.add(
            state=torch.randn(obs_dim, device=device),
            action=torch.randint(0, action_dim, (1,), device=device),
            reward=torch.randn((), device=device),
            is_first=torch.tensor(t == 0, device=device),
            is_last=torch.tensor(t == 2, device=device),
            is_terminal=torch.tensor(False, device=device),
        )

    # Episode 2: length 2
    for t in range(2):
        buffer.add(
            state=torch.randn(obs_dim, device=device),
            action=torch.randint(0, action_dim, (1,), device=device),
            reward=torch.randn((), device=device),
            is_first=torch.tensor(t == 0, device=device),
            is_last=torch.tensor(t == 1, device=device),
            is_terminal=torch.tensor(False, device=device),
        )

    assert buffer.num_episodes == 2
    assert buffer.num_transitions == 5

    ep1 = buffer.get_episode(0)
    ep2 = buffer.get_episode(1)

    assert ep1["state"].shape[0] == 3
    assert ep2["state"].shape[0] == 2

    assert ep1["is_first"][0]
    assert ep1["is_last"][-1]

    assert ep2["is_first"][0]
    assert ep2["is_last"][-1]

    # Contiguity
    assert (ep1["t"][1:] > ep1["t"][:-1]).all()
    assert (ep2["t"][1:] > ep2["t"][:-1]).all()
