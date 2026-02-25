import pytest
import torch

from dreamerrl.training.replay_buffer import DreamerReplayBuffer


@pytest.mark.infra
def test_sampling_shapes(device):
    num_envs = 1
    obs_dim = 8
    action_dim = 4
    capacity = 10

    buffer = DreamerReplayBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        capacity_episodes=capacity,
        device=device,
    )

    # Build one episode of length 10
    for t in range(10):
        buffer.add(
            state=torch.randn(obs_dim, device=device),
            action=torch.randint(0, action_dim, (1,), device=device),
            reward=torch.randn((), device=device),
            is_first=torch.tensor(t == 0, device=device),
            is_last=torch.tensor(t == 9, device=device),
            is_terminal=torch.tensor(False, device=device),
        )

    batch = buffer.sample(batch_size=4, seq_len=5)

    assert batch["state"].shape == (4, 5, obs_dim)
    assert batch["action"].shape == (4, 5)
    assert batch["reward"].shape == (4, 5)
    assert batch["is_first"].shape == (4, 5)
    assert batch["is_last"].shape == (4, 5)
    assert batch["is_terminal"].shape == (4, 5)
