import pytest
import torch

from dreamerrl.training.replay_buffer import DreamerReplayBuffer


@pytest.mark.infra
def test_sampling_contiguity(device):
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

    # Build one episode of length 12
    for t in range(12):
        buffer.add(
            state=torch.full((obs_dim,), float(t), device=device),
            action=torch.randint(0, action_dim, (1,), device=device),
            reward=torch.randn((), device=device),
            is_first=torch.tensor(t == 0, device=device),
            is_last=torch.tensor(t == 11, device=device),
            is_terminal=torch.tensor(False, device=device),
        )

    batch = buffer.sample(batch_size=3, seq_len=5)

    # Check contiguity: differences must be exactly 1
    for seq in batch["state"]:
        diffs = seq[:, 0] - seq[0, 0]
        assert torch.all(diffs == torch.arange(5, device=device))
