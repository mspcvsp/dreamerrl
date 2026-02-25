import pytest
import torch

from dreamerrl.training.replay_buffer import DreamerReplayBuffer


@pytest.mark.infra
def test_sampling_sequences(device):
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

    # Episode 1: length 4
    for t in range(4):
        buffer.add(
            state=torch.full((obs_dim,), float(t), device=device),
            action=torch.randint(0, action_dim, (1,), device=device),
            reward=torch.randn((), device=device),
            is_first=torch.tensor(t == 0, device=device),
            is_last=torch.tensor(t == 3, device=device),
            is_terminal=torch.tensor(False, device=device),
        )

    # Episode 2: length 6
    for t in range(6):
        buffer.add(
            state=torch.full((obs_dim,), float(100 + t), device=device),
            action=torch.randint(0, action_dim, (1,), device=device),
            reward=torch.randn((), device=device),
            is_first=torch.tensor(t == 0, device=device),
            is_last=torch.tensor(t == 5, device=device),
            is_terminal=torch.tensor(False, device=device),
        )

    # Sample sequences of length 3
    batch = buffer.sample(batch_size=10, seq_len=3)

    for seq in batch["state"]:
        # Either from episode 1 (0..3) or episode 2 (100..105)
        first = seq[0, 0].item()
        last = seq[-1, 0].item()

        # Episode 1
        if first < 50:
            assert last - first == 2
            assert 0 <= first <= 3
        # Episode 2
        else:
            assert last - first == 2
            assert 100 <= first <= 105
