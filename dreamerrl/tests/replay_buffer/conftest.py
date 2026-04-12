import pytest

from dreamerrl.replay_buffer.replay_buffer import DreamerReplayBuffer


@pytest.fixture
def replay_buffer(device):
    return DreamerReplayBuffer(
        num_envs=4,
        obs_dim=8,
        capacity_episodes=10,
        device=device,
    )
