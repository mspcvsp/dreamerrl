import pytest
import torch

from dreamerrl.training.replay_buffer import DreamerReplayBuffer


@pytest.mark.infra
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replay_buffer_cpu_gpu_equivalence():
    num_envs = 1
    obs_dim = 8
    action_dim = 4
    capacity = 10

    cpu_buffer = DreamerReplayBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        capacity_episodes=capacity,
        device=torch.device("cpu"),
    )

    gpu_buffer = DreamerReplayBuffer(
        num_envs=num_envs,
        obs_dim=obs_dim,
        capacity_episodes=capacity,
        device=torch.device("cuda"),
    )

    # Build identical episodes
    for t in range(6):
        state = torch.randn(obs_dim)
        action = torch.randint(0, action_dim, (1,))
        reward = torch.randn(())

        kwargs = dict(
            state=state,
            action=action,
            reward=reward,
            is_first=torch.tensor(t == 0),
            is_last=torch.tensor(t == 5),
            is_terminal=torch.tensor(False),
        )

        cpu_buffer.add(**kwargs)
        gpu_buffer.add(**kwargs)

    cpu_batch = cpu_buffer.sample(batch_size=4, seq_len=3, device=torch.device("cpu"))
    gpu_batch = gpu_buffer.sample(batch_size=4, seq_len=3, device=torch.device("cuda"))

    # Move GPU batch to CPU for comparison
    gpu_batch_cpu = {k: v.cpu() for k, v in gpu_batch.items()}

    for key in cpu_batch:
        assert torch.allclose(cpu_batch[key], gpu_batch_cpu[key], atol=1e-5, rtol=1e-5)
