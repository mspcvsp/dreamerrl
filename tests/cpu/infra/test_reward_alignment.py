import torch


def test_reward_alignment(fake_buffer_loader):
    buf = fake_buffer_loader()

    rewards = buf.rewards
    obs = buf.obs

    # 1. rewards must be (T, B)
    assert rewards.shape[:2] == obs.shape[:2]

    # 2. rewards must be float32
    assert rewards.dtype == torch.float32

    # 3. rewards must be consistent with FakeRolloutBuilder
    # FakeRolloutBuilder uses: rewards[t] = t
    T = rewards.shape[0]
    expected = torch.arange(T, dtype=torch.float32).unsqueeze(1).expand_as(rewards)
    assert torch.allclose(rewards, expected)
