import torch


def test_gae_time_major(fake_buffer_loader):
    """
    Ensures GAE is computed in time-major format (T, B)
    and aligned with rewards, values, and masks.
    """
    buf = fake_buffer_loader()

    # Explicitly compute GAE (required)
    last_value = torch.zeros(buf.cfg.num_envs, device=buf.device)
    buf.compute_returns_and_advantages(last_value)

    T, B = buf.rewards.shape

    # 1. GAE outputs must be time-major
    assert buf.advantages.shape == (T, B)
    assert buf.returns.shape == (T, B)

    # 2. rewards, values, masks must also be time-major
    assert buf.rewards.shape == (T, B)
    assert buf.values.shape == (T, B)
    assert buf.masks.shape == (T, B)

    # 3. returns must be the normalized (values + advantages)
    raw_returns = buf.values + buf.advantages
    expected = (raw_returns - raw_returns.mean()) / (raw_returns.std(unbiased=False) + 1e-8)
    assert torch.allclose(buf.returns, expected)

    # 4. sanity: no NaNs or Infs
    assert torch.isfinite(buf.advantages).all()
    assert torch.isfinite(buf.returns).all()
