import torch


def test_gae_time_major(fake_buffer_loader):
    """
    Ensures GAE is computed in time-major format (T, B)
    and aligned with rewards, values, and masks.
    """
    buf = fake_buffer_loader()

    # Explicitly compute GAE (required!)
    last_value = torch.zeros(buf.cfg.num_envs)
    buf.compute_returns_and_advantages(last_value)

    T, B = buf.rewards.shape

    # 1. GAE outputs must be time-major
    assert buf.advantages.shape == (T, B)
    assert buf.returns.shape == (T, B)

    # 2. rewards, values, masks must also be time-major
    assert buf.rewards.shape == (T, B)
    assert buf.values.shape == (T, B)
    assert buf.masks.shape == (T, B)

    # 3. returns = values + advantages
    assert torch.allclose(buf.returns, buf.values + buf.advantages)

    # 4. Terminated timesteps must not bootstrap
    terminated = buf.terminated

    # If terminated[t] = True, then bootstrap=False, so:
    # advantages[t] = delta[t] (no future accumulation)
    # We cannot assert zero, but we CAN assert no accumulation:
    if terminated.any():
        t = terminated.nonzero(as_tuple=False)[0, 0]
        if t < T - 1:
            assert not torch.allclose(buf.advantages[t], buf.advantages[t + 1])

    # 5. Sanity: advantages should not be NaN or Inf
    assert torch.isfinite(buf.advantages).all()
    assert torch.isfinite(buf.returns).all()
