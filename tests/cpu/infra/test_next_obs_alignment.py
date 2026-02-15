import torch


def test_next_obs_alignment(fake_buffer_loader):
    buf = fake_buffer_loader()

    batch = next(buf.get_recurrent_minibatches())

    obs = batch.obs
    next_obs = batch.next_obs

    # 1. next_obs[:-1] == obs[1:]
    assert torch.allclose(next_obs[:-1], obs[1:])

    # 2. final timestep padded
    assert torch.all(next_obs[-1] == 0)
