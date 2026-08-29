import pytest
import torch


@pytest.mark.minihack_invariants
def test_minihack_vec_episode_end(minihack_vec_env):
    for _ in range(minihack_vec_env.batch_size):
        out = minihack_vec_env.step(torch.zeros(minihack_vec_env.batch_size, dtype=torch.long))
    assert out["is_last"].any()
