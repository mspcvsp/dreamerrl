import pytest


@pytest.mark.minihack_invariants
def test_minihack_vec_shapes(minihack_vec_env):
    out = minihack_vec_env.reset()
    assert out["state"].shape == (minihack_vec_env.batch_size, minihack_vec_env.obs_dim)
