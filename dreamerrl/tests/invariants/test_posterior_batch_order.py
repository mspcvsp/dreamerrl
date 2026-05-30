# dreamerrl/tests/invariants/test_posterior_batch_order.py
import pytest
import torch


@pytest.mark.invariants
def test_posterior_batch_order_invariance(world_model, dummy_obs):
    """
    Posterior must be invariant to batch ordering.
    Shuffle → encode → unshuffle → compare.
    """
    B = dummy_obs.shape[0]
    perm = torch.randperm(B)

    post1 = world_model.encoder(dummy_obs)

    post2 = world_model.encoder(dummy_obs[perm])
    post2_unshuffled = post2[torch.argsort(perm)]

    assert torch.allclose(post1, post2_unshuffled, atol=0, rtol=0)
