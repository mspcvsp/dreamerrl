import pytest
import torch


@pytest.mark.invariants
def test_rssm_zero_latent_stability(world_model, dummy_actor):
    """
    Zero latent + zero action should not produce NaNs or infs.
    Ensures RSSMCore is numerically stable at the origin.
    """
    B = 4
    state = world_model.init_state(B)

    # Deterministic rollout with dummy actor
    next_state = world_model.imagine_step(state, dummy_actor, stochastic=False)

    assert torch.isfinite(next_state.h).all()
    assert torch.isfinite(next_state.z).all()
