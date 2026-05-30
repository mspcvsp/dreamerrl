import pytest
import torch


@pytest.mark.invariants
@pytest.mark.imagination_invariants
def test_imagination_horizon_stability_strong(world_model, dummy_actor):
    """
    Long-horizon imagination should remain numerically stable.
    Ensures no exploding activations in RSSMCore or Prior.
    """
    B = 4
    H = 30
    state = world_model.init_state(B)

    for _ in range(H):
        state = world_model.imagine_step(state, dummy_actor, deterministic_imagination=True)
        assert torch.isfinite(state.h).all()
        assert torch.isfinite(state.z).all()
