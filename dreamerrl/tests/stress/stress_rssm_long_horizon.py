import pytest
import torch


@pytest.mark.stress
def test_rssm_long_horizon_stability(world_model, dummy_actor):
    """
    Long-horizon imagination fuzz test.
    Ensures no NaNs or infs appear over 200 steps.
    """
    torch.manual_seed(0)

    B = 8
    H = 200
    state = world_model.init_state(B)

    for _ in range(H):
        state = world_model.imagine_step(state, dummy_actor, stochastic=False)
        assert torch.isfinite(state.h).all()
        assert torch.isfinite(state.z).all()
