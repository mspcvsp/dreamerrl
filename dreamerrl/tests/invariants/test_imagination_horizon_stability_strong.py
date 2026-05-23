import torch


def test_imagination_horizon_stability_strong(world_model, dummy_actor):
    """
    Long-horizon imagination should remain numerically stable.
    Ensures no exploding activations in RSSMCore or Prior.
    """
    B = 4
    H = 30
    state = world_model.init_state(B)

    for _ in range(H):
        state = world_model.imagine_step(state, dummy_actor, stochastic=False)
        assert torch.isfinite(state.h).all()
        assert torch.isfinite(state.z).all()
