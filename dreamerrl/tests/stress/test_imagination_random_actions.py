import pytest
import torch


@pytest.mark.stress
def test_imagination_random_actions(world_model):
    """
    Purpose: ensure imagination remains stable under random actions.
    """
    torch.manual_seed(0)

    B = 8
    H = 100
    state = world_model.init_state(B)

    for _ in range(H):
        action = torch.nn.functional.one_hot(
            torch.randint(0, world_model.net_cfg.action_dim, (B,)),
            num_classes=world_model.net_cfg.action_dim,
        ).float()

        # manually step RSSM
        h = world_model.rssm(state.h, action)
        prior = world_model.prior(h)
        z = prior["z"]

        state = state.__class__(h=h, z=z)

        assert torch.isfinite(state.h).all()
        assert torch.isfinite(state.z).all()
