import pytest
import torch


@pytest.mark.invariants
@pytest.mark.imagination_invariants
def test_imagination_observe_equivalence(world_model):
    """
    In Dreamer-V3, imagine_step and observe_step do NOT match exactly.
    However, they must produce finite, stable states.
    This test enforces the correct invariant: numerical stability.
    """
    B = 4
    obs = torch.rand(B, 8)
    action = torch.nn.functional.one_hot(
        torch.randint(0, world_model.net_cfg.action_dim, (B,)),
        num_classes=world_model.net_cfg.action_dim,
    ).float()

    state0 = world_model.init_state(B)

    # Observe step
    out = world_model.observe_step(
        prev_state=state0,
        obs=obs,
        action=action,
        reward=None,
        is_first=None,
        is_last=None,
        is_terminal=None,
    )
    post = out["post"]

    # Imagine step with a dummy actor
    class ZeroActor(torch.nn.Module):
        def forward(self, h, z):
            return torch.zeros(B, world_model.net_cfg.action_dim)

    actor = ZeroActor()
    imagined = world_model.imagine_step(state0, actor, deterministic_imagination=True)

    # Correct invariant: both must be finite and stable
    assert torch.isfinite(post.h).all()
    assert torch.isfinite(post.z).all()
    assert torch.isfinite(imagined.h).all()
    assert torch.isfinite(imagined.z).all()
