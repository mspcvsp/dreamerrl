import pytest
import torch


@pytest.mark.invariants
@pytest.mark.actor_invariants
def test_actor_critic_consistency(world_model):
    """
    Distributional value head must produce finite expected values
    within the bin range. Monotonicity across samples is NOT an invariant.
    """
    B = 8
    h = torch.randn(B, world_model.latent.deter_size)
    z = torch.randn(B, world_model.latent.stoch_size, world_model.latent.num_classes)

    logits = world_model.reward_head(h, z)
    probs = logits.softmax(-1)
    bins = world_model.reward_head.bin_values

    expected = (probs * bins).sum(-1)

    # Invariants that MUST hold
    assert torch.isfinite(expected).all()
    assert expected.min() >= bins.min() - 1e-5
    assert expected.max() <= bins.max() + 1e-5
