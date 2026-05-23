import pytest
import torch


@pytest.mark.invariants
def test_reward_head_distributional_properties(world_model):
    B = 6
    h = torch.randn(B, world_model.latent.deter_size)
    z = torch.randn(B, world_model.latent.stoch_size, world_model.latent.num_classes)

    logits = world_model.reward_head(h, z)
    probs = logits.softmax(-1)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-6)
