import pytest
import torch


@pytest.mark.invariants
def test_continue_head_distributional_properties(world_model):
    B = 6
    h = torch.randn(B, world_model.latent.deter_size)
    z = torch.randn(B, world_model.latent.num_classes, world_model.latent.stoch_size)

    logits = world_model.continue_head(h, z)
    probs = world_model.continue_head.readout(logits)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(probs).all()
    assert logits.shape == (B, 1)
    assert ((probs >= 0) & (probs <= 1)).all()
