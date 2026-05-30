import pytest
import torch


@pytest.mark.stress
def test_reward_head_fuzz(world_model):
    """
    Purpose: ensure reward head remains stable under extreme latent noise.
    """
    torch.manual_seed(0)

    B = 32
    for _ in range(50):
        h = torch.randn(B, world_model.latent.deter_size) * 10
        z = torch.randn(B, world_model.latent.num_classes, world_model.latent.stoch_size) * 10

        logits = world_model.reward_head(h, z)
        probs = logits.softmax(-1)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
