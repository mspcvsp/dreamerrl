import pytest
import torch


@pytest.mark.stress
def test_posterior_fuzz(world_model):
    """
    Fuzz posterior with extreme embeddings.
    """
    torch.manual_seed(0)

    B = 16
    for _ in range(50):
        h = torch.randn(B, world_model.latent.deter_size) * 5
        embed = torch.randn(B, world_model.net_cfg.hidden_size) * 5

        post = world_model.posterior(h, embed)

        assert torch.isfinite(post["logits"]).all()
        assert torch.isfinite(post["probs"]).all()
        assert torch.allclose(post["probs"].sum(-1), torch.ones(B, world_model.latent.num_classes), atol=1e-5)
