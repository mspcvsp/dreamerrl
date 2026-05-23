import pytest
import torch


@pytest.mark.stress
def test_kl_randomized(world_model):
    """
    Randomized KL fuzz test.
    Ensures KL remains finite under random embeddings.
    """
    torch.manual_seed(0)

    B = 16
    for _ in range(50):
        h = torch.randn(B, world_model.latent.deter_size)
        embed = torch.randn(B, world_model.net_cfg.hidden_size)

        post = world_model.posterior(h, embed)
        prior = world_model.prior(h)

        kl = torch.sum(
            post["probs"] * (post["logits"] - prior["logits"]),
            dim=(-1, -2),
        )

        assert torch.isfinite(kl).all()
        assert (kl >= 0).all()
