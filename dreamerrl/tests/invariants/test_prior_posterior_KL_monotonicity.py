import torch


def test_prior_posterior_KL_monotonicity(world_model):
    """
    KL divergence must be non-negative and finite.
    """
    B = 6
    h = torch.randn(B, world_model.latent.deter_size)
    embed = torch.randn(B, world_model.net_cfg.hidden_size)

    post = world_model.posterior(h, embed)
    prior = world_model.prior(h)

    kl = torch.sum(post["probs"] * (post["logits"] - prior["logits"]), dim=(-1, -2))

    assert torch.isfinite(kl).all()
    assert (kl >= 0).all()
