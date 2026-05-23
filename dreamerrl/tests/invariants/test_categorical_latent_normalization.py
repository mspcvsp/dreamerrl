import torch


def test_categorical_latent_normalization(world_model):
    B = 4
    h = torch.randn(B, world_model.latent.deter_size)
    embed = torch.randn(B, world_model.net_cfg.hidden_size)

    post = world_model.posterior(h, embed)
    prior = world_model.prior(h)

    assert torch.allclose(post["probs"].sum(-1), torch.ones(B, world_model.latent.stoch_size))
    assert torch.allclose(prior["probs"].sum(-1), torch.ones(B, world_model.latent.stoch_size))
