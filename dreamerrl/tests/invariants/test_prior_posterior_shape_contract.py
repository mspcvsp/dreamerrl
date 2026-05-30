import pytest
import torch


@pytest.mark.invariants
def test_prior_posterior_shape_contract(world_model):
    B = 4
    h = torch.randn(B, world_model.latent.deter_size)
    embed = torch.randn(B, world_model.net_cfg.hidden_size)

    post = world_model.posterior(h, embed)
    prior = world_model.prior(h)

    K = world_model.latent.num_classes
    C = world_model.latent.stoch_size

    assert post["logits"].shape == (B, K, C)
    assert post["probs"].shape == (B, K, C)
    assert prior["logits"].shape == (B, K, C)
    assert prior["probs"].shape == (B, K, C)
