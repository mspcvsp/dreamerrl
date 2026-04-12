import torch


def test_world_model_kl_structure(world_model):
    B = 4
    h = torch.randn(B, world_model.deter_size)
    embed = torch.randn(B, world_model.embed_size)

    post = world_model.posterior(h, embed)
    prior = world_model.prior(h)

    dyn, rep = world_model.structured_kl(post, prior)

    assert torch.isfinite(dyn).all()
    assert torch.isfinite(rep).all()
    assert not torch.allclose(dyn, rep)
