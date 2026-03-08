import torch


def test_kl_finite(make_world_model):
    wm = make_world_model()
    B = 4
    h = torch.randn(B, wm.deter_size)
    embed = torch.randn(B, wm.embed_size)

    prior = wm.prior(h)
    post = wm.posterior(h, embed)

    kl = wm.kl_divergence(prior, post)
    assert torch.isfinite(kl).all()
