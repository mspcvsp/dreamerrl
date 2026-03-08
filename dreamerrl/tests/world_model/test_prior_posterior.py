import torch


def test_prior_posterior(make_world_model):
    wm = make_world_model()

    B = 4
    h = torch.randn(B, wm.deter_size)
    embed = torch.randn(B, wm.embed_size)

    prior = wm.prior(h)
    post = wm.posterior(h, embed)

    assert prior.mean.shape == (B, wm.stoch_size)
    assert prior.std.shape == (B, wm.stoch_size)
    assert post.mean.shape == (B, wm.stoch_size)
    assert post.std.shape == (B, wm.stoch_size)

    assert torch.isfinite(prior.mean).all()
    assert torch.isfinite(prior.std).all()
    assert torch.isfinite(post.mean).all()
    assert torch.isfinite(post.std).all()
