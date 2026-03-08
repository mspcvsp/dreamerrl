import torch


def test_posterior_std_positive(make_world_model):
    wm = make_world_model()

    B = 4
    h = torch.randn(B, wm.deter_size)
    embed = torch.randn(B, wm.embed_size)

    post = wm.posterior(h, embed)

    assert (post["std"] > 0).all()
    assert torch.isfinite(post["std"]).all()
