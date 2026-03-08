import torch


def test_prior_std_positive(make_world_model):
    wm = make_world_model()

    B = 4
    h = torch.randn(B, wm.deter_size)

    prior = wm.prior(h)

    assert (prior["std"] > 0).all()
    assert torch.isfinite(prior["std"]).all()
