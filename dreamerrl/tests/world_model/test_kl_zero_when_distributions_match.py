import torch


def test_kl_zero_when_distributions_match(make_world_model):
    wm = make_world_model()
    B = 4
    h = torch.randn(B, wm.deter_size)

    prior = wm.prior(h)
    kl = wm.kl_divergence(prior, prior)

    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)
