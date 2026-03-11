import torch


def test_prior_deterministic_initialization(make_world_model):
    wm = make_world_model()
    B = 4
    h = torch.randn(B, wm.deter_size)

    torch.manual_seed(0)
    out1 = wm.prior(h)

    torch.manual_seed(0)
    out2 = wm.prior(h)

    torch.testing.assert_close(out1["mean"], out2["mean"])
    torch.testing.assert_close(out1["std"], out2["std"])
