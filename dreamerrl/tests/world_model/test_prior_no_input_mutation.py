import torch


def test_prior_no_input_mutation(make_world_model):
    wm = make_world_model()
    B = 4
    h = torch.randn(B, wm.deter_size)

    h_clone = h.clone()
    _ = wm.prior(h)

    torch.testing.assert_close(h, h_clone)
