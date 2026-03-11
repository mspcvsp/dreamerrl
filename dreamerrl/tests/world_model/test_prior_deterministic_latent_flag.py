import torch


def test_prior_deterministic_latent_flag(make_world_model, monkeypatch):
    monkeypatch.setenv("DREAMER_DETERMINISTIC_TEST", "1")
    wm = make_world_model()

    B = 4
    h = torch.randn(B, wm.deter_size)

    out1 = wm.prior(h)
    out2 = wm.prior(h)

    torch.testing.assert_close(out1["z"], out2["z"])
