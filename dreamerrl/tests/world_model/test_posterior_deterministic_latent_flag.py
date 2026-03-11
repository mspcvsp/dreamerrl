import torch


def test_posterior_deterministic_latent_flag(make_world_model, monkeypatch):
    monkeypatch.setenv("DREAMER_DETERMINISTIC_TEST", "1")
    wm = make_world_model()

    B = 4
    h = torch.randn(B, wm.deter_size)
    embed = torch.randn(B, wm.embed_size)

    out1 = wm.posterior(h, embed)
    out2 = wm.posterior(h, embed)

    torch.testing.assert_close(out1["z"], out2["z"])
