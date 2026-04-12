import torch


def test_rssm_seed_determinism(world_model):
    B = 4
    h = torch.randn(B, world_model.deter_size)
    embed = torch.randn(B, world_model.embed_size)

    torch.manual_seed(0)
    out1 = world_model.posterior(h, embed)

    torch.manual_seed(0)
    out2 = world_model.posterior(h, embed)

    assert torch.allclose(out1["z"], out2["z"])
    assert torch.allclose(out1["mean"], out2["mean"])
    assert torch.allclose(out1["std"], out2["std"])
