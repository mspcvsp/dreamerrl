import torch


def test_posterior_no_input_mutation(make_world_model):
    wm = make_world_model()
    B = 4
    h = torch.randn(B, wm.deter_size)
    embed = torch.randn(B, wm.embed_size)

    h_clone = h.clone()
    embed_clone = embed.clone()

    _ = wm.posterior(h, embed)

    torch.testing.assert_close(h, h_clone)
    torch.testing.assert_close(embed, embed_clone)
