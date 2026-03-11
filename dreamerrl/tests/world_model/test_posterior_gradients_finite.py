import torch


def test_posterior_gradients_finite(make_world_model):
    wm = make_world_model()
    B = 4
    h = torch.randn(B, wm.deter_size, requires_grad=True)
    embed = torch.randn(B, wm.embed_size)

    out = wm.posterior(h, embed)
    loss = (out["mean"] ** 2 + out["std"] ** 2).mean()
    loss.backward()

    for p in wm.posterior.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
