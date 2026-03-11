import torch

from dreamerrl.models.world_model_core import RSSMCore


def test_rssm_gradients_finite():
    B, deter, stoch = 4, 32, 16
    rssm = RSSMCore(deter, stoch, hidden_size=64)

    h = torch.randn(B, deter, requires_grad=True)
    z = torch.randn(B, stoch, requires_grad=True)

    out = rssm(h, z).sum()
    out.backward()

    for p in rssm.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
