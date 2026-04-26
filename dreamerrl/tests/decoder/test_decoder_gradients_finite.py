import torch

from dreamerrl.models.decoder import ObsDecoder


def test_decoder_gradients_finite():
    B, deter, stoch, nclasses, hidden, obs_dim = 4, 32, 16, 32, 64, 8
    dec = ObsDecoder(deter, stoch, nclasses, hidden, obs_dim)

    h = torch.randn(B, deter, requires_grad=True)
    z = torch.randn(B, stoch, requires_grad=True)

    out = dec(h, z).sum()
    out.backward()

    for p in dec.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
