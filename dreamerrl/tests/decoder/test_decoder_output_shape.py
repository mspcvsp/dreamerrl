import torch

from dreamerrl.models.decoder import ObsDecoder


def test_decoder_output_shape():
    B, deter, stoch, nclasses, hidden, obs_dim = 5, 32, 16, 32, 64, 8
    dec = ObsDecoder(deter, stoch, nclasses, hidden, obs_dim)

    h = torch.randn(B, deter)
    z = torch.randn(B, stoch * nclasses)

    out = dec(h, z)
    assert out.shape == (B, obs_dim)
