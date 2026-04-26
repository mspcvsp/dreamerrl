import torch

from dreamerrl.models.decoder import ObsDecoder


def test_decoder_batch_size_invariance():
    deter, stoch, nclasses, hidden, obs_dim = 32, 16, 64, 8, 3
    dec = ObsDecoder(deter, stoch, nclasses, hidden, obs_dim)

    h1 = torch.randn(1, deter)
    z1 = torch.randn(1, stoch * nclasses)

    h4 = h1.repeat(4, 1)
    z4 = z1.repeat(4, 1)

    out1 = dec(h1, z1)
    out4 = dec(h4, z4)

    torch.testing.assert_close(out4, out1.repeat(4, 1))
