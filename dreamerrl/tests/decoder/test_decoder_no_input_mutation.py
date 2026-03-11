import torch

from dreamerrl.models.decoder import ObsDecoder


def test_decoder_no_input_mutation():
    B, deter, stoch, hidden, obs_dim = 4, 32, 16, 64, 8
    dec = ObsDecoder(deter, stoch, hidden, obs_dim)

    h = torch.randn(B, deter)
    z = torch.randn(B, stoch)

    h_clone = h.clone()
    z_clone = z.clone()

    _ = dec(h, z)

    torch.testing.assert_close(h, h_clone)
    torch.testing.assert_close(z, z_clone)
