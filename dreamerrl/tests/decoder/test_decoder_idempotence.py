import torch

from dreamerrl.models.decoder import ObsDecoder


def test_decoder_idempotence():
    """
    Same inputs → same outputs within the same call context (ensures no mutation, no randomness, no hidden state)
    """
    B, deter_size, stoch_size, hidden_size, obs_dim = 4, 32, 16, 64, 8

    decoder = ObsDecoder(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
        obs_shape=obs_dim,
    )

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    out1 = decoder(h, z)
    out2 = decoder(h, z)

    # Must be exactly identical
    torch.testing.assert_close(out1, out2)
