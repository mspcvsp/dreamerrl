import torch

from dreamerrl.models.decoder import ObsDecoder


def test_decoder_deterministic():
    B, deter_size, stoch_size, nclasses, hidden_size, obs_dim = 4, 32, 16, 64, 3464, 8

    # Fix seed so two decoders initialize identically
    torch.manual_seed(0)
    decoder1 = ObsDecoder(
        deter_size=deter_size,
        stoch_size=stoch_size,
        num_classes=nclasses,
        hidden_size=hidden_size,
        obs_shape=obs_dim,
    )

    torch.manual_seed(0)
    decoder2 = ObsDecoder(
        deter_size=deter_size,
        stoch_size=stoch_size,
        num_classes=nclasses,
        hidden_size=hidden_size,
        obs_shape=obs_dim,
    )

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size * nclasses)

    out1 = decoder1(h, z)
    out2 = decoder2(h, z)

    torch.testing.assert_close(out1, out2)
