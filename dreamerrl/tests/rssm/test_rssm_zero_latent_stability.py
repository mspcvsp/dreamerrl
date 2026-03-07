import torch

from dreamerrl.models.world_model_core import RSSMCore


def test_rssm_zero_latent_stability():
    B, deter_size, stoch_size, hidden_size = 4, 32, 16, 64

    rssm = RSSMCore(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    )

    h = torch.randn(B, deter_size)
    z = torch.zeros(B, stoch_size)

    out = rssm(h, z)

    # Must be finite and stable
    assert torch.isfinite(out).all()
    # Optional: ensure output isn't trivially identical to input
    assert not torch.allclose(out, h)
