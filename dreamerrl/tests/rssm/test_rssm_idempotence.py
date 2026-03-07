import torch

from dreamerrl.models.world_model_core import RSSMCore


def test_rssm_idempotence():
    B, deter_size, stoch_size, hidden_size = 4, 32, 16, 64

    rssm = RSSMCore(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    )

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    out1 = rssm(h, z)
    out2 = rssm(h, z)

    torch.testing.assert_close(out1, out2)
