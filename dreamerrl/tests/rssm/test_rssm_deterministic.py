import torch

from dreamerrl.models.world_model_core import RSSMCore


def test_rssm_deterministic():
    B, deter_size, stoch_size, hidden_size = 4, 32, 16, 64

    torch.manual_seed(0)
    rssm1 = RSSMCore(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    )

    torch.manual_seed(0)
    rssm2 = RSSMCore(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    )

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    out1 = rssm1(h, z)
    out2 = rssm2(h, z)

    torch.testing.assert_close(out1, out2)
