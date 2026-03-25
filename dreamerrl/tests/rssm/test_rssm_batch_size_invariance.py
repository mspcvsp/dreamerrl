import torch

from dreamerrl.models.world_model_core import RSSMCore


def test_rssm_batch_size_invariance():
    deter, stoch = 32, 16
    rssm = RSSMCore(deter, stoch, hidden_size=64)

    h1 = torch.randn(1, deter)
    z1 = torch.randn(1, stoch)

    h4 = h1.repeat(4, 1)
    z4 = z1.repeat(4, 1)

    out1 = rssm(h1, z1)
    out4 = rssm(h4, z4)

    torch.testing.assert_close(out4, out1.repeat(4, 1))
