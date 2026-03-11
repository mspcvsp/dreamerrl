import torch

from dreamerrl.models.world_model_core import RSSMCore


def test_rssm_output_shape():
    B, deter, stoch = 5, 32, 16
    rssm = RSSMCore(deter, stoch, hidden_size=64)

    h = torch.randn(B, deter)
    z = torch.randn(B, stoch)

    out = rssm(h, z)
    assert out.shape == (B, deter)
