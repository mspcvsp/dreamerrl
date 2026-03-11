import torch

from dreamerrl.models.world_model_core import RSSMCore


def test_rssm_no_input_mutation():
    B, deter, stoch = 4, 32, 16
    rssm = RSSMCore(deter, stoch, hidden_size=64)

    h = torch.randn(B, deter)
    z = torch.randn(B, stoch)

    h_clone = h.clone()
    z_clone = z.clone()

    _ = rssm(h, z)

    torch.testing.assert_close(h, h_clone)
    torch.testing.assert_close(z, z_clone)
