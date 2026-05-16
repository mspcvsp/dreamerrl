import pytest
import torch
import torch.nn.functional as F

from dreamerrl.models.world_model_core import RSSMCore


@pytest.mark.rssm
def test_rssm_batch_size_invariance(latent, net):
    rssm = RSSMCore(latent=latent, net=net)

    h1 = torch.randn(1, latent.deter_size)
    a1 = F.one_hot(torch.randint(0, net.action_dim, (1,)), net.action_dim).float()

    h4 = h1.repeat(4, 1)
    a4 = a1.repeat(4, 1)

    out1 = rssm(h1, a1)
    out4 = rssm(h4, a4)

    torch.testing.assert_close(out4, out1.repeat(4, 1))
