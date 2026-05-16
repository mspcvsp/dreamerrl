import pytest
import torch
import torch.nn.functional as F

from dreamerrl.models.world_model_core import RSSMCore


@pytest.mark.invariants
@pytest.mark.rssm
def test_rssm_deterministic(latent, net):
    rssm = RSSMCore(latent=latent, net=net)
    h = torch.randn(4, latent.deter_size)
    a = F.one_hot(torch.randint(0, net.action_dim, (4,)), net.action_dim).float()

    out1 = rssm(h, a)
    out2 = rssm(h, a)

    assert torch.allclose(out1, out2)
