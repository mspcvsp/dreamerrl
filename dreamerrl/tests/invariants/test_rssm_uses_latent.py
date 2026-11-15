import pytest
import torch
import torch.nn.functional as F

from dreamerrl.models.world_model_core import RSSMCore


@pytest.mark.rssm
@pytest.mark.invariants
def test_rssm_uses_latent(latent, net):
    rssm = RSSMCore(latent=latent, net=net)

    h = torch.randn(1, latent.deter_size)

    a = F.one_hot(
        torch.randint(0, net.action_dim, (1,)),
        net.action_dim,
    ).float()

    z1 = torch.zeros(
        1,
        latent.num_classes,
        latent.stoch_size,
    )

    z2 = torch.ones(
        1,
        latent.num_classes,
        latent.stoch_size,
    )

    out1 = rssm(h, z1, a)
    out2 = rssm(h, z2, a)

    assert not torch.allclose(out1, out2)
