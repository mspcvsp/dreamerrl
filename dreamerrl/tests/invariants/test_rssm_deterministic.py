import pytest
import torch
import torch.nn.functional as F

from dreamerrl.models.world_model_core import RSSMCore


@pytest.mark.invariants
@pytest.mark.rssm
def test_rssm_deterministic(latent, net):
    rssm = RSSMCore(latent=latent, net=net)
    B = 4
    h = torch.randn(B, latent.deter_size)
    a = F.one_hot(torch.randint(0, net.action_dim, (4,)), net.action_dim).float()
    z = torch.randn(
        B,
        latent.stoch_size * latent.num_classes,
        dtype=torch.float32,
        device="cpu",
    )

    out1 = rssm(h, z, a)
    out2 = rssm(h, z, a)

    assert torch.allclose(out1, out2)
