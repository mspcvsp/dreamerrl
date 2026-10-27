import pytest
import torch
import torch.nn.functional as F

from dreamerrl.models.world_model_core import RSSMCore


@pytest.mark.invariants
@pytest.mark.rssm
def test_rssm_no_input_mutation(latent, net):
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

    h_clone = h.clone()
    a_clone = a.clone()
    z_clone = z.clone()

    _ = rssm(h, z, a)

    torch.testing.assert_close(h, h_clone)
    torch.testing.assert_close(z, z_clone)
    torch.testing.assert_close(a, a_clone)
