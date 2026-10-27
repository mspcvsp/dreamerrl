import pytest
import torch
import torch.nn.functional as F


@pytest.mark.rssm
@pytest.mark.rssm
def test_rssm_latent_affects_transition(
    rssm,
    latent,
    net,
):
    h = torch.randn(1, latent.deter_size)

    a = F.one_hot(
        torch.tensor([1]),
        num_classes=net.action_dim,
    ).float()

    z1 = torch.zeros(
        1,
        latent.num_classes,
        latent.stoch_size,
    )

    z2 = torch.zeros_like(z1)
    z2[:, :, 0] = 1.0

    out1 = rssm(h, z1, a)
    out2 = rssm(h, z2, a)

    assert not torch.allclose(out1, out2)
