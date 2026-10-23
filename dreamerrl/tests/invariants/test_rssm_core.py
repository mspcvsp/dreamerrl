import pytest
import torch
import torch.nn.functional as F

from dreamerrl.models.world_model_core import RSSMCore
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def _rssm():
    latent = LatentConfig(
        deter_size=200,
        stoch_size=30,
        num_classes=32,
    )

    net = NetworkConfig(
        hidden_size=256,
        action_dim=5,
        value_bins=41,
    )

    return RSSMCore(latent=latent, net=net)


@pytest.mark.invariants
@pytest.mark.rssm
def test_rssm_deterministic_transition(sample_z):
    torch.manual_seed(0)

    rssm = _rssm()

    h = torch.randn(4, 200)

    z = sample_z(4)

    a = F.one_hot(
        torch.randint(0, 5, (4,)),
        num_classes=5,
    ).float()

    out1 = rssm(h, z, a)
    out2 = rssm(h, z, a)

    assert torch.allclose(out1, out2)


@pytest.mark.invariants
def test_rssm_cpu_gpu_equivalence(sample_z):
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)

    rssm_cpu = _rssm().cpu()
    rssm_gpu = _rssm().cuda()

    rssm_gpu.load_state_dict(rssm_cpu.state_dict())

    h = torch.randn(4, 200)

    z = sample_z(4)

    a = F.one_hot(
        torch.randint(0, 5, (4,)),
        num_classes=5,
    ).float()

    out_cpu = rssm_cpu(h, z, a)

    out_gpu = rssm_gpu(
        h.cuda(),
        z.cuda(),
        a.cuda(),
    ).cpu()

    assert torch.allclose(
        out_cpu,
        out_gpu,
        atol=1e-5,
    )
