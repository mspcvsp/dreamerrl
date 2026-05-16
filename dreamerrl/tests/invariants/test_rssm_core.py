import torch

from dreamerrl.models.world_model_core import RSSMCore
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def _rssm():
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5, value_bins=41)
    return RSSMCore(latent=latent, net=net)


def test_rssm_deterministic_transition():
    torch.manual_seed(0)
    rssm = _rssm()

    h = torch.randn(4, 200)
    a = torch.nn.functional.one_hot(torch.randint(0, 5, (4,)), num_classes=5).float()

    out1 = rssm(h, a)
    out2 = rssm(h, a)

    assert torch.allclose(out1, out2)


def test_rssm_cpu_gpu_equivalence():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    rssm_cpu = _rssm().cpu()
    rssm_gpu = _rssm().cuda()
    rssm_gpu.load_state_dict(rssm_cpu.state_dict())

    h = torch.randn(4, 200)
    a = torch.nn.functional.one_hot(torch.randint(0, 5, (4,)), num_classes=5).float()

    out_cpu = rssm_cpu(h, a)
    out_gpu = rssm_gpu(h.cuda(), a.cuda()).cpu()

    assert torch.allclose(out_cpu, out_gpu, atol=1e-5)
