import pytest
import torch
import torch.nn.functional as F

from dreamerrl.models.world_model_core import RSSMCore


@pytest.mark.invariants
@pytest.mark.rssm
def test_rssm_cpu_gpu_equivalence(latent, net):
    if not torch.cuda.is_available():
        return

    rssm_cpu = RSSMCore(latent=latent, net=net).cpu()
    rssm_gpu = RSSMCore(latent=latent, net=net).cuda()
    rssm_gpu.load_state_dict(rssm_cpu.state_dict())

    h = torch.randn(4, latent.deter_size)
    a = F.one_hot(torch.randint(0, net.action_dim, (4,)), net.action_dim).float()

    out_cpu = rssm_cpu(h, a)
    out_gpu = rssm_gpu(h.cuda(), a.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu)
