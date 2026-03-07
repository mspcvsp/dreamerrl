import torch

from dreamerrl.models.world_model_core import RSSMCore


def test_rssm_cpu_gpu_equivalence():
    if not torch.cuda.is_available():
        return

    B, deter_size, stoch_size, hidden_size = 4, 32, 16, 64

    torch.manual_seed(0)
    rssm_cpu = RSSMCore(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    ).cpu()

    rssm_gpu = RSSMCore(
        deter_size=deter_size,
        stoch_size=stoch_size,
        hidden_size=hidden_size,
    ).cuda()

    rssm_gpu.load_state_dict(rssm_cpu.state_dict())

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    out_cpu = rssm_cpu(h, z)
    out_gpu = rssm_gpu(h.cuda(), z.cuda()).cpu()

    torch.testing.assert_close(out_cpu, out_gpu)
