import pytest
import torch


def test_posterior_cpu_gpu_equivalence(make_world_model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    wm_cpu = make_world_model().cpu()
    wm_gpu = make_world_model().cuda()
    wm_gpu.posterior.load_state_dict(wm_cpu.posterior.state_dict())

    B = 4
    h = torch.randn(B, wm_cpu.deter_size)
    embed = torch.randn(B, wm_cpu.embed_size)

    out_cpu = wm_cpu.posterior(h, embed)
    out_gpu = wm_gpu.posterior(h.cuda(), embed.cuda())

    torch.testing.assert_close(out_cpu["mean"], out_gpu["mean"].cpu(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_cpu["std"], out_gpu["std"].cpu(), atol=1e-6, rtol=1e-6)
