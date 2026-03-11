import pytest
import torch


def test_prior_cpu_gpu_equivalence(make_world_model):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    wm_cpu = make_world_model().cpu()
    wm_gpu = make_world_model().cuda()
    wm_gpu.prior.load_state_dict(wm_cpu.prior.state_dict())

    B = 4
    h = torch.randn(B, wm_cpu.deter_size)

    out_cpu = wm_cpu.prior(h)
    out_gpu = wm_gpu.prior(h.cuda())

    torch.testing.assert_close(out_cpu["mean"], out_gpu["mean"].cpu(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_cpu["std"], out_gpu["std"].cpu(), atol=1e-6, rtol=1e-6)
