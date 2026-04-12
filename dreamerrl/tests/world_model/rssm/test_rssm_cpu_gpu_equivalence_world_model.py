import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rssm_cpu_gpu_equivalence(world_model):
    B = 4
    h = torch.randn(B, world_model.deter_size)
    embed = torch.randn(B, world_model.embed_size)

    cpu_out = world_model.posterior(h, embed)
    gpu_out = world_model.to("cuda").posterior(h.cuda(), embed.cuda())

    assert torch.allclose(cpu_out["mean"], gpu_out["mean"].cpu(), atol=1e-4, rtol=1e-4)
    assert torch.allclose(cpu_out["std"], gpu_out["std"].cpu(), atol=1e-4, rtol=1e-4)
