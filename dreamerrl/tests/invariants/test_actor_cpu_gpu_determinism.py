# dreamerrl/tests/invariants/test_actor_cpu_gpu_determinism.py
import pytest
import torch


@pytest.mark.invariants
@pytest.mark.actor_invariants
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_actor_cpu_gpu_determinism(actor, dummy_h, dummy_z_actor):
    """
    Actor forward pass must be deterministic across CPU and GPU.
    """

    actor_cpu = actor.cpu()
    actor_gpu = actor.cuda()

    h_cpu = dummy_h.cpu()
    z_cpu = dummy_z_actor.cpu()

    h_gpu = dummy_h.cuda()
    z_gpu = dummy_z_actor.cuda()

    logits_cpu = actor_cpu(h_cpu, z_cpu)
    logits_gpu = actor_gpu(h_gpu, z_gpu).cpu()

    assert torch.allclose(logits_cpu, logits_gpu, atol=1e-6)
