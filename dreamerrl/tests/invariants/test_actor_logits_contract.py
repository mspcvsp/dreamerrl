import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.mark.invariants
def test_actor_logits_contract_basic():
    """
    Actor must output:
    - logits, not actions
    - shape (B, action_dim)
    - finite values
    - stable under no_grad
    """

    torch.manual_seed(0)

    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5, value_bins=41)

    actor = Actor(latent=latent, net=net)

    B = 16
    h = torch.randn(B, latent.deter_size)
    # V3 factored latent
    z = torch.randn(B, latent.stoch_size, latent.num_classes)

    logits = actor(h, z)

    # 1. Shape check
    assert logits.shape == (B, net.action_dim)

    # 2. Finite values
    assert torch.isfinite(logits).all()

    # 3. No softmax inside actor
    probs = torch.softmax(logits, dim=-1)
    assert not torch.allclose(logits, probs, atol=1e-5)

    # 4. Stable under no_grad
    with torch.no_grad():
        logits_ng = actor(h, z)
    assert torch.allclose(logits, logits_ng, atol=1e-6)


@pytest.mark.invariants
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_actor_logits_cpu_gpu_determinism():
    """
    Actor must produce identical logits on CPU and GPU when given identical
    parameters and inputs.
    """

    torch.manual_seed(1)

    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5, value_bins=41)

    actor_cpu = Actor(latent=latent, net=net).to("cpu")
    actor_gpu = Actor(latent=latent, net=net).to("cuda")

    actor_gpu.load_state_dict(actor_cpu.state_dict())

    B = 8
    h_cpu = torch.randn(B, latent.deter_size)
    # V3 factored latent
    z_cpu = torch.randn(B, latent.stoch_size, latent.num_classes)

    h_gpu = h_cpu.to("cuda")
    z_gpu = z_cpu.to("cuda")
