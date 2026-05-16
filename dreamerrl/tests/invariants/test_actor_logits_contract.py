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
    z = torch.randn(B, latent.z_dim)

    # Forward pass
    logits = actor(h, z)

    # 1. Shape check
    assert logits.shape == (B, net.action_dim), f"Actor must output shape (B, action_dim), got {logits.shape}"

    # 2. Finite values
    assert torch.isfinite(logits).all(), "Actor logits contain NaN or Inf"

    # 3. No softmax inside actor
    probs = torch.softmax(logits, dim=-1)
    assert not torch.allclose(logits, probs, atol=1e-5), "Actor appears to output probabilities instead of logits"

    # 4. Stable under no_grad
    with torch.no_grad():
        logits_ng = actor(h, z)

    assert torch.allclose(logits, logits_ng, atol=1e-6), "Actor logits differ under no_grad — should be identical"


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

    # Copy weights exactly
    actor_gpu.load_state_dict(actor_cpu.state_dict())

    B = 8
    h_cpu = torch.randn(B, latent.deter_size)
    z_cpu = torch.randn(B, latent.z_dim)

    h_gpu = h_cpu.to("cuda")
    z_gpu = z_cpu.to("cuda")

    with torch.no_grad():
        logits_cpu = actor_cpu(h_cpu, z_cpu)
        logits_gpu = actor_gpu(h_gpu, z_gpu).to("cpu")

    assert torch.allclose(logits_cpu, logits_gpu, atol=1e-5), "Actor logits differ between CPU and GPU"
