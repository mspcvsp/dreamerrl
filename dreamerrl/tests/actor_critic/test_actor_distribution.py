import pytest
import torch
from torch.distributions import Categorical

from dreamerrl.models.actor import Actor
from dreamerrl.models.world_model import WorldModelState


@pytest.mark.actor_critic
def test_actor_logits_shape():
    B = 8
    deter_size = 64
    stoch_size = 32
    hidden_size = 128
    action_dim = 5

    actor = Actor(deter_size, stoch_size, hidden_size, action_dim)

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    logits = actor(h, z)
    assert logits.shape == (B, action_dim)
    assert torch.isfinite(logits).all()


@pytest.mark.actor_critic
def test_actor_distribution_valid():
    B = 8
    deter_size = 64
    stoch_size = 32
    hidden_size = 128
    action_dim = 5

    actor = Actor(deter_size, stoch_size, hidden_size, action_dim)

    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    logits = actor(h, z)
    dist = Categorical(logits=logits)

    # Distribution must be valid
    assert dist.probs.shape == (B, action_dim)
    assert torch.isfinite(dist.logits).all()
    assert torch.isfinite(dist.probs).all()
    assert (dist.probs >= 0).all()
    assert torch.allclose(dist.probs.sum(dim=-1), torch.ones(B))


@pytest.mark.actor_critic
def test_actor_act_shapes():
    B = 8
    deter_size = 64
    stoch_size = 32
    hidden_size = 128
    action_dim = 5

    actor = Actor(deter_size, stoch_size, hidden_size, action_dim)

    state = WorldModelState(
        h=torch.randn(B, deter_size),
        z=torch.randn(B, stoch_size),
    )

    actions, logprobs = actor.act(state)

    assert actions.shape == (B,)
    assert logprobs.shape == (B,)
    assert torch.isfinite(logprobs).all()


@pytest.mark.actor_critic
@pytest.mark.cpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_actor_cpu_gpu_equivalence():
    B = 4
    deter_size = 64
    stoch_size = 32
    hidden_size = 128
    action_dim = 5

    # CPU actor
    actor_cpu = Actor(deter_size, stoch_size, hidden_size, action_dim).cpu()

    # GPU actor
    actor_gpu = Actor(deter_size, stoch_size, hidden_size, action_dim).cuda()

    # Same inputs
    h = torch.randn(B, deter_size)
    z = torch.randn(B, stoch_size)

    # Move to GPU
    h_gpu = h.cuda()
    z_gpu = z.cuda()

    # Forward pass
    logits_cpu = actor_cpu(h, z)
    logits_gpu = actor_gpu(h_gpu, z_gpu).cpu()

    # Numerical equivalence
    assert torch.allclose(logits_cpu, logits_gpu, atol=1e-5, rtol=1e-5)
