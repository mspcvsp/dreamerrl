import pytest
import torch
from torch.distributions import Categorical

from dreamerrl.models.actor import Actor
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.mark.invariants
@pytest.mark.actor_invariants
def test_actor_entropy_invariance():
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5)
    actor = Actor(latent=latent, net=net)

    B = 6
    h = torch.randn(B, latent.deter_size)
    z = torch.randn(B, latent.num_classes, latent.stoch_size)

    logits = actor(h, z)
    dist = Categorical(logits=logits)
    entropy = dist.entropy()

    assert torch.isfinite(entropy).all()
    assert (entropy >= 0).all()
