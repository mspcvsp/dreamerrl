import torch

from dreamerrl.models.actor import Actor
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def test_actor_forward_determinism():
    torch.manual_seed(0)

    B, deter, K, C = 4, 32, 4, 8
    latent = LatentConfig(deter_size=deter, stoch_size=C, num_classes=K)
    net = NetworkConfig(hidden_size=64, action_dim=5)

    actor = Actor(latent=latent, net=net)

    h = torch.randn(B, deter)
    z = torch.randn(B, K, C)

    logits1 = actor(h, z)
    logits2 = actor(h, z)

    assert torch.allclose(logits1, logits2), "Actor forward pass is nondeterministic"
