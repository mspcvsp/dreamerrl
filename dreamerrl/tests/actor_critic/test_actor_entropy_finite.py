import torch
from torch.distributions import Categorical


def test_actor_entropy_finite(actor, world_model):
    B = 4
    h = torch.randn(B, world_model.deter_size)
    z = torch.randn(B, world_model.stoch_size)

    logits = actor(h, z)
    dist = Categorical(logits=logits)

    entropy = dist.entropy()
    assert torch.isfinite(entropy).all()
