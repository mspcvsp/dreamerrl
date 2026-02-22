import torch

from dreamerrl.models.actor import Actor


def test_actor_shapes(device):
    B = 4
    deter, stoch, action_dim = 32, 16, 5

    actor = Actor(deter, stoch, 64, action_dim).to(device)

    h = torch.randn(B, deter, device=device)
    z = torch.randn(B, stoch, device=device)

    logits = actor(h, z)
    assert logits.shape == (B, action_dim)
