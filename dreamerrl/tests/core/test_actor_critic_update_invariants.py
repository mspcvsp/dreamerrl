import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.training.core import actor_critic_update


def test_actor_critic_update_invariants(world_model, device):
    B, L = 2, 4
    obs_dim = world_model.flat_obs_dim
    deter = world_model.deter_size
    stoch = world_model.stoch_size

    batch = {
        "state": torch.randn(B, L, obs_dim, device=device),
        "reward": torch.randn(B, L, device=device),
    }

    num_classes = world_model.num_classes
    hidden_size = world_model.hidden_size

    actor = Actor(deter, stoch, hidden_size=hidden_size, num_classes=num_classes, action_dim=3).to(device)
    critic = ValueHead(deter, stoch, num_classes=num_classes, hidden_size=hidden_size).to(device)

    actor_loss, critic_loss = actor_critic_update(
        world_model=world_model,
        actor=actor,
        critic=critic,
        batch=batch,
        imagination_horizon=5,
        discount=0.99,
        lam=0.95,
    )

    assert actor_loss.dim() == 0
    assert critic_loss.dim() == 0
    assert torch.isfinite(actor_loss)
    assert torch.isfinite(critic_loss)
