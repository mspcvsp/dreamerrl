import torch

from dreamerrl.training.core.actor_critic_update import actor_critic_update


def test_actor_critic_seed_determinism(world_model, actor, critic, obs_batch):
    torch.manual_seed(0)
    a1, c1 = actor_critic_update(world_model, actor, critic, obs_batch, 5, 0.99, 0.95)

    torch.manual_seed(0)
    a2, c2 = actor_critic_update(world_model, actor, critic, obs_batch, 5, 0.99, 0.95)

    assert torch.allclose(a1, a2)
    assert torch.allclose(c1, c2)
