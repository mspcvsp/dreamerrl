import numpy as np
import torch
from gymnasium.spaces import Box

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.core import actor_critic_update
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def test_actor_critic_update_shapes():
    B, T = 4, 5
    obs_dim = 8
    action_dim = 3

    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net_world = NetworkConfig(hidden_size=256, action_dim=action_dim, value_bins=41)
    net_actor = NetworkConfig(hidden_size=256, action_dim=action_dim)
    net_critic = NetworkConfig(hidden_size=256, value_bins=41)

    wm = WorldModel(
        obs_space=Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
        latent=latent,
        net=net_world,
        free_bits=0.0,
        device=torch.device("cpu"),
    )

    actor = Actor(latent=latent, net=net_actor)
    critic = ValueHead(latent=latent, net=net_critic)

    batch = {
        "state": torch.zeros(T, B, obs_dim),
        "action": torch.zeros(T, B, action_dim),
        "reward": torch.zeros(T, B),
        "is_terminal": torch.zeros(T, B, dtype=torch.bool),
    }

    actor_loss, critic_loss = actor_critic_update(
        world_model=wm,
        actor=actor,
        critic=critic,
        batch=batch,
        imagination_horizon=5,
        discount=0.99,
        lam=0.95,
    )

    assert actor_loss.shape == ()
    assert critic_loss.shape == ()
