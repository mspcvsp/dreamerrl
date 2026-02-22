import gymnasium as gym
import numpy as np
import pytest

from dreamerrl.agents.dreamer_agent import DreamerAgent
from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel


@pytest.fixture
def obs_space():
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(8,),
        dtype=np.float32,
    )


@pytest.fixture
def agent(device, obs_space):
    world = WorldModel(
        obs_space=obs_space,
        action_dim=3,
        deter_size=32,
        stoch_size=16,
        encoder_hidden=64,
        rssm_hidden=64,
        decoder_hidden=64,
        reward_hidden=64,
        use_stochastic_latent=True,
        device=device,
    ).to(device)

    actor = Actor(32, 16, 64, 3).to(device)
    critic = ValueHead(32, 16, 64).to(device)

    return DreamerAgent(world, actor, critic, device)
