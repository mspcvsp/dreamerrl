import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel, WorldModelState
from dreamerrl.replay_buffer import ReplayBuffer
from dreamerrl.training.test_trainer import _TestDreamerTrainer


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def obs_dim():
    return 8


@pytest.fixture(scope="session")
def obs_space(obs_dim):
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32,
    )


@pytest.fixture(scope="session")
def action_dim():
    return 3


@pytest.fixture(scope="session")
def world_model_config():
    return dict(
        deter_size=32,
        stoch_size=16,
        encoder_hidden=64,
        rssm_hidden=64,
        decoder_hidden=64,
        reward_hidden=64,
    )


# ---------------------------------------------------------------------
# World model constructors
# ---------------------------------------------------------------------
@pytest.fixture
def make_world_model(device, obs_space, action_dim, world_model_config):
    def _make(**overrides):
        cfg = dict(world_model_config)
        cfg.update(overrides)
        wm = WorldModel(
            obs_space=obs_space,
            action_dim=action_dim,
            device=device,
            **cfg,
        )
        return wm.to(device)

    return _make


@pytest.fixture
def world_model(device, obs_space, action_dim, world_model_config):
    wm = WorldModel(
        obs_space=obs_space,
        action_dim=action_dim,
        device=device,
        **world_model_config,
    )
    return wm.to(device)


# ---------------------------------------------------------------------
# Actor / Critic fixtures (needed for trainer tests)
# ---------------------------------------------------------------------
@pytest.fixture
def actor(world_model, action_dim, device):
    return Actor(
        world_model.deter_size,
        world_model.stoch_size,
        hidden_size=128,
        action_dim=action_dim,
    ).to(device)


@pytest.fixture
def critic(world_model, device):
    return ValueHead(
        world_model.deter_size,
        world_model.stoch_size,
        hidden_size=128,
    ).to(device)


# ---------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------
@pytest.fixture
def fake_obs(device, obs_dim):
    torch.manual_seed(0)
    return torch.randn(4, obs_dim, device=device)


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
@pytest.fixture
def test_trainer(world_model, actor, critic, replay_buffer_factory, device):
    return _TestDreamerTrainer(
        world_model=world_model,
        actor=actor,
        critic=critic,
        replay_buffer=replay_buffer_factory(),
        device=device,
    )


# ---------------------------------------------------------------------
# Imagination input
# ---------------------------------------------------------------------
@pytest.fixture
def imagine_input(world_model, batch_size=4, device="cpu"):
    h = torch.randn(batch_size, world_model.deter_size, device=device)
    z = torch.randn(batch_size, world_model.stoch_size, device=device)
    return WorldModelState(h=h, z=z)


# ---------------------------------------------------------------------
# RSSM + obs fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def rssm(world_model):
    return world_model.rssm


@pytest.fixture
def obs_batch(obs_space, action_dim, batch_size=4, device="cpu"):
    obs = torch.randn(batch_size, *obs_space.shape, device=device)
    action = torch.randint(0, action_dim, (batch_size,), device=device)
    return {"obs": obs, "action": action}


@pytest.fixture
def obs_input(obs_batch):
    return obs_batch


@pytest.fixture
def replay_buffer_factory(device, obs_dim):
    def make(
        num_envs=4,
        capacity_episodes=100,
        min_episode_len=2,
        store_device=torch.device("cpu"),
    ):
        return ReplayBuffer(
            num_envs=num_envs,
            obs_dim=obs_dim,
            capacity_episodes=capacity_episodes,
            device=device,
            store_device=store_device,
            min_episode_len=min_episode_len,
        )

    return make
