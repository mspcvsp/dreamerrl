import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel, WorldModelState
from dreamerrl.training.replay_buffer import DreamerReplayBuffer
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


@pytest.fixture
def fake_batch(device, obs_dim):
    torch.manual_seed(0)
    B, L = 4, 5
    return {
        "state": torch.randn(B, L, obs_dim, device=device),
        "reward": torch.randn(B, L, device=device),
        "is_first": torch.zeros(B, L, dtype=torch.bool, device=device),
        "is_last": torch.zeros(B, L, dtype=torch.bool, device=device),
        "is_terminal": torch.zeros(B, L, dtype=torch.bool, device=device),
    }


# ---------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------
@pytest.fixture
def state_to_cpu():
    def _to_cpu(state: WorldModelState) -> WorldModelState:
        return WorldModelState(
            h=state.h.cpu(),
            z=state.z.cpu(),
            prior_stats=({k: v.cpu() for k, v in state.prior_stats.items()} if state.prior_stats else None),
            post_stats=({k: v.cpu() for k, v in state.post_stats.items()} if state.post_stats else None),
        )

    return _to_cpu


# ---------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------
@pytest.fixture
def replay_buffer_factory(obs_dim, device):
    def make():
        num_envs = 1
        capacity_episodes = 100
        rb = DreamerReplayBuffer(
            num_envs=num_envs,
            obs_dim=obs_dim,
            capacity_episodes=capacity_episodes,
            device=device,
        )
        torch.manual_seed(0)
        for _ in range(20):
            rb.add(
                state=torch.randn(obs_dim, device=device),
                action=torch.zeros(1, device=device),
                reward=torch.randn((), device=device),
                is_first=torch.tensor(True, device=device),
                is_last=torch.tensor(False, device=device),
                is_terminal=torch.tensor(False, device=device),
            )
            rb.add(
                state=torch.randn(obs_dim, device=device),
                action=torch.zeros(1, device=device),
                reward=torch.randn((), device=device),
                is_first=torch.tensor(False, device=device),
                is_last=torch.tensor(True, device=device),
                is_terminal=torch.tensor(False, device=device),
            )
        return rb

    return make


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
@pytest.fixture
def test_trainer(world_model, replay_buffer_factory, device):
    return _TestDreamerTrainer(
        world_model=world_model,
        actor=None,
        critic=None,
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
