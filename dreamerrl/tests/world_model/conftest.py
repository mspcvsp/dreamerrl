import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.world_model import WorldModel


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
        use_stochastic_latent=True,
    )


@pytest.fixture
def world_model(device, obs_space, action_dim, world_model_config):
    wm = WorldModel(
        obs_space=obs_space,
        action_dim=action_dim,
        device=device,
        **world_model_config,
    )
    return wm.to(device)


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
