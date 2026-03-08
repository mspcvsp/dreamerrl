import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.world_model import WorldModel, WorldModelState


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
        use_stochastic_latent=True,
    )


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


@pytest.fixture
def state_to_cpu():
    def _to_cpu(state: WorldModelState) -> WorldModelState:
        return WorldModelState(
            h=state.h.cpu(),
            z=state.z.cpu(),
            prior_stats=({k: v.cpu() for k, v in state.prior_stats.items()} if state.prior_stats is not None else None),
            post_stats=({k: v.cpu() for k, v in state.post_stats.items()} if state.post_stats is not None else None),
        )

    return _to_cpu
