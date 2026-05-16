import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.fixture
def dummy_batch():
    B, L, obs_dim, action_dim = 4, 5, 8, 5

    return {
        "obs": torch.rand(B, L, obs_dim),
        "action": torch.nn.functional.one_hot(torch.randint(0, action_dim, (B, L)), num_classes=action_dim).float(),
        "reward": torch.randn(B, L),
        "is_terminal": torch.zeros(B, L),
        "is_first": torch.zeros(B, L),
        "is_last": torch.zeros(B, L),
    }


@pytest.fixture
def world_model():
    """
    Minimal world model fixture for invariants testing.
    Matches the latent + network config used in your tests.
    """

    obs_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(8,),  # matches dummy_batch obs_dim
        dtype=np.float32,
    )

    latent = LatentConfig(
        deter_size=200,
        stoch_size=30,
        num_classes=32,
    )

    net = NetworkConfig(
        hidden_size=256,
        action_dim=5,  # matches dummy_batch action_dim
        value_bins=41,
    )

    wm = WorldModel(
        obs_space=obs_space,
        latent=latent,
        net=net,
        device=torch.device("cpu"),
    )

    return wm
