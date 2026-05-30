import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Space

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


@pytest.fixture
def latent():
    return LatentConfig(
        deter_size=200,
        stoch_size=30,
        num_classes=32,
    )


@pytest.fixture
def net():
    return NetworkConfig(
        hidden_size=256,
        action_dim=5,
        value_bins=41,
        bin_min=-20.0,
        bin_max=20.0,
    )


class DummyActor(torch.nn.Module):
    """
    Minimal actor for invariants tests.
    Consumes V3 latents (B, K, C) and returns logits (B, action_dim).
    """

    def __init__(self, latent: LatentConfig, net: NetworkConfig) -> None:
        super().__init__()
        assert net.action_dim is not None
        in_features = latent.deter_size + latent.stoch_size * latent.num_classes
        self.fc = torch.nn.Linear(in_features, net.action_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Flatten z from (B, K, C) → (B, K*C)
        z_flat = z.view(z.size(0), -1)
        x = torch.cat([h, z_flat], dim=-1)
        return self.fc(x)


@pytest.fixture
def dummy_actor(latent, net):
    return DummyActor(latent, net)


class DummyObsSpace(Space):
    """
    Minimal stand‑in for gymnasium.spaces.Box that:
      • satisfies Pylance type checking
      • satisfies WorldModel requirements
      • avoids assigning to read‑only properties
      • avoids randn(tuple) type errors
    """

    def __init__(self, obs_dim: int):
        # Call parent constructor with correct dtype + shape
        super().__init__(shape=(obs_dim,), dtype=np.float32)

        # Store shape as a normal attribute (NOT overriding the property)
        self._shape = (obs_dim,)
        self._dtype = np.float32

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def sample(self):
        # Pylance wants explicit ints, not a tuple
        return torch.randn(*self._shape, dtype=torch.float32)
