import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def _wm():
    obs = gym.spaces.Box(0, 1, shape=(8,), dtype=np.float32)
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5, value_bins=41)
    return WorldModel(obs_space=obs, latent=latent, net=net)


@pytest.mark.invariants
def test_observe_step_no_mutation():
    wm = _wm()
    state = wm.init_state(batch_size=4)
    state_clone = state.clone()

    obs = torch.rand(4, 8)
    action = torch.nn.functional.one_hot(torch.randint(0, 5, (4,)), num_classes=5).float()

    wm.observe_step(state, obs, action)

    assert torch.allclose(state.h, state_clone.h)
    assert torch.allclose(state.z, state_clone.z)


@pytest.mark.invariants
def test_observe_step_deterministic():
    wm = _wm()
    state = wm.init_state(batch_size=4)
    obs = torch.rand(4, 8)
    action = torch.nn.functional.one_hot(torch.randint(0, 5, (4,)), num_classes=5).float()

    out1 = wm.observe_step(state, obs, action)
    out2 = wm.observe_step(state, obs, action)

    # Deterministic state h must be stable
    assert torch.allclose(out1["post"].h, out2["post"].h, atol=1e-6)

    # Stochastic latent z is not deterministic — check invariants instead
    z1, z2 = out1["post"].z, out2["post"].z
    assert z1.shape == (4, wm.latent.num_classes, wm.latent.stoch_size)
    assert z2.shape == (4, wm.latent.num_classes, wm.latent.stoch_size)
    assert torch.isfinite(z1).all()
    assert torch.isfinite(z2).all()
