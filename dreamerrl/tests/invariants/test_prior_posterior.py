import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def _wm(device="cpu"):
    obs = gym.spaces.Box(0, 1, shape=(8,), dtype=np.float32)
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5, value_bins=41)
    return WorldModel(obs_space=obs, latent=latent, net=net, device=torch.device(device))


@pytest.mark.invariants
def test_prior_posterior_prob_simplex():
    wm = _wm()
    state = wm.init_state(batch_size=4)
    obs = torch.rand(4, 8)
    action = torch.nn.functional.one_hot(torch.randint(0, 5, (4,)), num_classes=5).float()

    out = wm.observe_step(state, obs, action)
    post = out["post_stats"]
    prior = out["prior_stats"]

    # V3: probs shape = (B, stoch_size, num_classes)
    assert post["probs"].dim() == 3
    assert prior["probs"].dim() == 3

    # Each categorical distribution must sum to 1
    assert torch.allclose(post["probs"].sum(-1), torch.ones_like(post["probs"].sum(-1)))
    assert torch.allclose(prior["probs"].sum(-1), torch.ones_like(prior["probs"].sum(-1)))


@pytest.mark.invariants
def test_kl_non_negative():
    wm = _wm()
    state = wm.init_state(batch_size=4)
    obs = torch.rand(4, 8)
    action = torch.nn.functional.one_hot(torch.randint(0, 5, (4,)), num_classes=5).float()

    out = wm.observe_step(state, obs, action)
    kl = out["kl"]

    assert torch.isfinite(kl).all()
    assert (kl >= 0).all()
