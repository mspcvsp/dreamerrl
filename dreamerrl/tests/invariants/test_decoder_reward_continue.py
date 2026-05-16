import gymnasium as gym
import numpy as np
import torch

from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def _wm():
    obs = gym.spaces.Box(0, 1, shape=(8,), dtype=np.float32)
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5, value_bins=41)
    return WorldModel(obs_space=obs, latent=latent, net=net)


def test_decoder_deterministic():
    wm = _wm()
    h = torch.randn(4, wm.latent.deter_size)
    z = torch.randn(4, wm.latent.z_dim)

    out1 = wm.decoder(h, z)
    out2 = wm.decoder(h, z)

    assert torch.allclose(out1, out2)


def test_reward_continue_finite():
    wm = _wm()
    h = torch.randn(4, wm.latent.deter_size)
    z = torch.randn(4, wm.latent.z_dim)

    reward = wm.reward_head(h, z)
    cont = wm.continue_head(h, z)

    assert torch.isfinite(reward).all()
    assert torch.isfinite(cont).all()
