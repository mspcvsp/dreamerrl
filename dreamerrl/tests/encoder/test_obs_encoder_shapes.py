import numpy as np
import torch
from gymnasium.spaces import Box

from dreamerrl.models.obs_encoder import build_obs_encoder, get_flat_obs_dim


def test_obs_encoder_shapes():
    B, obs_dim = 5, 8
    embed_dim = 64
    obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    encoder = build_obs_encoder(obs_space, embed_dim=embed_dim)

    x = torch.randn(B, get_flat_obs_dim(obs_space))
    y = encoder(x)

    assert y.shape == (B, embed_dim)
