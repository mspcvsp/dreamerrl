import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.mark.invariants
def test_decoder_symlog_consistency():
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=3)
    obs_space = Box(low=0, high=1, shape=(8,), dtype=np.float32)

    wm = WorldModel(obs_space=obs_space, latent=latent, net=net)

    B = 4
    h = torch.randn(B, latent.deter_size)
    z = torch.randn(B, latent.stoch_size, latent.num_classes)

    recon = wm.decoder(h, z)
    assert torch.isfinite(recon).all()
    assert recon.shape[-1] == obs_space.shape[0]
