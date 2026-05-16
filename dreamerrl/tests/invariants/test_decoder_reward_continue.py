import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from dreamerrl.models.categorical_kl import KLConfig
from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def _wm():
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5, value_bins=41)
    obs_space = Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)  # dummy obs space
    return WorldModel(
        obs_space=obs_space,
        latent=latent,
        net=net,
        free_bits=0.0,
        kl_cfg=KLConfig(require_nonzero=False),
        device=torch.device("cpu"),
    )


@pytest.mark.invariants
def test_decoder_deterministic():
    wm = _wm()
    h = torch.randn(4, wm.latent.deter_size)
    # V3 factored latent
    z = torch.randn(4, wm.latent.stoch_size, wm.latent.num_classes)

    out1 = wm.decoder(h, z)
    out2 = wm.decoder(h, z)

    torch.testing.assert_close(out1, out2)


@pytest.mark.invariants
def test_reward_continue_finite():
    wm = _wm()
    h = torch.randn(4, wm.latent.deter_size)
    # V3 factored latent
    z = torch.randn(4, wm.latent.stoch_size, wm.latent.num_classes)

    reward = wm.reward_head(h, z)
    cont = wm.continue_head(h, z)

    assert reward.shape == (4, 1)
    assert cont.shape == (4, 1)
    assert torch.isfinite(reward).all()
    assert torch.isfinite(cont).all()
