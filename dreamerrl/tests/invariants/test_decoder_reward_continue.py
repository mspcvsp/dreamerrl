from dataclasses import replace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


def _wm():
    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=5, value_bins=41)
    obs_space = Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)  # dummy obs space

    dreamer_cfg = DreamerConfig()
    dreamer_cfg.world.kl_cfg = replace(dreamer_cfg.world.kl_cfg, require_nonzero=False)

    return WorldModel(
        dreamer_cfg=dreamer_cfg,
        obs_space=obs_space,
        latent=latent,
        net=net,
        device=torch.device("cpu"),
    )


@pytest.mark.invariants
@pytest.mark.decoder_invariants
def test_decoder_deterministic():
    wm = _wm()
    h = torch.randn(4, wm.latent.deter_size)
    # V3 factored latent
    z = torch.randn(4, wm.latent.num_classes, wm.latent.stoch_size)

    out1 = wm.decoder(h, z)
    out2 = wm.decoder(h, z)

    torch.testing.assert_close(out1, out2)


@pytest.mark.invariants
@pytest.mark.decoder_invariants
def test_reward_continue_finite():
    wm = _wm()
    h = torch.randn(4, wm.latent.deter_size)
    # V3 factored latent
    z = torch.randn(4, wm.latent.num_classes, wm.latent.stoch_size)

    reward = wm.reward_head(h, z)
    cont = wm.continue_head(h, z)

    assert reward.shape == (4, wm.net_cfg.value_bins)
    assert cont.shape == (4, wm.net_cfg.value_bins)
    assert torch.isfinite(reward).all()
    assert torch.isfinite(cont).all()
