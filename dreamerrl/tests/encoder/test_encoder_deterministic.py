import numpy as np
import torch
from gymnasium.spaces import Box

from dreamerrl.models.obs_encoder import build_obs_encoder


def test_encoder_deterministic():
    """
    ✔ What this test checks - Two independently constructed encoders with the same seed produce identical outputs

    - Initialization is deterministic
    - No randomness in forward pass
    - No device‑dependent initialization

    This is the deterministic initialization invariant.
    """
    B, obs_dim = 4, 8
    obs_space = Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype=np.float32)

    # Fix seed so two encoders initialize identically
    torch.manual_seed(0)
    encoder1 = build_obs_encoder(obs_space, embed_dim=64)

    torch.manual_seed(0)
    encoder2 = build_obs_encoder(obs_space, embed_dim=64)

    obs = torch.randn(B, obs_dim)

    out1 = encoder1(obs)
    out2 = encoder2(obs)

    torch.testing.assert_close(out1, out2)
