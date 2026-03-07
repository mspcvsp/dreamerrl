import numpy as np
import torch
from gymnasium.spaces import Box

from dreamerrl.models.obs_encoder import build_obs_encoder


def test_encoder_idempotence():
    """
    ✔ What this test checks: Calling the encoder twice with the same obs produces identical outputs

    - No hidden state
    - No randomness
    - No mutation of inputs
    - No nondeterministic ops

    This is the pure function invariant.
    """
    B, obs_dim = 4, 8
    obs_space = Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype=np.float32)

    encoder = build_obs_encoder(obs_space, embed_dim=64)

    obs = torch.randn(B, obs_dim)

    out1 = encoder(obs)
    out2 = encoder(obs)

    torch.testing.assert_close(out1, out2)
