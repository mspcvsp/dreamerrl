import numpy as np
import torch
from gymnasium.spaces import Box

from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def test_world_model_observe_step_keys_and_shapes():
    B = 4
    obs_space = Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
    action_dim = 3

    latent = LatentConfig(deter_size=200, stoch_size=30, num_classes=32)
    net = NetworkConfig(hidden_size=256, action_dim=action_dim, value_bins=41)

    wm = WorldModel(
        obs_space=obs_space,
        action_dim=action_dim,
        latent=latent,
        net=net,
        free_bits=0.0,
        device=torch.device("cpu"),
    )

    state = wm.init_state(B)
    obs = torch.zeros(B, *obs_space.shape)
    action = torch.zeros(B, action_dim)
    reward = torch.zeros(B)
    is_first = torch.zeros(B, dtype=torch.bool)
    is_last = torch.zeros(B, dtype=torch.bool)
    is_terminal = torch.zeros(B, dtype=torch.bool)

    out = wm.observe_step(
        prev_state=state,
        obs=obs,
        action=action,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )

    post = out["post"]
    assert post.h.shape == (B, latent.deter_size)
    assert post.z.shape == (B, latent.z_dim)
    assert out["reward_logits"].shape[-1] == net.value_bins
    assert out["cont_logits"].shape == (B,)
    assert out["kl"].shape == ()
