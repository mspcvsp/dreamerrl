import pytest
import torch

from dreamerrl.models.world_model import WorldModel
from tests.helpers.fake_dreamer_batch import make_fake_world_model_batch


def test_world_model_shapes(device):
    B, L, obs_dim = 4, 5, 8

    world = WorldModel(
        obs_space=None,  # not needed for shape tests
        action_dim=3,
        deter_size=32,
        stoch_size=16,
        encoder_hidden=64,
        rssm_hidden=64,
        decoder_hidden=64,
        reward_hidden=64,
        use_stochastic_latent=True,
        device=device,
    ).to(device)

    batch = make_fake_world_model_batch(B, L, obs_dim, device)
    obs = batch["state"]

    state = world.init_state(B)

    out = world.observe_step(state, obs[:, 0])

    assert out["state"].h.shape == (B, 32)
    assert out["state"].z.shape == (B, 16)
    assert out["recon"].shape == (B, obs_dim)
    assert out["reward_pred"].shape == (B, 1)
    assert torch.isfinite(out["kl"]).all()
