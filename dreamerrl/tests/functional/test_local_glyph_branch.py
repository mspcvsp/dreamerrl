"""Local crop geometry, stateless missing-player handling, and end-to-end gradients."""

import copy

import gymnasium as gym
import numpy as np
import torch

from dreamerrl.models.glyph_model import GlyphEncoder, PlayerCenteredGlyphCrop
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.core.world_model_update import world_model_training_step
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


def test_row_major_crop_boundaries_missing_and_batch_isolation():
    crop = PlayerCenteredGlyphCrop(20, (9,))
    grids = torch.tensor(
        [
            [[0, 1, 2], [3, 9, 5], [6, 7, 8]],
            [[9, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[9, 9, 2], [3, 4, 5], [6, 7, 8]],
        ]
    )
    actual = crop(grids)
    assert actual[0].tolist() == [0, 1, 2, 3, 9, 5, 6, 7, 8]
    assert actual[1].tolist() == [20, 20, 20, 20, 9, 1, 20, 3, 4]
    assert actual[2:].tolist() == [[21] * 9, [21] * 9]
    # A missing row never borrows the position of a preceding call or batch row.
    torch.testing.assert_close(crop(grids[2:3]), actual[2:3])
    assert crop(torch.full((1, 3, 3), 2)).tolist() == [[21] * 9]


def test_disabled_branch_preserves_state_dict_and_output_dimensions():
    torch.manual_seed(7)
    baseline = GlyphEncoder((3, 3), 20, 16)
    torch.manual_seed(7)
    disabled = GlyphEncoder((3, 3), 20, 16, local_branch=False)
    disabled.load_state_dict(baseline.state_dict(), strict=True)
    obs = torch.arange(9).float()[None]
    torch.testing.assert_close(baseline(obs), disabled(obs), rtol=0, atol=0)
    assert baseline.output_size == disabled.output_size == 16
    enabled = GlyphEncoder((3, 3), 20, 16, local_branch=True, player_ids=(9,))
    assert enabled.output_size == 160
    assert enabled(obs).shape == (1, 160)


def test_local_branch_world_model_loss_and_gradient():
    cfg = DreamerConfig()
    cfg.world.disable_aux_losses = True
    cfg.world.glyph_shape, cfg.world.glyph_vocab_size = (3, 3), 20
    cfg.world.glyph_local_branch, cfg.world.glyph_player_ids = True, (9,)
    net = NetworkConfig(hidden_size=16, action_dim=3, value_bins=21, bin_min=-5, bin_max=5)
    wm = WorldModel(
        obs_space=gym.spaces.Box(0, 19, (9,), np.float32),
        dreamer_cfg=cfg,
        latent=LatentConfig(deter_size=8, num_classes=2, stoch_size=3),
        net=net,
    )
    observation = torch.arange(9).float().repeat(2, 4, 1)
    observation[:, :, 4] = 9
    batch = dict(
        obs=observation,
        action=torch.zeros(2, 4, dtype=torch.long),
        reward=torch.zeros(2, 4),
        is_first=torch.tensor([[True, False, False, False]] * 2),
        done=torch.zeros(2, 4),
        is_terminal=torch.zeros(2, 4),
    )
    metrics = world_model_training_step(wm, batch)
    assert torch.isfinite(metrics.total_loss)
    metrics.total_loss.backward()
    assert isinstance(wm.encoder, GlyphEncoder)
    grad = wm.encoder.local_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert wm.embed_size == 160 and wm.posterior.fc1.in_features == 168
    copy.deepcopy(wm).load_state_dict(wm.state_dict(), strict=True)
