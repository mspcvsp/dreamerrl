"""Minimal stateless auxiliary extension; examples are synthetic, not registered objectives."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.core.world_model_update import world_model_training_step
from dreamerrl.utils.types import AuxObjectiveConfig, DreamerConfig, LatentConfig, NetworkConfig


def squared_h(h, z, batch):
    return h.square().mean()


def squared_z(h, z, batch):
    return z.square().mean()


def fail_if_called(h, z, batch):
    raise AssertionError("Disabled callback evaluated")


def make(objectives, disabled=False):
    cfg = DreamerConfig()
    cfg.world.aux_objectives, cfg.world.disable_aux_losses = objectives, disabled
    cfg.train.deterministic_env = True
    return WorldModel(
        obs_space=gym.spaces.Box(-100, 100, (4,), np.float32),
        dreamer_cfg=cfg,
        latent=LatentConfig(deter_size=16, num_classes=3, stoch_size=4),
        net=NetworkConfig(hidden_size=16, action_dim=3, value_bins=21),
    )


def batch():
    return dict(
        obs=torch.randn(2, 4, 4),
        action=torch.zeros(2, 4, dtype=torch.long),
        reward=torch.randn(2, 4),
        is_terminal=torch.zeros(2, 4),
    )


def test_explicit_weights_are_additive_not_averaged():
    model = make([AuxObjectiveConfig("test_h", squared_h, 0.2), AuxObjectiveConfig("test_z", squared_z, 0.7)])
    data = batch()
    weighted = world_model_training_step(model, data)
    model.cfg.disable_aux_losses = True
    baseline = world_model_training_step(model, data)
    torch.testing.assert_close(
        weighted.total_loss - baseline.total_loss, 0.2 * weighted.aux_losses[0] + 0.7 * weighted.aux_losses[1]
    )
    assert not any("aux" in key for key in model.state_dict())
    weighted.aux_losses[0].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.rssm.parameters())


@pytest.mark.parametrize("disabled,weight", [(True, 1), (False, 0)])
def test_disabled_or_zero_weight_skips_callback(disabled, weight):
    model = make([AuxObjectiveConfig("skip", fail_if_called, weight)], disabled)
    metrics = world_model_training_step(model, batch())
    assert torch.isfinite(metrics.total_loss)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError, match="weights"):
        make([AuxObjectiveConfig("test", squared_h, weight)])


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        make([AuxObjectiveConfig("x", squared_h), AuxObjectiveConfig("x", squared_z)])


def vector_loss(h, z, batch):
    return h.sum(-1)


def nan_loss(h, z, batch):
    return h.sum() * float("nan")


@pytest.mark.parametrize("fn", [vector_loss, nan_loss])
def test_malformed_loss_rejected(fn):
    model = make([AuxObjectiveConfig("bad", fn)])
    with pytest.raises(ValueError, match="finite scalar"):
        world_model_training_step(model, batch())


def test_representation_gradients_reach_encoder_posterior_and_rssm():
    model = make([AuxObjectiveConfig("test_z", squared_z)])
    result = world_model_training_step(model, batch())
    result.aux_losses[0].backward()
    for module in (model.encoder, model.posterior, model.rssm):
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters()
        )


def test_resolved_configuration_records_explicit_weight_and_callback():
    pytest.importorskip("nle", reason="MiniHack run monitor belongs to the legacy profile")
    from dreamerrl.training.run_monitor import configuration

    model = make([AuxObjectiveConfig("test_h", squared_h, 0.25)])
    cfg = DreamerConfig()
    cfg.world = model.cfg
    result = configuration(cfg)
    assert result["world"]["aux_objectives"][0]["weight"] == 0.25
    assert result["world"]["aux_objectives"][0]["callback"].endswith("squared_h")
