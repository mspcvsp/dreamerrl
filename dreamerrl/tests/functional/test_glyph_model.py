"""Categorical IDs, spatial gradients, reconstruction units, and clock handling."""

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dreamerrl.env.minihack.minihack_utils import glyph_vocab_size
from dreamerrl.models.glyph_model import GlyphDecoder, GlyphEncoder, GlyphReconstruction
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.core.world_model_update import world_model_training_step
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


def glyph_world(clock=False):
    cfg = DreamerConfig()
    cfg.world.disable_aux_losses = True
    cfg.world.glyph_shape = (3, 5)
    cfg.world.glyph_vocab_size = 11
    wm = WorldModel(
        obs_space=gym.spaces.Box(0, 10, (15 + int(clock),), np.float32),
        dreamer_cfg=cfg,
        latent=LatentConfig(deter_size=8, num_classes=2, stoch_size=3),
        net=NetworkConfig(hidden_size=16, action_dim=3, value_bins=21, bin_min=-5, bin_max=5),
    )
    wm.posterior.deterministic_latent_for_tests = True
    wm.prior.deterministic_latent_for_tests = True
    return wm


def test_embedding_is_invariant_to_consistent_glyph_relabeling():
    encoder = GlyphEncoder((3, 5), 11, 16)
    obs = torch.randint(11, (2, 15)).float()
    expected = encoder(obs)
    permutation = torch.randperm(11)
    original = encoder.embedding.weight.detach().clone()
    with torch.no_grad():
        encoder.embedding.weight[permutation] = original
    torch.testing.assert_close(encoder(permutation[obs.long()].float()), expected)


@pytest.mark.parametrize("value", [-1.0, 11.0, 1.5, float("nan")])
def test_invalid_glyphs_fail_instead_of_silently_clamping(value):
    encoder = GlyphEncoder((3, 5), 11, 16)
    obs = torch.zeros(1, 15)
    obs[0, 0] = value
    with pytest.raises(ValueError, match="Glyph"):
        encoder(obs)


def test_categorical_loss_matches_dense_cross_entropy_and_gradient():
    wm = glyph_world()
    decoder = wm.decoder
    assert isinstance(decoder, GlyphDecoder)
    h = torch.randn(2, 8)
    z = torch.randn(2, 2, 3)
    target = torch.randint(11, (2, 15)).float()
    decoded = decoder(h, z)
    expected = F.cross_entropy(
        decoder.classifier(decoded["glyph_features"]).reshape(-1, 11), target.long().flatten(), reduction="none"
    ).reshape(2, 15)
    actual = decoder.cell_loss(decoded, target)
    torch.testing.assert_close(actual, expected)
    parameters = tuple(decoder.parameters())
    expected_grads = torch.autograd.grad(expected.mean(), parameters, retain_graph=True)
    actual_grads = torch.autograd.grad(actual.mean(), parameters)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)
    assert decoder.predict(decoded).shape == target.shape


@pytest.mark.parametrize("clock", [False, True])
def test_training_reaches_spatial_modules_and_tracks_changed_cells(clock):
    wm = glyph_world(clock)
    obs = torch.zeros(2, 3, 15 + int(clock))
    obs[:, 1:, 4] = 3
    if clock:
        obs[..., -1] = torch.tensor([1.0, 0.5, 0.0])
    batch = dict(
        obs=obs,
        action=torch.zeros(2, 3, dtype=torch.long),
        reward=torch.zeros(2, 3),
        is_first=torch.tensor([[True, False, False]]).repeat(2, 1),
    )
    assert isinstance(wm.encoder, GlyphEncoder)
    assert isinstance(wm.decoder, GlyphDecoder)
    metrics = world_model_training_step(wm, batch)
    assert metrics.reconstruction_metrics["glyph_changed_cells"] == 2
    assert torch.isfinite(metrics.reconstruction_metrics["glyph_changed_loss"])
    metrics.total_loss.backward()
    parameters = dict(wm.named_parameters())
    for name in (
        "encoder.embedding.weight",
        "encoder.spatial.0.weight",
        "decoder.spatial.0.weight",
        "decoder.classifier.weight",
    ):
        grad = parameters[name].grad
        assert grad is not None and grad.abs().sum() > 0
    if clock:
        assert wm.decoder.clock_head is not None
        grad = wm.decoder.clock_head.weight.grad
        assert grad is not None and grad.abs().sum() > 0
    batch["is_first"][:, 1] = True
    metrics = world_model_training_step(wm, batch)
    assert metrics.reconstruction_metrics["glyph_changed_cells"] == 0
    assert "glyph_changed_accuracy" not in metrics.reconstruction_metrics


def test_chunked_vocabulary_loss_matches_dense_over_multiple_chunks():
    wm = glyph_world()
    assert isinstance(wm.decoder, GlyphDecoder)
    features = torch.randn(80, 15, 16, requires_grad=True)
    targets = torch.randint(11, (80, 15)).float()
    decoded: GlyphReconstruction = {"glyph_features": features, "clock": None}
    loss = wm.decoder.cell_loss(decoded, targets).mean()
    expected = F.cross_entropy(wm.decoder.classifier(features).reshape(-1, 11), targets.long().flatten())
    torch.testing.assert_close(loss, expected)
    gradient = torch.autograd.grad(loss, features, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(expected, features)[0])


@pytest.mark.parametrize("finite", [False, True])
def test_real_minihack_spatial_update_and_checkpoint(tmp_path, finite):
    from dreamerrl.tests.manual.utils import minihack_config
    from dreamerrl.training.trainer import DreamerTrainer

    cfg = minihack_config(0, tmp_path)
    cfg.env.finite_horizon = finite
    with gym.make(cfg.env.env_id) as probe:
        obs, _ = probe.reset(seed=0)
        rows, columns = obs["glyphs"].shape
        cfg.world.glyph_shape = (int(rows), int(columns))
    cfg.world.glyph_vocab_size = glyph_vocab_size()
    trainer = DreamerTrainer(cfg)
    resumed = None
    try:
        trainer.train(1)
        assert isinstance(trainer.world.encoder, GlyphEncoder)
        assert isinstance(trainer.world.decoder, GlyphDecoder)
        assert trainer.env.obs_space.high.max() == glyph_vocab_size() - 1
        batch = trainer.replay.sample(2, seed=0)
        metrics = world_model_training_step(trainer.world, batch)
        assert torch.isfinite(metrics.total_loss)
        assert "glyph_accuracy" in metrics.reconstruction_metrics
        assert metrics.reconstruction_metrics["glyph_player_count"] > 0
        assert (
            metrics.reconstruction_metrics["glyph_player_paired_count"]
            == metrics.reconstruction_metrics["glyph_player_persistence_count"]
        )
        path = tmp_path / "spatial.pt"
        trainer.save_checkpoint(path)
        resumed = DreamerTrainer.from_checkpoint(path, cuda=False)
        assert isinstance(resumed.world.encoder, GlyphEncoder)
        assert isinstance(resumed.world.decoder, GlyphDecoder)
        from dreamerrl.eval.minihack_eval import evaluate_minihack

        assert evaluate_minihack(cfg, trainer.world, trainer.actor, [1234]) == evaluate_minihack(
            resumed.cfg, resumed.world, resumed.actor, [1234]
        )
        assert resumed.cfg.world.glyph_shape == cfg.world.glyph_shape
        torch.testing.assert_close(trainer.world.encoder(batch["obs"][:, 0]), resumed.world.encoder(batch["obs"][:, 0]))
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
        if resumed is not None:
            resumed.env.venv.close()
            resumed.tb.close()


def test_changed_cell_weight_uses_normalized_categorical_loss(monkeypatch):
    wm = glyph_world()
    assert isinstance(wm.decoder, GlyphDecoder)
    wm.cfg.glyph_changed_weight = 4
    obs = torch.zeros(1, 3, 15)
    obs[:, 1:, 4] = 3
    batch = dict(obs=obs, action=torch.zeros(1, 3, dtype=torch.long), reward=torch.zeros(1, 3))
    original = wm.decoder.cell_loss
    captured = []

    def capture(*args):
        loss = original(*args)
        captured.append(loss)
        return loss

    monkeypatch.setattr(wm.decoder, "cell_loss", capture)
    metrics = world_model_training_step(wm, batch)
    weights = torch.ones(1, 3, 15)
    weights[:, 1, 4] = 4
    torch.testing.assert_close(metrics.recon_loss, (captured[0] * weights).sum() / weights.sum())
    assert metrics.reconstruction_metrics["glyph_changed_cells"] == 1


def test_group_balanced_glyph_training_uses_separate_changed_and_static_means(monkeypatch):
    wm = glyph_world()
    assert isinstance(wm.decoder, GlyphDecoder)
    wm.cfg.glyph_loss_balance = "groups"
    wm.cfg.glyph_salient_fraction = 0.6
    wm.cfg.reconstruction_scale = 2
    obs = torch.zeros(1, 3, 15)
    obs[:, 1:, 4] = 3
    batch = dict(obs=obs, action=torch.zeros(1, 3, dtype=torch.long), reward=torch.zeros(1, 3))
    original = wm.decoder.cell_loss
    captured = []

    def capture(*args):
        loss = original(*args)
        captured.append(loss)
        return loss

    monkeypatch.setattr(wm.decoder, "cell_loss", capture)
    metrics = world_model_training_step(wm, batch)
    mask = torch.zeros(1, 3, 15, dtype=torch.bool)
    mask[:, 1, 4] = True
    expected = 2 * (0.6 * captured[0][mask].mean() + 0.4 * captured[0][~mask].mean())
    torch.testing.assert_close(metrics.recon_loss, expected)
    torch.testing.assert_close(metrics.reconstruction_metrics["glyph_salient_loss_mass"], torch.tensor(0.6))
    metrics.recon_loss.backward()
    grad = wm.posterior.fc_logits.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_stationary_foreground_gets_salient_budget_without_movement(monkeypatch):
    wm = glyph_world()
    assert isinstance(wm.decoder, GlyphDecoder)
    wm.cfg.glyph_loss_balance = "groups"
    wm.cfg.glyph_salient_fraction = 0.6
    wm.cfg.glyph_foreground_ids = (3,)
    obs = torch.zeros(1, 3, 15)
    obs[..., 4] = 3
    captured = []
    original = wm.decoder.cell_loss

    def capture(*args):
        ce = original(*args)
        captured.append(ce)
        return ce

    monkeypatch.setattr(wm.decoder, "cell_loss", capture)
    metrics = world_model_training_step(
        wm, dict(obs=obs, action=torch.zeros(1, 3, dtype=torch.long), reward=torch.zeros(1, 3))
    )
    foreground = obs == 3
    expected = 0.6 * captured[0][foreground].mean() + 0.4 * captured[0][~foreground].mean()
    torch.testing.assert_close(metrics.recon_loss, expected)
    assert metrics.reconstruction_metrics["glyph_changed_cells"] == 0
    assert metrics.reconstruction_metrics["glyph_foreground_cells"] == 3
    torch.testing.assert_close(metrics.reconstruction_metrics["glyph_salient_loss_mass"], torch.tensor(0.6))
