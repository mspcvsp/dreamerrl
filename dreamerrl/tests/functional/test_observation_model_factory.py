"""Observation composition preserves the generic interface and initialization order."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models import world_model as world_module
from dreamerrl.models.base_obs_encoder import BaseEncoder
from dreamerrl.models.glyph_model import GlyphDecoder
from dreamerrl.models.observation_model import build_observation_model
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


def configuration(kind):
    cfg = DreamerConfig()
    cfg.world.disable_aux_losses = True
    dimension = 52
    if kind.startswith("glyph"):
        cfg.world.glyph_shape, cfg.world.glyph_vocab_size = (3, 5), 11
        dimension = 15
    if kind == "glyph_local_clock":
        cfg.world.glyph_local_branch, cfg.world.glyph_player_ids = True, (1,)
        dimension += 1
    if kind == "host":
        cfg.world.obs_encoder = "cage2_host"
        cfg.world.cage2_host_order = tuple(f"h{i}" for i in range(13))
        cfg.world.cage2_subnet_ids = (0,) * 4 + (1,) * 4 + (2,) * 5
    return cfg, gym.spaces.Box(0, 10, (dimension,), np.float32)


@pytest.mark.parametrize("kind", ["flat", "glyph", "glyph_local_clock", "host"])
def test_generic_encoder_decoder_forward_backward(kind):
    cfg, space = configuration(kind)
    latent = LatentConfig(deter_size=8, num_classes=2, stoch_size=3)
    net = NetworkConfig(hidden_size=16, action_dim=3, value_bins=21)
    encoder = build_observation_model(
        "encoder", obs_space=space, cfg=cfg.world, latent=latent, net=net, device=torch.device("cpu")
    )
    decoder = build_observation_model(
        "decoder", obs_space=space, cfg=cfg.world, latent=latent, net=net, device=torch.device("cpu")
    )
    assert isinstance(encoder, BaseEncoder)
    obs = torch.randint(11, (2, space.shape[0])).float()
    if kind == "glyph_local_clock":
        obs[:, -1] = 1
    embed = encoder(obs)
    assert embed.shape == (2, encoder.output_size)
    assert encoder.output_size == 16 + (144 if kind == "glyph_local_clock" else 0)
    h, z = torch.randn(2, 8), torch.randn(2, 2, 3)
    reconstruction = decoder(h, z)
    if isinstance(decoder, GlyphDecoder):
        assert decoder.predict(reconstruction).shape == (2, 15)
        loss = decoder.cell_loss(reconstruction, obs).mean()
    else:
        assert reconstruction.shape == obs.shape
        loss = reconstruction.square().mean()
    (loss + embed.square().mean()).backward()
    for module in (encoder, decoder):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        assert sum(float(value.abs().sum()) for value in gradients) > 0


def test_world_model_constructs_decoder_after_recurrent_modules(monkeypatch):
    calls = []
    factory = world_module.build_observation_model

    def observe_factory(component, **kwargs):
        calls.append(component)
        return factory(component, **kwargs)

    monkeypatch.setattr(world_module, "build_observation_model", observe_factory)

    def observe_constructor(name, constructor):
        def build(*args, **kwargs):
            calls.append(name)
            return constructor(*args, **kwargs)

        return build

    for name in ("RSSMCore", "Prior", "Posterior"):
        monkeypatch.setattr(world_module, name, observe_constructor(name, getattr(world_module, name)))
    cfg, space = configuration("glyph_local_clock")
    model = world_module.WorldModel(
        obs_space=space,
        dreamer_cfg=cfg,
        latent=LatentConfig(deter_size=8, num_classes=2, stoch_size=3),
        net=NetworkConfig(hidden_size=16, action_dim=3, value_bins=21),
    )
    assert calls == ["encoder", "RSSMCore", "Prior", "Posterior", "decoder"]
    assert model.embed_size == model.encoder.output_size == 160
    assert any(key.startswith("encoder.") for key in model.state_dict())
    assert any(key.startswith("decoder.") for key in model.state_dict())
    assert not any(key.startswith("observation_model.") for key in model.state_dict())


@pytest.mark.parametrize(
    "change, message",
    [
        ({"glyph_shape": (0, 5), "glyph_vocab_size": 11}, "positive"),
        ({"glyph_shape": (3, 5), "glyph_vocab_size": 1}, "vocabulary"),
        ({"glyph_local_branch": True}, "requires"),
        ({"obs_encoder": "cage2_host"}, "Host identities"),
        ({"obs_encoder": "invalid"}, "Unknown"),
    ],
)
def test_observation_validation_stays_at_factory_boundary(change, message):
    cfg, space = configuration("flat")
    for key, value in change.items():
        setattr(cfg.world, key, value)
    with pytest.raises(ValueError, match=message):
        build_observation_model(
            "encoder",
            obs_space=space,
            cfg=cfg.world,
            latent=LatentConfig(deter_size=8, num_classes=2, stoch_size=3),
            net=NetworkConfig(hidden_size=16, action_dim=3, value_bins=21),
            device=torch.device("cpu"),
        )
