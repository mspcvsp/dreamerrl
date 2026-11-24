"""Observation component construction; environment-specific choices stay here."""

import gymnasium as gym
import torch
from torch import nn

from dreamerrl.env.obs_utils import get_flat_obs_dim
from dreamerrl.utils.types import LatentConfig, NetworkConfig, WorldModelConfig

from .categorical_fields import CategoricalLossConfig
from .decoder import ObsDecoder
from .glyph_model import GlyphDecoder, GlyphEncoder
from .host_obs_encoder import HostObsEncoder
from .obs_encoder import build_obs_encoder


def build_observation_model(
    component: str,
    *,
    obs_space: gym.Space,
    cfg: WorldModelConfig,
    latent: LatentConfig,
    net: NetworkConfig,
    device: torch.device,
) -> nn.Module:
    """Build one component at its existing initialization point.

    Encoder and decoder are requested separately because WorldModel constructs
    its recurrent/prior/posterior modules between them. Building both eagerly
    would change random initialization, even when their shapes stay identical.
    """
    flat_dim = get_flat_obs_dim(obs_space)
    if component == "decoder":
        if cfg.glyph_shape is not None:
            cells = cfg.glyph_shape[0] * cfg.glyph_shape[1]
            return GlyphDecoder(
                latent=latent,
                net=net,
                shape=cfg.glyph_shape,
                vocab_size=cfg.glyph_vocab_size,
                clock=flat_dim == cells + 1,
            ).to(device)
        return ObsDecoder(latent=latent, net=net, output_dim=flat_dim).to(device)
    if component != "encoder":
        raise ValueError("Unknown observation component")

    encoder_kind = getattr(cfg, "obs_encoder", "flat")
    if encoder_kind not in ("flat", "cage2_host"):
        raise ValueError("Unknown observation encoder kind")
    if encoder_kind == "cage2_host":
        if cfg.glyph_shape is not None or flat_dim != 52:
            raise ValueError("Host encoder requires the unmodified 52-value CAGE2 vector")
        encoder = HostObsEncoder(cfg.cage2_host_order, cfg.cage2_subnet_ids, net.hidden_size).to(device)
    elif cfg.glyph_shape is not None:
        if len(cfg.glyph_shape) != 2 or min(cfg.glyph_shape) < 1:
            raise ValueError("Glyph grid dimensions must be positive (height, width)")
        cells = cfg.glyph_shape[0] * cfg.glyph_shape[1]
        if cfg.glyph_vocab_size < 2 or flat_dim not in (cells, cells + 1):
            raise ValueError("Glyph model requires a valid vocabulary and matching grid dimensions")
        if cfg.glyph_changed_weight < 1 or not torch.isfinite(torch.tensor(cfg.glyph_changed_weight)):
            raise ValueError("glyph_changed_weight must be finite and at least 1")
        if any(
            not isinstance(glyph, int) or not 0 <= glyph < cfg.glyph_vocab_size for glyph in cfg.glyph_foreground_ids
        ):
            raise ValueError("Foreground glyph IDs must belong to the configured vocabulary")
        CategoricalLossConfig(
            balance=cfg.glyph_loss_balance,
            salient_weight=cfg.glyph_changed_weight,
            salient_fraction=cfg.glyph_salient_fraction,
        )
        encoder = GlyphEncoder(
            cfg.glyph_shape,
            cfg.glyph_vocab_size,
            net.hidden_size,
            clock=flat_dim == cells + 1,
            local_branch=getattr(cfg, "glyph_local_branch", False),
            player_ids=getattr(cfg, "glyph_player_ids", ()),
        ).to(device)
    else:
        encoder = build_obs_encoder(obs_space, embed_dim=net.hidden_size).to(device)
    if getattr(cfg, "glyph_local_branch", False) and cfg.glyph_shape is None:
        raise ValueError("The local glyph branch requires the spatial glyph encoder")
    return encoder
