"""Categorical, spatial modeling of flattened MiniHack glyph grids."""

import math
from typing import TypedDict

import torch
import torch.nn as nn

from dreamerrl.models.base_obs_encoder import BaseEncoder
from dreamerrl.models.categorical_fields import CategoricalField, CategoricalFieldHead


class GlyphReconstruction(TypedDict):
    glyph_features: torch.Tensor
    clock: torch.Tensor | None


def glyph_ids(obs: torch.Tensor, cells: int, vocab_size: int) -> torch.Tensor:
    glyphs = obs[..., :cells]
    if not torch.isfinite(glyphs).all() or (glyphs != glyphs.round()).any():
        raise ValueError("Glyph observations must contain finite integer IDs")
    ids = glyphs.long()
    if (ids < 0).any() or (ids >= vocab_size).any():
        raise ValueError("Glyph ID outside configured vocabulary")
    return ids


class PlayerCenteredGlyphCrop(nn.Module):
    """Stateless 3x3 categorical crop; separate padding and missing-player tokens."""

    player_ids: torch.Tensor
    offsets: torch.Tensor

    def __init__(self, vocab_size, player_ids):
        super().__init__()
        if not player_ids or any(not isinstance(i, int) or not 0 <= i < vocab_size for i in player_ids):
            raise ValueError("Local glyph crops require valid player glyph IDs")
        self.padding_id, self.missing_id = vocab_size, vocab_size + 1
        self.register_buffer("player_ids", torch.tensor(player_ids, dtype=torch.long))
        self.register_buffer(
            "offsets", torch.tensor([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), persistent=False
        )

    def forward(self, grid):
        if grid.ndim != 3 or min(grid.shape[1:]) < 1:
            raise ValueError("Crop input must be (batch,height,width)")
        batch, height, width = grid.shape
        matches = torch.isin(grid, self.player_ids).flatten(1)
        located = matches.sum(-1) == 1
        center = matches.long().argmax(-1)
        x = center[:, None] % width + self.offsets[:, 0]
        y = center[:, None] // width + self.offsets[:, 1]
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        indices = (y * width + x).clamp(0, height * width - 1)
        crop = grid.reshape(batch, -1).gather(1, indices)
        crop = torch.where(valid, crop, self.padding_id)
        return torch.where(located[:, None], crop, self.missing_id)


class GlyphEncoder(BaseEncoder):
    def __init__(self, shape, vocab_size, embed_dim, clock=False, *, local_branch=False, player_ids=()):
        super().__init__()
        self.shape = tuple(shape)
        self.cells = math.prod(shape)
        self.vocab_size = vocab_size
        self.clock = clock
        self.embedding = nn.Embedding(vocab_size, 16)
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self._output_size = embed_dim + (9 * 16 if local_branch else 0)
        self.local_branch = local_branch
        coarse_cells = math.ceil(shape[0] / 4) * math.ceil(shape[1] / 4)
        self.project = nn.Linear(32 * coarse_cells + int(clock), embed_dim)
        if local_branch:
            self.local_crop = PlayerCenteredGlyphCrop(vocab_size, player_ids)
            self.local_embedding = nn.Embedding(vocab_size + 2, 16)

    @property
    def output_size(self):
        return self._output_size

    def forward(self, obs):
        if obs.ndim != 2 or obs.shape[1] != self.cells + int(self.clock):
            raise ValueError("Observation does not match glyph grid and optional clock")
        ids = glyph_ids(obs, self.cells, self.vocab_size)
        x = self.embedding(ids).reshape(-1, *self.shape, 16).permute(0, 3, 1, 2)
        x = self.spatial(x)
        if self.clock:
            x = torch.cat([x, obs[:, -1:]], -1)
        global_features = self.project(x)
        if self.local_branch:
            crop = self.local_crop(ids.reshape(-1, *self.shape))
            local_features = self.local_embedding(crop).flatten(1)
            return torch.cat([global_features, local_features], -1)
        return global_features


class GlyphDecoder(nn.Module):
    """Spatial categorical parameters; vocabulary logits are materialized in chunks.

    forward returns compact per-cell features and an optional clock prediction.
    Use predict to obtain glyph IDs. Checkpointed cross entropy avoids retaining
    a batch x cells x NLE-vocabulary tensor for backward.
    """

    def __init__(self, *, latent, net, shape, vocab_size, clock=False):
        super().__init__()
        self.shape = tuple(shape)
        self.cells = math.prod(shape)
        self.vocab_size = vocab_size
        self.clock = clock
        self.coarse = (math.ceil(shape[0] / 4), math.ceil(shape[1] / 4))
        self.project = nn.Linear(latent.deter_size + latent.z_dim, 32 * math.prod(self.coarse))
        self.spatial = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.classifier = CategoricalFieldHead(16, CategoricalField("glyph", vocab_size))
        self.clock_head = nn.Linear(latent.deter_size + latent.z_dim, 1) if clock else None

    def forward(self, h, z) -> GlyphReconstruction:
        state = torch.cat([h, z.flatten(1)], -1)
        x = self.project(state).reshape(-1, 32, *self.coarse)
        x = self.spatial(x)[:, :, : self.shape[0], : self.shape[1]]
        features = x.permute(0, 2, 3, 1).reshape(-1, self.cells, 16)
        return {"glyph_features": features, "clock": self.clock_head(state) if self.clock_head is not None else None}

    def cell_loss(self, decoded: GlyphReconstruction, target: torch.Tensor) -> torch.Tensor:
        ids = glyph_ids(target, self.cells, self.vocab_size)
        features = decoded["glyph_features"].reshape(*ids.shape, 16)
        return self.classifier.per_entity_loss(features, ids)

    @torch.no_grad()
    def predict(self, decoded: GlyphReconstruction) -> torch.Tensor:
        return self.classifier.predict(decoded["glyph_features"])
