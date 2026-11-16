"""Categorical, spatial modeling of flattened MiniHack glyph grids."""

import math
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dreamerrl.models.base_obs_encoder import BaseEncoder


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


class GlyphEncoder(BaseEncoder):
    def __init__(self, shape, vocab_size, embed_dim, clock=False):
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
        self._output_size = embed_dim
        coarse_cells = math.ceil(shape[0] / 4) * math.ceil(shape[1] / 4)
        self.project = nn.Linear(32 * coarse_cells + int(clock), embed_dim)

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
        return self.project(x)


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
        self.classifier = nn.Linear(16, vocab_size)
        self.clock_head = nn.Linear(latent.deter_size + latent.z_dim, 1) if clock else None

    def forward(self, h, z) -> GlyphReconstruction:
        state = torch.cat([h, z.flatten(1)], -1)
        x = self.project(state).reshape(-1, 32, *self.coarse)
        x = self.spatial(x)[:, :, : self.shape[0], : self.shape[1]]
        features = x.permute(0, 2, 3, 1).reshape(-1, self.cells, 16)
        return {"glyph_features": features, "clock": self.clock_head(state) if self.clock_head is not None else None}

    def cell_loss(self, decoded: GlyphReconstruction, target: torch.Tensor) -> torch.Tensor:
        features = decoded["glyph_features"].reshape(-1, 16)
        ids = glyph_ids(target, self.cells, self.vocab_size).reshape(-1)

        def cross_entropy(x, y):
            return F.cross_entropy(self.classifier(x).float(), y, reduction="none")

        losses = []
        for x, y in zip(features.split(1024), ids.split(1024)):
            if torch.is_grad_enabled():
                losses.append(checkpoint(cross_entropy, x, y, use_reentrant=False))
            else:
                losses.append(cross_entropy(x, y))
        return torch.cat(losses).reshape(*target.shape[:-1], self.cells)

    @torch.no_grad()
    def predict(self, decoded: GlyphReconstruction) -> torch.Tensor:
        features = decoded["glyph_features"]
        ids = torch.cat([self.classifier(x).argmax(-1) for x in features.reshape(-1, 16).split(1024)])
        return ids.reshape(*features.shape[:-1])
