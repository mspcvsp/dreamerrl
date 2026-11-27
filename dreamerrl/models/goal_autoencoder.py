"""Categorical bottleneck for detached RSSM goal features."""

import torch
from torch import nn
from torch.nn import functional as F


def goal_features(h, z):
    return torch.cat((h.float(), z.flatten(-2).float()), dim=-1)


def goal_similarity(goal, achieved):
    return F.cosine_similarity(goal.float(), achieved.float(), dim=-1).clamp(-1, 1)


def fit_goal_scale(fitting, floor):
    """Population moments of an already selected fitting set."""
    center = fitting.mean(0)
    spread = fitting.std(dim=0, unbiased=False)
    return center, spread.clamp_min(floor)


class GoalAutoencoder(nn.Module):
    def __init__(self, features, codes, hidden, *, standardized=False):
        super().__init__()
        self.codes = codes
        self.register_buffer("goal_center", torch.zeros(features) if standardized else None)
        self.register_buffer("goal_scale", torch.ones(features) if standardized else None)
        self.encoder = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, codes))
        self.decoder = nn.Sequential(nn.Linear(codes, hidden), nn.SiLU(), nn.Linear(hidden, features))

    @torch.no_grad()
    def set_normalization(self, center, scale):
        if self.goal_center is None or self.goal_scale is None:
            raise ValueError("Normalization requested for an absolute autoencoder")
        if (
            center.shape != self.goal_center.shape or scale.shape != self.goal_scale.shape
            or not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any()
        ):
            raise ValueError("Invalid goal normalization statistics")
        self.goal_center.copy_(center)
        self.goal_scale.copy_(scale)

    def to_training_coordinates(self, features):
        if self.goal_center is not None:
            return (features - self.goal_center) / self.goal_scale
        return features

    def to_native_coordinates(self, features):
        if self.goal_center is not None:
            return features * self.goal_scale + self.goal_center
        return features

    def decode_ids(self, ids):
        return self.to_native_coordinates(self.decoder(F.one_hot(ids, self.codes).float()))

    def forward(self, features):
        logits, decoded = self._reconstruct(self.to_training_coordinates(features))
        return logits, self.to_native_coordinates(decoded)

    def _reconstruct(self, features):
        logits = self.encoder(features)
        probabilities = logits.softmax(-1)
        hard = F.one_hot(logits.argmax(-1), self.codes).to(probabilities.dtype)
        code = hard + (probabilities - probabilities.detach())
        return logits, self.decoder(code)

    def reconstruction(self, features, weights):
        # This optimizer is independent of AMP actor/critic scaling.
        with torch.autocast(features.device.type, enabled=False):
            target = self.to_training_coordinates(features.detach().float())
            logits, decoded = self._reconstruct(target)
            denominator = weights.sum().clamp_min(1)
            mse = ((decoded - target).square().mean(-1) * weights).sum() / denominator
            occupancy = (logits.softmax(-1) * weights[:, None]).sum(0) / denominator
            entropy = -(occupancy * occupancy.clamp_min(1e-8).log()).sum()
            hard = (F.one_hot(logits.argmax(-1), self.codes) * weights[:, None]).sum(0)
            return mse - 0.01 * entropy, mse.detach(), (hard > 0).sum()
