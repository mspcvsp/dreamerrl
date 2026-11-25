"""Categorical bottleneck for detached RSSM goal features."""

import torch
from torch import nn
from torch.nn import functional as F


def goal_features(h, z):
    return torch.cat((h.float(), z.flatten(-2).float()), dim=-1)


def goal_similarity(goal, achieved):
    return F.cosine_similarity(goal.float(), achieved.float(), dim=-1).clamp(-1, 1)


class GoalAutoencoder(nn.Module):
    def __init__(self, features, codes, hidden):
        super().__init__()
        self.codes = codes
        self.encoder = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, codes))
        self.decoder = nn.Sequential(nn.Linear(codes, hidden), nn.SiLU(), nn.Linear(hidden, features))

    def decode_ids(self, ids):
        return self.decoder(F.one_hot(ids, self.codes).float())

    def forward(self, features):
        logits = self.encoder(features)
        probabilities = logits.softmax(-1)
        hard = F.one_hot(logits.argmax(-1), self.codes).to(probabilities.dtype)
        code = hard + (probabilities - probabilities.detach())
        return logits, self.decoder(code)

    def reconstruction(self, features, weights):
        # This optimizer is independent of AMP actor/critic scaling.
        with torch.autocast(features.device.type, enabled=False):
            target = features.detach().float()
            logits, decoded = self(target)
            denominator = weights.sum().clamp_min(1)
            mse = ((decoded - target).square().mean(-1) * weights).sum() / denominator
            occupancy = (logits.softmax(-1) * weights[:, None]).sum(0) / denominator
            entropy = -(occupancy * occupancy.clamp_min(1e-8).log()).sum()
            hard = (F.one_hot(logits.argmax(-1), self.codes) * weights[:, None]).sum(0)
            return mse - 0.01 * entropy, mse.detach(), (hard > 0).sum()
