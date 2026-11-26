"""Causal readout probes over frozen, episode-grouped representation histories."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def project_fit_scaled(frames, fit, width, seed):
    """Shared fit-only standardization and a fixed projection for all architectures."""
    x = torch.as_tensor(frames, dtype=torch.float32)
    if x.ndim != 2 or not np.any(fit) or width < 1:
        raise ValueError("Expected feature rows and a nonempty fit fold")
    fit_x = x[fit]
    x = (x - fit_x.mean(0)) / fit_x.std(0, unbiased=False).clamp_min(1e-5)
    if x.shape[1] > width:
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randint(2, (x.shape[1], width), generator=generator).float().mul_(2).sub_(1)
        x = x @ (projection / width**.5)
    return x


def past_windows(frames, episode_ids, timestep, length):
    """Left-aligned prefixes ending at t; excludes successors and other episodes."""
    frames = torch.as_tensor(frames, dtype=torch.float32)
    episode_ids, timestep = np.asarray(episode_ids), np.asarray(timestep)
    if frames.ndim != 2 or length < 1 or episode_ids.shape != timestep.shape or timestep.shape != (len(frames),):
        raise ValueError("Invalid frame or episode layout")
    windows = frames.new_zeros((len(frames), length, frames.shape[-1]))
    valid = torch.zeros((len(frames), length), dtype=torch.bool)
    for episode in np.unique(episode_ids):
        rows = np.flatnonzero(episode_ids == episode)
        rows = rows[np.argsort(timestep[rows])]
        if not np.array_equal(timestep[rows], np.arange(len(rows))):
            raise ValueError("Each episode must contain one frame per ordered timestep")
        finish = np.arange(len(rows))
        start = np.maximum(finish - length + 1, 0)
        access = start[:, None] + np.arange(length)
        in_window = access <= finish[:, None]
        access = np.minimum(access, finish[:, None])
        mask = torch.from_numpy(in_window)
        windows[rows] = frames[rows[access]].masked_fill(~mask[..., None], 0)
        valid[rows] = mask
    return windows, valid


class HistoryProbe(nn.Module):
    """Fixed pooled history readouts with identical feature and label interfaces."""

    def __init__(self, kind, features, steps, width):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(features, 1)
        elif kind in ("current_mlp", "fixed_mlp"):
            input_width = features * (steps if kind == "fixed_mlp" else 1)
            self.net = nn.Sequential(nn.Linear(input_width, width), nn.SiLU(), nn.Linear(width, 1))
        elif kind == "gru":
            self.net = nn.GRU(features, width, batch_first=True)
            self.head = nn.Linear(width, 1)
        elif kind == "transformer":
            self.input = nn.Linear(features, width)
            self.pos = nn.Parameter(torch.zeros(steps, width))
            self.net = nn.TransformerEncoderLayer(width, 2, dim_feedforward=2*width, dropout=0,
                                                  activation="gelu", batch_first=True)
            self.head = nn.Linear(width, 1)
        else:
            raise ValueError(f"Unknown temporal probe {kind}")

    def forward(self, windows, valid):
        last = (valid.sum(-1) - 1).clamp_min(0)
        row = torch.arange(len(windows), device=windows.device)
        if self.kind in ("linear", "current_mlp"):
            return self.net(windows[row, last]).squeeze(-1)
        windows = windows.masked_fill(~valid[..., None], 0)
        if self.kind == "fixed_mlp":
            return self.net(windows.flatten(1)).squeeze(-1)
        if self.kind == "gru":
            sequence, _ = self.net(windows)
        else:
            causal = torch.ones((windows.shape[1], windows.shape[1]), device=windows.device, dtype=torch.bool).triu(1)
            sequence = self.net(self.input(windows) + self.pos, src_mask=causal, src_key_padding_mask=~valid,
                                is_causal=True)
        return self.head(sequence[row, last]).squeeze(-1)


def budgeted_probe(kind, features, steps, budget):
    """Use the largest supported width within the common trainable parameter cap."""
    factors = {
        "current_mlp": (0, features + 2),
        "fixed_mlp": (0, features * steps + 2),
        "gru": (3, 3*features + 7),
        "transformer": (8, features + steps + 13),
    }
    if kind != "linear" and kind not in factors:
        raise ValueError(f"Unknown temporal probe {kind}")
    width = 1
    if kind != "linear":
        quadratic, linear = factors[kind]
        quantum = 2 if kind == "transformer" else 1
        width = max((w for w in range(quantum, budget + 1, quantum)
                     if quadratic*w*w + linear*w + 1 <= budget), default=0)
        if width == 0:
            raise ValueError("Parameter budget cannot support this architecture")
    model = HistoryProbe(kind, features, steps, width)
    if parameter_count(model) > budget:
        raise ValueError("Parameter budget cannot support this architecture")
    return model


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass(frozen=True)
class AttentionEvidence:
    """Evidence from a separate early-stage study on the intended target."""

    fit_episodes: int
    validation_episodes: int
    test_episodes: int
    target_adequately_supported: bool
    matched_history_increment_ci95: tuple[float, float] | None

    @property
    def permits_fit(self):
        return (self.fit_episodes >= 60 and min(self.validation_episodes, self.test_episodes) >= 20
                and self.target_adequately_supported and self.matched_history_increment_ci95 is not None
                and self.matched_history_increment_ci95[0] > 0)


def fit_history_probe(kind, windows, valid, targets, split, *, seed, budget, learning_rates, epochs,
                      batch_size=256, weight_decay=1e-4, task="regression", attention_evidence=None):
    """Fit only the offline probe and select learning rate/epoch on validation."""
    if kind == "transformer" and (attention_evidence is None or not attention_evidence.permits_fit):
        raise ValueError("Transformer fitting requires independent temporal evidence and episode coverage")
    y = torch.as_tensor(targets, dtype=torch.float32)
    if y.shape != (len(windows),) or valid.shape != windows.shape[:2] or not valid[:, 0].all():
        raise ValueError("Expected aligned labels and nonempty causal prefixes")
    if not np.all(sum(np.asarray(mask, dtype=int) for mask in split.values()) == 1):
        raise ValueError("Probe folds must partition rows")
    fit = torch.from_numpy(np.flatnonzero(split["fit"]))
    val = torch.from_numpy(np.flatnonzero(split["validation"]))
    if not len(fit) or not len(val) or not epochs or not learning_rates:
        raise ValueError("Provide fit and validation rows and a nonempty training schedule")
    if task == "binary":
        if not torch.all((y == 0) | (y == 1)) or not 0 < y[fit].mean() < 1:
            raise ValueError("Binary targets require labels 0 and 1 in fit")
        mean, scale = 0., 1.
        prevalence = y[fit].mean()
        loss = lambda prediction, label: F.binary_cross_entropy_with_logits(
            prediction, label, weight=torch.where(label.bool(), .5/prevalence, .5/(1-prevalence)))
    elif task == "regression":
        mean, scale = y[fit].mean(), y[fit].std(unbiased=False).clamp_min(1e-6)
        loss = F.mse_loss
    else:
        raise ValueError(f"Unknown target task {task}")
    target = (y - mean) / scale
    best = None
    with torch.random.fork_rng(devices=[]):
        for lr in learning_rates:
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            model = budgeted_probe(kind, windows.shape[-1], windows.shape[1], budget)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            for epoch in range(1, epochs + 1):
                model.train()
                for rows in fit[torch.randperm(len(fit), generator=generator)].split(batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss(model(windows[rows], valid[rows]), target[rows]).backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    value = float(loss(model(windows[val], valid[val]), target[val]))
                if best is None or value < best["validation_loss"]:
                    best = dict(validation_loss=value, learning_rate=lr, epoch=epoch, model=deepcopy(model))
    model = best.pop("model")
    with torch.inference_mode():
        parts = [model(x, m) * scale + mean for x, m in zip(windows.split(batch_size), valid.split(batch_size), strict=True)]
    return dict(best, predictions=torch.cat(parts).numpy(), parameters=parameter_count(model))
