"""Frozen MiniHack location probes; labels never enter the world-model input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import minihack  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from minihack import MiniHack

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


@dataclass
class FrozenAgent:
    world: WorldModel
    actor: Actor
    cfg: DreamerConfig
    checkpoint: str
    update: int
    critic: ValueHead | None = None


@dataclass
class LocationData:
    features: dict[str, torch.Tensor]
    positions: torch.Tensor
    level_seeds: torch.Tensor
    grid_shape: tuple[int, int]

    def summary(self) -> dict[str, Any]:
        return {
            "samples": len(self.positions),
            "level_seeds": self.level_seeds.unique().tolist(),
            "unique_cells": self.positions.unique(dim=0).shape[0],
            "samples_per_level": {
                str(int(seed)): int((self.level_seeds == seed).sum()) for seed in self.level_seeds.unique()
            },
        }


def load_frozen_agent(path: Path, *, sample_latents=False, load_critic=False) -> FrozenAgent:
    """Load trusted weights on CPU without restoring replay or live collectors."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") != 4:
        raise ValueError("Location probes require a v4 checkpoint")
    cfg = saved["config"]
    if not cfg.env.env_id.lower().startswith("minihack"):
        raise ValueError("Location probes require a MiniHack checkpoint")
    with gym.make(cfg.env.env_id) as env:
        obs, _ = env.reset(seed=0)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        action_dim = int(env.action_space.n)
        obs_dim = obs["glyphs"].size + int(cfg.env.finite_horizon)
    latent = LatentConfig(
        deter_size=cfg.world.deter_size, num_classes=cfg.world.num_classes, stoch_size=cfg.world.stoch_size
    )
    net = NetworkConfig(
        hidden_size=cfg.world.hidden_size,
        action_dim=action_dim,
        value_bins=cfg.world.value_bins,
        unimix=cfg.world.unimix,
    )
    world = WorldModel(
        obs_space=gym.spaces.Box(0, 10000, (obs_dim,), np.float32), dreamer_cfg=cfg, latent=latent, net=net
    )
    world.load_state_dict(saved["world"])
    world.eval().requires_grad_(False)
    world.posterior.deterministic_latent_for_tests = not sample_latents
    actor = Actor(
        latent=latent, net=NetworkConfig(hidden_size=cfg.ac.actor_hidden, action_dim=action_dim, unimix=cfg.ac.unimix)
    )
    actor.load_state_dict(saved["actor"])
    actor.eval().requires_grad_(False)
    critic = None
    if load_critic:
        if "critic" not in saved:
            raise ValueError("Critic diagnostics require a checkpoint with the online critic weights")
        critic = ValueHead(
            latent=latent,
            net=NetworkConfig(hidden_size=cfg.ac.critic_hidden, value_bins=cfg.world.value_bins),
        )
        critic.load_state_dict(saved["critic"])
        critic.eval().requires_grad_(False)
    return FrozenAgent(world, actor, cfg, str(path.resolve()), saved["completed_updates"], critic)


def validate_seed_split(train_seeds, test_seeds):
    if not train_seeds or not test_seeds or set(train_seeds) & set(test_seeds):
        raise ValueError("Probe train/test seeds must be nonempty and disjoint")
    if len(set(train_seeds)) != len(train_seeds) or len(set(test_seeds)) != len(test_seeds):
        raise ValueError("Probe seeds must be unique within each split")


@torch.no_grad()
def collect_locations(
    agent: FrozenAgent,
    seeds: list[int],
    *,
    max_steps=256,
    max_per_cell=4,
    policy="random",
    sampling_seed=42,
    initial: FrozenAgent | None = None,
) -> LocationData:
    """One episode per seed; observe obs[t] with its incoming action before labelling it.

    Limit stored duplicates per position per episode, while still filtering every
    step. The optional initial model observes exactly the same trajectory.
    """
    if max_steps < 1 or max_per_cell < 1 or policy not in ("random", "sampled", "greedy"):
        raise ValueError("Invalid collection settings")
    if initial is not None and (initial.cfg.world != agent.cfg.world or initial.cfg.env != agent.cfg.env):
        raise ValueError("Initial and trained world/environment configurations must match")
    features: dict[str, list[torch.Tensor]] = {name: [] for name in ("h", "z", "hz", "encoder", "posterior_probs")}
    if initial is not None:
        features["initial_hz"] = []
    positions, labels = [], []
    with torch.random.fork_rng(devices=[]), gym.make(agent.cfg.env.env_id) as env:
        for seed in seeds:
            torch.default_generator.manual_seed(sampling_seed + seed)
            cast(MiniHack, env.unwrapped).seed(core=seed, disp=seed, reseed=False, lgen=seed)
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(sampling_seed + seed)
            state = agent.world.init_state(1)
            initial_state = initial.world.init_state(1) if initial is not None else None
            action_dim = agent.world.net_cfg.action_dim
            assert action_dim is not None
            action = torch.zeros(1, action_dim)
            counts: dict[tuple[int, int], int] = {}
            grid_shape = tuple(int(value) for value in obs["glyphs"].shape)
            for t in range(min(max_steps, agent.cfg.env.max_episode_steps)):
                # Only glyphs (and the already-configured clock) are agent inputs.
                observation = torch.as_tensor(np.asarray(obs["glyphs"]).reshape(1, -1).copy(), dtype=torch.float32)
                if agent.cfg.env.finite_horizon:
                    observation = torch.cat(
                        [observation, torch.tensor([[1 - t / agent.cfg.env.max_episode_steps]])], -1
                    )
                embed = agent.world.encoder(observation)
                first = torch.tensor([t == 0])
                state = agent.world.filter_step(state, embed, action, first)
                if initial is not None:
                    with torch.random.fork_rng(devices=[]):
                        initial_state = initial.world.filter_step(
                            initial_state, initial.world.encoder(observation), action, first
                        )
                x, y = (int(value) for value in obs["blstats"][:2])
                if not (0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]):
                    raise ValueError("Position label lies outside the glyph grid")
                cell = (x, y)
                if counts.get(cell, 0) < max_per_cell:
                    counts[cell] = counts.get(cell, 0) + 1
                    h, z = state.h[0], state.z.flatten(1)[0]
                    stats = state.post_stats
                    assert stats is not None
                    for name, value in (
                        ("h", h),
                        ("z", z),
                        ("hz", torch.cat([h, z])),
                        ("encoder", embed[0]),
                        ("posterior_probs", stats["probs"].flatten(1)[0]),
                    ):
                        features[name].append(value.detach().cpu())
                    if initial_state is not None:
                        features["initial_hz"].append(
                            torch.cat([initial_state.h, initial_state.z.flatten(1)], -1)[0].cpu()
                        )
                    positions.append(cell)
                    labels.append(seed)
                if policy == "random":
                    action_id = int(env.action_space.sample())
                else:
                    selected, _ = agent.actor.act(state, deterministic=policy == "greedy")
                    action_id = int(selected.item())
                obs, _, terminated, truncated, _ = env.step(action_id)
                if terminated or truncated:
                    break
                action = F.one_hot(torch.tensor([action_id]), action_dim).float()
    if not positions:
        raise ValueError("No labelled probe samples collected")
    return LocationData(
        {name: torch.stack(values) for name, values in features.items()},
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(labels),
        (grid_shape[0], grid_shape[1]),
    )


@dataclass
class FittedProbe:
    model: nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    target_mean: torch.Tensor
    scale: torch.Tensor

    @torch.no_grad()
    def predict(self, features):
        return (self.model((features - self.mean) / self.std) + self.target_mean) * self.scale


def fit_probe(features, positions, grid_shape, *, kind="linear", epochs=100, seed=42) -> FittedProbe:
    """Only training data is accepted here, including for normalization."""
    if kind not in ("linear", "mlp") or epochs < 1 or len(features) < 2:
        raise ValueError("Probe fitting needs at least two samples and valid kind/epochs")
    features, positions = features.detach().float(), positions.detach().float()
    mean = features.mean(0)
    std = features.std(0, unbiased=False)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    scale = torch.tensor([max(1, grid_shape[1] - 1), max(1, grid_shape[0] - 1)], dtype=torch.float32)
    target_mean = (positions / scale).mean(0)
    x, y = (features - mean) / std, positions / scale - target_mean
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = (
            nn.Linear(features.shape[1], 2)
            if kind == "linear"
            else nn.Sequential(nn.Linear(features.shape[1], 64), nn.SiLU(), nn.Linear(64, 2))
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            for indices in torch.randperm(len(x), generator=generator).split(256):
                optimizer.zero_grad()
                F.mse_loss(model(x[indices]), y[indices]).backward()
                optimizer.step()
    return FittedProbe(model.eval().requires_grad_(False), mean, std, target_mean, scale)


def position_scores(prediction, data: LocationData) -> dict[str, Any]:
    """Unclipped regression error; rounded cells are also left unclipped."""
    error = prediction - data.positions
    manhattan = error.abs().sum(-1)
    rounded_distance = (prediction.round() - data.positions).abs().sum(-1)
    variance = (data.positions - data.positions.mean(0)).square().sum()
    return {
        "mean_manhattan_cells": float(manhattan.mean()),
        "mae_xy_cells": error.abs().mean(0).tolist(),
        "exact_cell_accuracy": float((rounded_distance == 0).float().mean()),
        "within_one_cell_accuracy": float((rounded_distance <= 1).float().mean()),
        "r2": float(1 - error.square().sum() / variance) if variance > 0 else None,
        "macro_level_manhattan_cells": float(
            torch.stack([manhattan[data.level_seeds == seed].mean() for seed in data.level_seeds.unique()]).mean()
        ),
    }


def evaluate_probes(train: LocationData, test: LocationData, *, epochs=100, seed=42) -> dict[str, Any]:
    validate_seed_split(train.level_seeds.unique().tolist(), test.level_seeds.unique().tolist())
    if train.grid_shape != test.grid_shape or train.features.keys() != test.features.keys():
        raise ValueError("Probe splits must have matching grid/features")
    constant = train.positions.mean(0).expand_as(test.positions)
    results: dict[str, Any] = {"constant": position_scores(constant, test)}
    for name in train.features:
        for kind in ("linear", "mlp"):
            probe = fit_probe(
                train.features[name], train.positions, train.grid_shape, kind=kind, epochs=epochs, seed=seed
            )
            results[f"{name}_{kind}"] = {
                "train": position_scores(probe.predict(train.features[name]), train),
                "test": position_scores(probe.predict(test.features[name]), test),
            }
    permutation = torch.randperm(len(train.positions), generator=torch.Generator().manual_seed(seed))
    for kind in ("linear", "mlp"):
        shuffled = fit_probe(
            train.features["hz"], train.positions[permutation], train.grid_shape, kind=kind, epochs=epochs, seed=seed
        )
        results[f"shuffled_hz_{kind}"] = position_scores(shuffled.predict(test.features["hz"]), test)
    return results
