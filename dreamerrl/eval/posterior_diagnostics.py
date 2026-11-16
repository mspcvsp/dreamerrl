"""Frozen posterior usage, conditional sensitivity, and decoder-ablation diagnostics."""

from __future__ import annotations

import math
from typing import cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import TimeLimit
from minihack import MiniHack

from dreamerrl.eval.location_probe import FrozenAgent
from dreamerrl.models.categorical_kl import categorical_kl
from dreamerrl.models.glyph_model import GlyphDecoder
from dreamerrl.utils.transforms import symlog


def entropy(probs):
    return -(probs * probs.clamp_min(1e-8).log()).sum(-1)


def js_divergence(q, other):
    mean = (q + other) / 2
    return (categorical_kl(q, mean) + categorical_kl(other, mean)) / 2


def posterior_usage(q, prior, z, free_nats):
    """Descriptive metrics, not a threshold-based diagnosis of collapse."""
    N, K, C = q.shape
    raw_kl = categorical_kl(q, prior).sum(-1)
    modes, samples = q.argmax(-1), z.argmax(-1)
    factor_rows = []
    for factor in range(K):
        counts = torch.bincount(modes[:, factor], minlength=C)
        factor_rows.append(
            {
                "factor": factor,
                "normalized_entropy": float(entropy(q[:, factor]).mean() / math.log(C)),
                "mode_categories_used": int((counts > 0).sum()),
                "mode_majority_fraction": float(counts.max() / N),
                "sample_categories_used": int(samples[:, factor].unique().numel()),
            }
        )
    return {
        "observations": N,
        "raw_kl_total_mean_nats": float(raw_kl.mean()),
        "raw_kl_total_median_nats": float(raw_kl.median()),
        "raw_kl_total_p95_nats": float(torch.quantile(raw_kl, 0.95)),
        "fraction_below_free_nats": float((raw_kl < free_nats).float().mean()),
        "floored_kl_mean_nats": float(raw_kl.clamp_min(free_nats).mean()),
        "posterior_normalized_entropy": float(entropy(q).mean() / math.log(C)),
        "prior_normalized_entropy": float(entropy(prior).mean() / math.log(C)),
        "posterior_top_probability_mean": float(q.max(-1).values.mean()),
        "expected_independent_sample_agreement": float(q.square().sum(-1).mean()),
        "marginal_entropy_gap_total_nats": float((entropy(q.mean(0)) - entropy(q).mean(0)).sum()),
        "mean_mode_categories_used": sum(row["mode_categories_used"] for row in factor_rows) / K,
        "mean_mode_majority_fraction": sum(row["mode_majority_fraction"] for row in factor_rows) / K,
        "per_factor": factor_rows,
    }


@torch.no_grad()
def collect_trace(agent: FrozenAgent, seeds, max_steps=128, sampling_seed=42):
    observations, actions, firsts, positions, rewards, ranges = [], [], [], [], [], []
    with gym.make(agent.cfg.env.env_id) as base:
        assert isinstance(base.action_space, gym.spaces.Discrete)
        action_dim = int(base.action_space.n)
    with TimeLimit(gym.make(agent.cfg.env.env_id), max_episode_steps=agent.cfg.env.max_episode_steps) as env:
        for seed in seeds:
            cast(MiniHack, env.unwrapped).seed(core=seed, disp=seed, reseed=False, lgen=seed)
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(sampling_seed + seed)
            action = torch.zeros(action_dim)
            reward, done = 0.0, False
            start = len(observations)
            for t in range(min(max_steps, agent.cfg.env.max_episode_steps) + 1):
                observation = torch.tensor(np.asarray(obs["glyphs"]).reshape(-1).copy(), dtype=torch.float32)
                if agent.cfg.env.finite_horizon:
                    observation = torch.cat([observation, torch.tensor([1 - t / agent.cfg.env.max_episode_steps])])
                observations.append(observation)
                actions.append(action)
                firsts.append(t == 0)
                positions.append(tuple(int(v) for v in obs["blstats"][:2]))
                rewards.append(float(reward))
                if done or t == min(max_steps, agent.cfg.env.max_episode_steps):
                    break
                selected = int(env.action_space.sample())
                obs, reward, terminated, truncated, _ = env.step(selected)
                done = terminated or truncated
                action = F.one_hot(torch.tensor(selected), action_dim).float()
            ranges.append((start, len(observations)))
    return {
        "obs": torch.stack(observations),
        "action": torch.stack(actions),
        "is_first": torch.tensor(firsts),
        "positions": torch.tensor(positions),
        "reward": torch.tensor(rewards),
        "ranges": ranges,
    }


@torch.no_grad()
def filter_trace(agent: FrozenAgent, trace, *, sampled=False, seed=42):
    world = agent.world
    old_mode = world.posterior.deterministic_latent_for_tests
    world.posterior.deterministic_latent_for_tests = not sampled
    hs, zs, qs, ps = [], [], [], []
    try:
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            embeds = torch.cat([world.encoder(chunk) for chunk in trace["obs"].split(256)])
            for start, end in trace["ranges"]:
                state = world.init_state(1)
                for t in range(start, end):
                    state = world.filter_step(
                        state, embeds[t : t + 1], trace["action"][t : t + 1], trace["is_first"][t : t + 1]
                    )
                    assert state.post_stats is not None
                    with torch.random.fork_rng(devices=[]):
                        prior = world.prior(state.h)["probs"]
                    hs.append(state.h[0])
                    zs.append(state.z[0])
                    qs.append(state.post_stats["probs"][0])
                    ps.append(prior[0])
        return {
            "h": torch.stack(hs),
            "z": torch.stack(zs),
            "q": torch.stack(qs),
            "prior": torch.stack(ps),
            "embed": embeds,
        }
    finally:
        world.posterior.deterministic_latent_for_tests = old_mode


@torch.no_grad()
def conditional_sensitivity(world, states, trace, seed=42):
    N = len(states["h"])
    # Pair every observation with a different frame, using a seeded circular shift.
    shift = int(torch.randint(1, N, (1,), generator=torch.Generator().manual_seed(seed))) if N > 1 else 0
    permutation = torch.arange(N).roll(shift)
    with torch.random.fork_rng(devices=[]):
        swapped_obs = world.posterior(states["h"], states["embed"][permutation])["probs"]
        swapped_history = world.posterior(states["h"][permutation], states["embed"])["probs"]
    q = states["q"]
    result = {
        "observation_swap_js_total_nats": float(js_divergence(q, swapped_obs).sum(-1).mean()),
        "observation_swap_tv_per_factor": float((q - swapped_obs).abs().sum(-1).mean() / 2),
        "observation_swap_mode_change_fraction": float((q.argmax(-1) != swapped_obs.argmax(-1)).float().mean()),
        "history_swap_js_total_nats": float(js_divergence(q, swapped_history).sum(-1).mean()),
        "history_swap_mode_change_fraction": float((q.argmax(-1) != swapped_history.argmax(-1)).float().mean()),
        "swapped_position_differs_fraction": float(
            (trace["positions"] != trace["positions"][permutation]).any(-1).float().mean()
        ),
        "swapped_observation_differs_fraction": float(
            (trace["obs"] != trace["obs"][permutation]).any(-1).float().mean()
        ),
    }
    # A stricter control keeps the level fixed and requires a changed position.
    paired = torch.arange(N)
    valid = torch.zeros(N, dtype=torch.bool)
    for start, end in trace["ranges"]:
        xy = trace["positions"][start:end]
        differs = (xy[:, None] != xy[None, :]).any(-1)
        has_other = differs.any(-1)
        paired[start:end] = start + differs.long().argmax(-1)
        valid[start:end] = has_other
    result["same_level_position_swap_pairs"] = int(valid.sum())
    if valid.any():
        with torch.random.fork_rng(devices=[]):
            other = world.posterior(states["h"][valid], states["embed"][paired[valid]])["probs"]
        result["same_level_position_swap_js_total_nats"] = float(js_divergence(q[valid], other).sum(-1).mean())
        result["same_level_position_swap_mode_change_fraction"] = float(
            (q[valid].argmax(-1) != other.argmax(-1)).float().mean()
        )
    return result


@torch.no_grad()
def reconstruction_ablations(world, states, trace, *, frames=64, cells=64, seed=42):
    """Uniform-frame/cell CE is an estimate; player-cell CE exposes static-map dominance."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(states["h"]), generator=generator)[:frames]
    h, z, obs = states["h"][indices], states["z"][indices], trace["obs"][indices]
    permutation = torch.randperm(len(indices), generator=generator)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        prior_z = world.prior(h)["z"]
    variants = {
        "posterior": (h, z),
        "shuffled_z": (h, z[permutation]),
        "zero_z": (h, torch.zeros_like(z)),
        "prior_z": (h, prior_z),
        "shuffled_h": (h[permutation], z),
    }
    result = {}
    if isinstance(world.decoder, GlyphDecoder):
        decoder = world.decoder
        selected_cells = torch.randperm(decoder.cells, generator=generator)[:cells]
        xy = trace["positions"][indices]
        player_cells = xy[:, 1] * decoder.shape[1] + xy[:, 0]
        for name, (hh, zz) in variants.items():
            decoded = decoder(hh, zz)["glyph_features"]
            random_features = decoded[:, selected_cells].reshape(-1, 16)
            random_targets = obs[:, selected_cells].long().flatten()
            random_ce = F.cross_entropy(decoder.classifier(random_features), random_targets)
            player_features = decoded[torch.arange(len(indices)), player_cells]
            player_targets = obs[torch.arange(len(indices)), player_cells].long()
            player_logits = decoder.classifier(player_features)
            result[name] = {
                "sampled_grid_ce": float(random_ce),
                "player_cell_ce": float(F.cross_entropy(player_logits, player_targets)),
                "player_cell_accuracy": float((player_logits.argmax(-1) == player_targets).float().mean()),
            }
    else:
        for name, (hh, zz) in variants.items():
            result[name] = {"observation_symlog_mse": float(F.mse_loss(world.decoder(hh, zz), symlog(obs)))}
    return {"frames": len(indices), "uniform_cells_per_frame": min(cells, obs.shape[1]), "variants": result}
