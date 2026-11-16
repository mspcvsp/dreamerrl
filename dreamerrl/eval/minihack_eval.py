"""Offline, fixed-seed MiniHack evaluation without changing training state."""

from __future__ import annotations

import math
import random
from statistics import mean, stdev
from typing import TypedDict, cast

import gymnasium as gym
import minihack  # noqa: F401
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from minihack import MiniHack

from dreamerrl.env.minihack.minihack_utils import extract_glyphs


@torch.no_grad()
def evaluate_minihack(
    cfg,
    world,
    actor,
    episode_seeds,
    random_policy=False,
    should_stop=None,
    *,
    sample_actions=False,
    sample_latents=False,
    sampling_seed=0,
):
    """Whole episode returns on held-out seeds, including the final reward.

    Use a separate, single environment to avoid vector autoreset ambiguities.
    Evaluation restores training modes and torch RNG state. Latent modes and
    greedy actions give a repeatable policy evaluation; random actions use the
    independently seeded environment action space. Action and latent sampling can
    be enabled independently. Each episode gets a fixed Torch sampling seed;
    caller RNG states are restored even when evaluation is interrupted.
    """
    device = next(world.parameters()).device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    env = None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    world_training, actor_training = world.training, actor.training
    deterministic_posterior = world.posterior.deterministic_latent_for_tests
    returns = []
    with torch.random.fork_rng(devices=devices):
        try:
            env = TimeLimit(gym.make(cfg.env.env_id), max_episode_steps=cfg.env.max_episode_steps)
            world.eval()
            actor.eval()
            world.posterior.deterministic_latent_for_tests = not sample_latents
            for seed in episode_seeds:
                if should_stop is not None and should_stop():
                    raise InterruptedError("Evaluation stop requested")
                torch.default_generator.manual_seed(sampling_seed + seed)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(sampling_seed + seed)
                # Gymnasium reset seeds its own RNG, not NetHack's separate RNGs.
                cast(MiniHack, env.unwrapped).seed(core=seed, disp=seed, reseed=False, lgen=seed)
                obs, _ = env.reset(seed=seed)
                env.action_space.seed(seed)
                state = world.init_state(1)
                action_dim = world.net_cfg.action_dim
                if action_dim is None:
                    raise ValueError("Evaluation requires a discrete action dimension")
                action = torch.zeros(1, action_dim, device=device)
                total = 0.0
                for t in range(cfg.env.max_episode_steps):
                    if should_stop is not None and should_stop():
                        raise InterruptedError("Evaluation stop requested")
                    if random_policy:
                        action_id = env.action_space.sample()
                    else:
                        observation = torch.as_tensor(
                            extract_glyphs(obs).reshape(1, -1), dtype=torch.float32, device=device
                        )
                        if cfg.env.finite_horizon:
                            observation = torch.cat(
                                [
                                    observation,
                                    torch.tensor([[1 - t / cfg.env.max_episode_steps]], device=device),
                                ],
                                -1,
                            )
                        state = world.observe_step(
                            state,
                            observation,
                            action,
                            is_first=torch.tensor([t == 0], device=device),
                        )["post"]
                        selected, _ = actor.act(state, deterministic=not sample_actions)
                        action_id = int(selected.item())
                    obs, reward, terminated, truncated, _ = env.step(action_id)
                    total += float(reward)
                    if terminated or truncated:
                        break
                    action = torch.nn.functional.one_hot(
                        torch.tensor([action_id], device=device),
                        action_dim,
                    ).float()
                returns.append(total)
        finally:
            if env is not None:
                env.close()
            world.posterior.deterministic_latent_for_tests = deterministic_posterior
            world.train(world_training)
            actor.train(actor_training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
    return returns


class ReturnComparison(TypedDict):
    mean_difference: float
    ci95_low: float | None
    ci95_high: float | None


def compare_returns(trained: list[float], baseline: list[float]) -> ReturnComparison:
    """Paired normal-approximation interval; descriptive, not a sequential test."""
    if not trained or len(trained) != len(baseline):
        raise ValueError("Paired evaluation requires equal, nonempty episode lists")
    differences = [a - b for a, b in zip(trained, baseline)]
    delta = mean(differences)
    half_width = 1.96 * stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else None
    return {
        "mean_difference": delta,
        "ci95_low": None if half_width is None else delta - half_width,
        "ci95_high": None if half_width is None else delta + half_width,
    }
