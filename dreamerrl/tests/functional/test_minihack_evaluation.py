"""Evaluation must count whole episodes and leave training state untouched."""

from dataclasses import replace
from enum import IntEnum
from typing import cast

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.eval.corridor_diagnostics import CorridorEpisode
from dreamerrl.eval.minihack_eval import EpisodeDiagnostics, evaluate_minihack
from dreamerrl.models.actor import Actor
from dreamerrl.models.world_model import WorldModel
from dreamerrl.tests.manual.utils import minihack_config


class RewardSentinel(gym.Env):
    action_space = gym.spaces.Discrete(5)
    observation_space = gym.spaces.Dict({"glyphs": gym.spaces.Box(0, 100, (2, 4), np.float32)})

    def __init__(self, terminal):
        self.terminal = terminal
        self.seeds = []
        self.closed = False
        self.nle_seeds = []

    def seed(self, *, core, disp, reseed, lgen):
        self.nle_seeds.append((core, disp, reseed, lgen))

    def reset(self, *, seed=None, options=None):
        assert self.nle_seeds[-1] == (seed, seed, False, seed)
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.t = 0
        return {"glyphs": np.zeros((2, 4), dtype=np.float32)}, {}

    def step(self, action):
        self.t += 1
        return (
            {"glyphs": np.zeros((2, 4), dtype=np.float32)},
            float(self.t + 1),
            self.terminal and self.t == 2,
            not self.terminal and self.t == 2,
            {},
        )

    def close(self):
        self.closed = True


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("random_policy", [False, True])
def test_held_out_evaluation_counts_final_reward_and_restores_state(
    world_model,
    tmp_path,
    monkeypatch,
    terminal,
    random_policy,
    finite,
):
    env = RewardSentinel(terminal)
    monkeypatch.setattr("dreamerrl.eval.minihack_eval.gym.make", lambda *args, **kwargs: env)
    cfg = minihack_config(0, tmp_path)
    cfg.env.finite_horizon = finite
    if finite:
        world_model = WorldModel(
            obs_space=gym.spaces.Box(0.0, 100.0, (9,), np.float32),
            dreamer_cfg=cfg,
            latent=world_model.latent,
            net=world_model.net_cfg,
        )
    clocks = []
    hook = world_model.encoder.register_forward_pre_hook(lambda module, args: clocks.append(args[0][:, -1].item()))
    actor = Actor(latent=world_model.latent, net=world_model.net_cfg)
    world_model.train()
    actor.train()
    posterior_mode = world_model.posterior.deterministic_latent_for_tests
    rng = torch.get_rng_state().clone()
    result = evaluate_minihack(cfg, world_model, actor, [10001, 10002], random_policy)
    hook.remove()
    if finite and not random_policy:
        assert clocks == pytest.approx([1.0, 5 / 6, 1.0, 5 / 6])
    assert result == [5.0, 5.0]  # 2 + 3, including the reward at termination/truncation
    assert env.seeds == [10001, 10002]
    assert env.closed
    assert actor.training and world_model.training
    assert world_model.posterior.deterministic_latent_for_tests == posterior_mode
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in world_model.parameters())


@pytest.mark.parametrize("sample_actions", [False, True])
@pytest.mark.parametrize("sample_latents", [False, True])
def test_evaluation_modes_are_independent_repeatable_and_restore_rng(
    world_model, tmp_path, monkeypatch, sample_actions, sample_latents
):
    env = RewardSentinel(True)
    monkeypatch.setattr("dreamerrl.eval.minihack_eval.gym.make", lambda *args, **kwargs: env)
    actor = Actor(latent=world_model.latent, net=world_model.net_cfg)
    original_act = actor.act
    choices = []

    def act(state, deterministic=False):
        assert deterministic == (not sample_actions)
        assert world_model.posterior.deterministic_latent_for_tests == (not sample_latents)
        action, logits = original_act(state, deterministic=deterministic)
        choices.append((action.clone(), state.z.clone()))
        return action, logits

    monkeypatch.setattr(actor, "act", act)
    cfg = minihack_config(0, tmp_path)
    original_mode = world_model.posterior.deterministic_latent_for_tests
    rng = torch.get_rng_state().clone()
    first = evaluate_minihack(
        cfg,
        world_model,
        actor,
        [10001, 10002],
        sample_actions=sample_actions,
        sample_latents=sample_latents,
        sampling_seed=42,
    )
    first_choices = list(choices)
    choices.clear()
    torch.rand(17)  # Evaluation sampling must not depend on caller RNG position.
    second = evaluate_minihack(
        cfg,
        world_model,
        actor,
        [10001, 10002],
        sample_actions=sample_actions,
        sample_latents=sample_latents,
        sampling_seed=42,
    )
    assert first == second == [5.0, 5.0]
    assert len(choices) == len(first_choices) == 4
    for (a, z), (expected_a, expected_z) in zip(choices, first_choices):
        assert torch.equal(a, expected_a) and torch.equal(z, expected_z)
    assert world_model.posterior.deterministic_latent_for_tests == original_mode
    torch.set_rng_state(rng)
    evaluate_minihack(
        cfg,
        world_model,
        actor,
        [10001],
        sample_actions=sample_actions,
        sample_latents=sample_latents,
        sampling_seed=42,
    )
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("env_id", ["MiniHack-Room-5x5-v0", "MiniHack-Corridor-R2-v0"])
def test_real_minihack_evaluation_repeats_trajectories(world_model, tmp_path, monkeypatch, env_id):
    """Check actual NLE observations and rewards, not just coincidentally equal returns."""
    make_env = gym.make
    with make_env(env_id) as probe:
        assert isinstance(probe.action_space, gym.spaces.Discrete)
        world_model.net_cfg = replace(world_model.net_cfg, action_dim=probe.action_space.n)
    trajectories = []

    class RecordTrajectory(gym.Wrapper):
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            trajectories.append([(obs["glyphs"].copy(), None, None)])
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            trajectories[-1].append((obs["glyphs"].copy(), reward, (terminated, truncated)))
            return obs, reward, terminated, truncated, info

    monkeypatch.setattr(
        "dreamerrl.eval.minihack_eval.gym.make",
        lambda *args, **kwargs: RecordTrajectory(make_env(*args, **kwargs)),
    )
    cfg = minihack_config(0, tmp_path)
    cfg.env.env_id = env_id
    cfg.env.max_episode_steps = 20
    actor = Actor(latent=world_model.latent, net=world_model.net_cfg)
    first = evaluate_minihack(cfg, world_model, actor, [123, 456], random_policy=True)
    second = evaluate_minihack(cfg, world_model, actor, [123, 456], random_policy=True)
    assert first == second
    for original, repeated in zip(trajectories[:2], trajectories[2:]):
        assert len(original) == len(repeated)
        for (obs, reward, done), (other_obs, other_reward, other_done) in zip(original, repeated):
            np.testing.assert_array_equal(obs, other_obs)
            assert (reward, done) == (other_reward, other_done)


def test_movement_diagnostics_include_final_step_and_reset_each_episode(world_model, tmp_path, monkeypatch):
    from nle import nethack

    class PositionedSentinel(RewardSentinel):
        actions = tuple(cast(type[IntEnum], nethack.CompassDirection))[:5]

        def observation(self):
            stats = np.zeros(27, dtype=np.int64)
            stats[:2] = (1, 1 + int(self.t == 2))
            stats[int(getattr(nethack, "NLE_BL_TIME"))] = int(self.t == 2)
            return {"glyphs": np.zeros((2, 4), dtype=np.float32), "blstats": stats}

        def reset(self, **kwargs):
            super().reset(**kwargs)
            return self.observation(), {}

        def step(self, action):
            _, reward, terminated, truncated, info = super().step(action)
            return self.observation(), reward, terminated, truncated, info

    env = PositionedSentinel(True)
    monkeypatch.setattr("dreamerrl.eval.minihack_eval.gym.make", lambda *args, **kwargs: env)
    actor = Actor(latent=world_model.latent, net=world_model.net_cfg)
    episodes: list[EpisodeDiagnostics] = []
    returns = evaluate_minihack(
        minihack_config(0, tmp_path), world_model, actor, [1, 2], random_policy=True, episode_diagnostics=episodes
    )
    assert returns == [5.0, 5.0]
    for episode in episodes:
        assert episode["steps"] == 2
        assert episode["longest_stationary_streak"] == 1
        assert episode["unique_tiles"] == 2
        assert episode["movement_steps"] == 1
        assert episode["frozen_game_turns"] == 1
        assert sum(episode["action_counts"].values()) == 2
        assert episode["terminated"] and not episode["truncated"]


def corridor_eval_setup(world_model, tmp_path, monkeypatch, max_steps):
    cfg = minihack_config(0, tmp_path)
    cfg.env.env_id = "MiniHack-Corridor-R2-v0"
    cfg.env.max_episode_steps = max_steps
    with gym.make(cfg.env.env_id) as env:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        world_model.net_cfg = replace(world_model.net_cfg, action_dim=env.action_space.n)
    # These tests exercise the real simulator and actor, independent of glyph encoding.
    monkeypatch.setattr(world_model, "observe_step", lambda state, *args, **kwargs: {"post": state})
    return cfg, Actor(latent=world_model.latent, net=world_model.net_cfg)


def test_stair_diagnostic_matches_scripted_step_40(world_model, tmp_path, monkeypatch):
    from dreamerrl.eval.corridor_diagnostics import summarize_stair_opportunities

    cfg, actor = corridor_eval_setup(world_model, tmp_path, monkeypatch, 41)
    path = iter([2, *([3] * 15), 2, *([3] * 16), 0, *([3] * 6)])

    def act(state, deterministic=False):
        action = torch.tensor([next(path)])
        logits = 4 * torch.nn.functional.one_hot(action, 11).float()
        return action, logits

    monkeypatch.setattr(actor, "act", act)
    episodes: list[CorridorEpisode] = []
    assert evaluate_minihack(cfg, world_model, actor, [27017], corridor_diagnostics=episodes) == [1]
    (episode,) = episodes
    assert episode["end_status"] == "TASK_SUCCESSFUL"
    assert episode["terminated"] and not episode["truncated"] and episode["immediate_entry"]
    opportunity = episode["opportunity"]
    assert opportunity is not None
    assert opportunity["step"] == 40 and opportunity["winning_action"] == opportunity["chosen_action"] == 3
    assert opportunity["winning_action_rank"] == 1
    assert summarize_stair_opportunities(episodes)["win_rate_given_opportunity"] == 1
    without_opportunity: CorridorEpisode = {**episode, "opportunity": None}
    assert summarize_stair_opportunities([without_opportunity])["win_rate_given_opportunity"] is None


def test_stair_diagnostics_preserve_sampled_actions_and_native_outcomes(world_model, tmp_path, monkeypatch):
    cfg, actor = corridor_eval_setup(world_model, tmp_path, monkeypatch, 24)
    make = gym.make
    traces = []

    class RecordSteps(gym.Wrapper):
        def reset(self, **kwargs):
            traces.append([])
            return super().reset(**kwargs)

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            traces[-1].append((action, reward, terminated, truncated, info["end_status"]))
            return obs, reward, terminated, truncated, info

    monkeypatch.setattr("dreamerrl.eval.minihack_eval.gym.make", lambda env_id: RecordSteps(make(env_id)))
    seeds = [27000, 27017]
    before = evaluate_minihack(
        cfg, world_model, actor, seeds, sample_actions=True, sample_latents=True, sampling_seed=20
    )
    episodes: list[CorridorEpisode] = []
    after = evaluate_minihack(
        cfg,
        world_model,
        actor,
        seeds,
        corridor_diagnostics=episodes,
        sample_actions=True,
        sample_latents=True,
        sampling_seed=20,
    )
    assert before == after
    assert traces[:2] == traces[2:]
    assert len(episodes) == len(seeds)
    # Seed 27000 starts within one move of the stairs.
    opportunity = episodes[0]["opportunity"]
    assert opportunity is not None
    assert opportunity["step"] == 1
    assert opportunity["chosen_action"] == traces[0][0][0]
