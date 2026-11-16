"""Evaluation must count whole episodes and leave training state untouched."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.eval.minihack_eval import evaluate_minihack
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
