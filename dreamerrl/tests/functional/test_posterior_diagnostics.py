"""Separate uncertainty, peaked collapse, observation sensitivity, and raw KL."""

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dreamerrl.eval.location_probe import FrozenAgent
from dreamerrl.eval.posterior_diagnostics import (
    collect_trace,
    conditional_sensitivity,
    filter_trace,
    js_divergence,
    posterior_usage,
    reconstruction_ablations,
)
from dreamerrl.models.actor import Actor
from dreamerrl.utils.types import DreamerConfig


def test_uniform_posterior_has_high_uncertainty_despite_constant_mode():
    q = torch.full((40, 2, 4), 0.25)
    z = F.one_hot(torch.arange(40) % 4, 4).float().unsqueeze(1).repeat(1, 2, 1)
    result = posterior_usage(q, q, z, 1.0)
    assert result["posterior_normalized_entropy"] == 1
    assert result["mean_mode_categories_used"] == 1
    assert result["per_factor"][0]["sample_categories_used"] == 4
    assert result["raw_kl_total_mean_nats"] == 0
    assert result["floored_kl_mean_nats"] == 1
    assert result["fraction_below_free_nats"] == 1
    assert result["marginal_entropy_gap_total_nats"] == 0


def test_peaked_but_varying_posterior_is_not_a_single_category_collapse():
    q = F.one_hot(torch.arange(40) % 2, 4).float().unsqueeze(1).repeat(1, 2, 1)
    result = posterior_usage(q, torch.full_like(q, 0.25), q, 1.0)
    assert result["posterior_normalized_entropy"] == 0
    assert result["mean_mode_categories_used"] == 2
    assert abs(result["raw_kl_total_mean_nats"] - 2 * math.log(4)) < 1e-5
    assert abs(result["marginal_entropy_gap_total_nats"] - 2 * math.log(2)) < 1e-5
    torch.testing.assert_close(js_divergence(q, q), torch.zeros(40, 2))


class TerminalTraceEnv(gym.Env):
    observation_space = gym.spaces.Dict({"glyphs": gym.spaces.Box(0, 10, (2, 4), np.int32)})
    action_space = gym.spaces.Discrete(5)

    def seed(self, **kwargs):
        pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.obs(), {}

    def obs(self):
        return {"glyphs": np.full((2, 4), self.t, np.int32), "blstats": np.array([self.t, 0])}

    def step(self, action):
        self.t += 1
        return self.obs(), float(self.t == 2), self.t == 2, False, {}


def test_terminal_alignment_freezing_sensitivity_and_ablations(world_model, monkeypatch):
    monkeypatch.setattr("dreamerrl.eval.posterior_diagnostics.gym.make", lambda *args, **kwargs: TerminalTraceEnv())
    cfg = DreamerConfig()
    cfg.env.finite_horizon = True
    cfg.env.max_episode_steps = 4
    # Collection verifies clock semantics separately; remove it for this vector fixture.
    agent = FrozenAgent(
        world_model.eval().requires_grad_(False),
        Actor(latent=world_model.latent, net=world_model.net_cfg),
        cfg,
        "test",
        0,
    )
    trace = collect_trace(agent, [10, 11], max_steps=4)
    assert trace["ranges"] == [(0, 3), (3, 6)]
    assert trace["reward"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert trace["positions"].tolist() == [[0, 0], [1, 0], [2, 0]] * 2
    torch.testing.assert_close(trace["obs"][:3, -1], torch.tensor([1.0, 0.75, 0.5]))
    trace["obs"] = trace["obs"][:, :-1]
    before = {name: value.clone() for name, value in world_model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    states = filter_trace(agent, trace, sampled=True)
    sensitivity = conditional_sensitivity(world_model, states, trace)
    assert sensitivity["same_level_position_swap_pairs"] == 6
    assert sensitivity["same_level_position_swap_js_total_nats"] >= -1e-6
    assert sensitivity["observation_swap_js_total_nats"] >= -1e-6
    assert 0 <= sensitivity["observation_swap_mode_change_fraction"] <= 1
    ablations = reconstruction_ablations(world_model, states, trace, frames=4, cells=4)
    assert set(ablations["variants"]) == {"posterior", "shuffled_z", "zero_z", "prior_z", "shuffled_h"}
    assert torch.equal(torch.get_rng_state(), rng)
    torch.rand(10)
    assert ablations == reconstruction_ablations(world_model, states, trace, frames=4, cells=4)
    for name, value in world_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
