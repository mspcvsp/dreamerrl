"""Evaluation must count whole episodes and leave training state untouched."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.tests.manual.utils import evaluate_minihack, minihack_config


class RewardSentinel(gym.Env):
    action_space = gym.spaces.Discrete(5)
    observation_space = gym.spaces.Dict({"glyphs": gym.spaces.Box(0, 100, (2, 4), np.float32)})

    def __init__(self, terminal):
        self.terminal = terminal
        self.seeds = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
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


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("random_policy", [False, True])
def test_held_out_evaluation_counts_final_reward_and_restores_state(
    world_model,
    tmp_path,
    monkeypatch,
    terminal,
    random_policy,
):
    env = RewardSentinel(terminal)
    monkeypatch.setattr("dreamerrl.tests.manual.utils.gym.make", lambda *args, **kwargs: env)
    cfg = minihack_config(0, tmp_path)
    actor = Actor(latent=world_model.latent, net=world_model.net_cfg)
    world_model.train()
    actor.train()
    posterior_mode = world_model.posterior.deterministic_latent_for_tests
    rng = torch.get_rng_state().clone()
    result = evaluate_minihack(cfg, world_model, actor, [10001, 10002], random_policy)
    assert result == [5.0, 5.0]  # 2 + 3, including the reward at termination/truncation
    assert env.seeds == [10001, 10002]
    assert env.closed
    assert actor.training and world_model.training
    assert world_model.posterior.deterministic_latent_for_tests == posterior_mode
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in world_model.parameters())
