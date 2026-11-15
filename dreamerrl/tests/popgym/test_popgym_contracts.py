"""Sentinel regressions for native rewards, independent resets, and evaluation."""

import importlib

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.eval.popgym_eval import aggregate_popgym_results, evaluate_popgym, train_popgym_seed
from dreamerrl.models.actor import Actor
from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import EnvironmentConfig


class CueSentinel(gym.Env):
    observation_space = gym.spaces.Discrete(8)
    action_space = gym.spaces.Discrete(5)

    def __init__(self, index):
        self.index = index

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return 0, {}

    def step(self, action):
        self.t += 1
        end = self.t == self.index + 1
        return self.t, -0.25 * self.t, end and self.index == 0, end and self.index != 0, {}


@pytest.mark.parametrize("parallel", [False, True])
def test_popgym_native_rewards_and_reset_contract(parallel, monkeypatch):
    name = "popgym_parallel_env" if parallel else "popgym_wrappers"
    module = importlib.import_module("dreamerrl.env.popgym." + name)
    monkeypatch.setattr(module, "make_env", lambda cfg, i: lambda: CueSentinel(i))
    cls = module.PopGymParallelEnv if parallel else module.PopGymVecEnv
    env = cls(EnvironmentConfig(num_envs=3), torch.device("cpu"))
    try:
        env.reset(seed=123)
        out = env.step(torch.zeros(3, dtype=torch.long))
        assert out["state"].argmax(-1).tolist() == [0, 1, 1]
        assert out["transition_state"].argmax(-1).tolist() == [1, 1, 1]
        assert out["is_terminal"].tolist() == [True, False, False]
        assert out["is_first"].tolist() == [True, False, False]
        torch.testing.assert_close(out["reward"], torch.full((3,), -0.25))
        out = env.step(torch.ones(3, dtype=torch.long))
        assert out["state"].argmax(-1).tolist() == [0, 0, 2]
        assert out["transition_state"].argmax(-1).tolist() == [1, 2, 2]
        assert out["is_last"].tolist() == [True, True, False]
        assert out["is_terminal"].tolist() == [True, False, False]
        torch.testing.assert_close(out["reward"], torch.tensor([-0.25, -0.5, -0.5]))
    finally:
        env.venv.close()


@pytest.mark.parametrize("parallel", [False, True])
def test_popgym_evaluation_latches_first_end(world_model, parallel, monkeypatch):
    name = "popgym_parallel_env" if parallel else "popgym_wrappers"
    module = importlib.import_module("dreamerrl.env.popgym." + name)
    monkeypatch.setattr(module, "make_env", lambda cfg, i: lambda: CueSentinel(i))
    cls = module.PopGymParallelEnv if parallel else module.PopGymVecEnv
    env = cls(EnvironmentConfig(num_envs=3), torch.device("cpu"))
    actor = Actor(latent=world_model.latent, net=world_model.net_cfg)
    rng = torch.get_rng_state().clone()
    posterior_mode = world_model.posterior.deterministic_latent_for_tests
    try:
        result = evaluate_popgym(env, world_model, actor, episodes=2)
        assert result["episode_returns"] == [[-0.25, -0.75, -1.5]] * 2
        assert result["per_env"] == [-0.25, -0.75, -1.5]
        assert torch.equal(torch.get_rng_state(), rng)
        assert world_model.training and actor.training
        assert world_model.posterior.deterministic_latent_for_tests == posterior_mode
    finally:
        env.venv.close()


def test_popgym_training_harness_records_full_episodes(tmp_path):
    cfg = minihack_config(3, tmp_path)
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"
    trainer = DreamerTrainer(cfg)
    original_step = trainer.env.step
    try:
        result = train_popgym_seed(trainer, steps=2)
        # The trainer independently accumulates episode returns in collection.
        np.testing.assert_allclose(result["returns"], trainer.recent_returns, rtol=0, atol=0)
        assert len(result["returns"]) > 0
        assert result["wm_loss"].shape == (2,)
        assert trainer.env.step == original_step
        assert cfg.train.collect_steps == 12
        summary = aggregate_popgym_results([result])
        assert all(np.isfinite(list(summary.values())))
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
