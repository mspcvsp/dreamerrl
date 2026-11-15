"""Discrete-cue PopGym adapters with the same transition contract as MiniHack."""

from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import TimeLimit

import popgym  # noqa: F401
from dreamerrl.env.env import EnvInterface
from dreamerrl.utils.types import EnvironmentConfig


def make_env(env_cfg: EnvironmentConfig, idx: int) -> Callable[[], gym.Env]:
    def thunk():
        env = TimeLimit(gym.make(env_cfg.env_id), max_episode_steps=env_cfg.max_episode_steps)
        env.reset(seed=env_cfg.seed + idx)
        return env

    return thunk


class PopGymVecEnv(EnvInterface):
    """One-hot observations, native rewards, and preserved final observations.

    state is ready for the next policy action; transition_state is the observation
    produced by the executed action, including on same-step autoresets.
    """

    def _build_vector(self, cfg):
        return SyncVectorEnv([make_env(cfg, i) for i in range(cfg.num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)

    def __init__(self, env_cfg: EnvironmentConfig, device: torch.device, probe=None):
        self.device = device
        self.probe = probe
        self.base_seed = env_cfg.seed
        self._batch_size = env_cfg.num_envs
        self.venv = self._build_vector(env_cfg)
        obs_space = self.venv.single_observation_space
        action_space = self.venv.single_action_space
        if not isinstance(obs_space, Discrete) or not isinstance(action_space, Discrete):
            self.venv.close()
            raise ValueError("PopGym adapters currently require Discrete observations and actions")
        self._num_categories = int(obs_space.n)
        self._action_dim = int(action_space.n)
        self._obs_space = Box(0, 1, (self._num_categories,), np.float32)
        self._prev_action = torch.zeros(self._batch_size, self._action_dim, device=device)

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def obs_dim(self):
        return self._num_categories

    @property
    def action_dim(self):
        return self._action_dim

    def _one_hot(self, obs):
        ids = torch.as_tensor(np.asarray(obs).copy(), dtype=torch.long, device=self.device).reshape(-1)
        return torch.nn.functional.one_hot(ids, self._num_categories).float()

    def reset(self, seed=None) -> dict[str, Any]:
        obs, info = self.venv.reset(seed=self.base_seed if seed is None else seed)
        self._prev_action.zero_()
        if self.probe:
            self.probe.env_reset(obs.tolist())
        return dict(
            state=self._one_hot(obs),
            prev_action=self._prev_action.clone(),
            reward=torch.zeros(self.batch_size, device=self.device),
            is_first=torch.ones(self.batch_size, dtype=torch.bool, device=self.device),
            is_last=torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
            is_terminal=torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
            info=info,
        )

    def step(self, actions) -> dict[str, Any]:
        action_ids = actions.detach().reshape(-1).long()
        obs, reward, terminated, truncated, info = self.venv.step(action_ids.cpu().numpy())
        state = self._one_hot(obs)
        terminal = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        last = terminal | torch.as_tensor(truncated, dtype=torch.bool, device=self.device)
        transition_state = state.clone()
        for i in last.nonzero(as_tuple=True)[0].tolist():
            transition_state[i] = self._one_hot([info["final_obs"][i]])[0]
        self._prev_action = torch.nn.functional.one_hot(action_ids.to(self.device), self.action_dim).float()
        self._prev_action[last] = 0
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        if self.probe:
            self.probe.env_step(obs.tolist(), reward.tolist(), terminated.tolist(), truncated.tolist(), last.tolist())
        return dict(
            state=state,
            transition_state=transition_state,
            reward=reward,
            prev_action=self._prev_action.clone(),
            is_first=last.clone(),
            is_last=last,
            is_terminal=terminal,
            info=info,
        )

    def action_mask(self):
        return None

    def get_episode_stats(self):
        return {}
