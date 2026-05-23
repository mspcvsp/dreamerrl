from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete


class PopGymVecWrapper:
    """
    Vectorized PopGym wrapper using Gymnasium's SyncVectorEnv.
    Pylance-clean, Dreamer-V3 compatible.
    """

    def __init__(self, env_name: str, num_envs: int, device: torch.device):
        # Create N independent env constructors
        def make_env():
            return gym.make(env_name)

        self.envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
        self.device = device
        self.num_envs = num_envs

        # Extract spaces
        assert self.envs.single_observation_space.shape is not None, "Only flat observation spaces are supported"
        self.obs_dim = int(np.prod(self.envs.single_observation_space.shape))

        action_space = self.envs.single_action_space
        assert isinstance(action_space, Discrete)
        self.action_dim = action_space.n

        # Metrics
        self.episode_return = np.zeros(num_envs, dtype=np.float32)
        self.episode_length = np.zeros(num_envs, dtype=np.int32)
        self.episode_success = np.zeros(num_envs, dtype=np.int32)

        self.metrics = {
            "return": [],
            "length": [],
            "success": [],
        }

    def reset(self):
        obs, info = self.envs.reset()
        self.episode_return[:] = 0
        self.episode_length[:] = 0
        self.episode_success[:] = 0
        return self._process_obs(obs), info

    def step(self, action_logits: torch.Tensor):
        dist = torch.distributions.Categorical(logits=action_logits)
        actions = dist.sample().cpu().numpy()

        next_obs, reward, terminated, truncated, info = self.envs.step(actions)
        done = np.logical_or(terminated, truncated)

        # Update metrics
        self.episode_return += reward
        self.episode_length += 1
        self.episode_success = np.logical_or(
            self.episode_success,
            info.get("success", np.zeros_like(done)),
        )

        # Log completed episodes
        for i in range(self.num_envs):
            if done[i]:
                self.metrics["return"].append(float(self.episode_return[i]))
                self.metrics["length"].append(int(self.episode_length[i]))
                self.metrics["success"].append(int(self.episode_success[i]))

        return (
            self._process_obs(next_obs),
            reward.astype(np.float32),
            done,
            info,
        )

    def _process_obs(self, obs):
        obs = obs.reshape(self.num_envs, -1).astype(np.float32)
        return torch.from_numpy(obs).to(self.device)

    def get_recent_metrics(self, k=100):
        return {
            "return": np.mean(self.metrics["return"][-k:]) if self.metrics["return"] else 0.0,
            "length": np.mean(self.metrics["length"][-k:]) if self.metrics["length"] else 0.0,
            "success": np.mean(self.metrics["success"][-k:]) if self.metrics["success"] else 0.0,
        }
