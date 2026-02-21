from typing import Any, Dict, Optional, cast

import gymnasium as gym
import torch
from env.env import EnvInterface
from gymnasium.spaces import Discrete
from gymnasium.vector import SyncVectorEnv

from .popgym_preprocessing import flatten_obs


def make_env(env_id):
    def thunk():
        return gym.make(env_id)

    return thunk


class PopGymVecEnv(EnvInterface):
    def __init__(self, env_id: str, batch_size: int, device: torch.device):
        self._batch_size = batch_size
        self.device = device

        self.venv = SyncVectorEnv([make_env(env_id) for _ in range(batch_size)])

        # Observation dimension
        self._obs_dim = int(torch.tensor(self.venv.single_observation_space.shape).prod())

        # Action dimension (narrow type for Pylance)
        assert isinstance(self.venv.single_action_space, Discrete)
        self._action_dim: int = int(self.venv.single_action_space.n)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if seed is None:
            seeds = None
        else:
            seeds = cast(list[int | None], [seed + i for i in range(self._batch_size)])

        obs, info = self.venv.reset(seed=seeds)
        obs = flatten_obs(obs, self.venv.single_observation_space)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        return {"obs": obs}

    def step(self, actions: torch.Tensor) -> Dict[str, Any]:
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions_np = actions.squeeze(-1).cpu().numpy()
        else:
            actions_np = actions.cpu().numpy()

        obs, reward, terminated, truncated, info = self.venv.step(actions_np)

        obs = flatten_obs(obs, self.venv.single_observation_space)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(terminated | truncated, dtype=torch.bool, device=self.device)

        return {
            "obs": obs,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def action_mask(self):
        return None

    def get_episode_stats(self):
        return {}
