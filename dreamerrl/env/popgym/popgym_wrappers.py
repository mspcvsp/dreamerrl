import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch
from typing import Optional, Dict

from env.env import EnvInterface
from .popgym_preprocessing import flatten_obs


def make_env(env_id):
    def thunk():
        return gym.make(env_id)
    return thunk


class PopGymVecEnv(EnvInterface):
    """
    Vectorized PopGym environment wrapper for Dreamer.
    """

    def __init__(self, env_id: str, batch_size: int, device: torch.device):
        self.device = device
        self.batch_size = batch_size

        self.venv = SyncVectorEnv([make_env(env_id) for _ in range(batch_size)])

        self._obs_dim = int(torch.tensor(self.venv.single_observation_space.shape).prod())
        self._action_dim = self.venv.single_action_space.n

    def reset(self, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if seed is None:
            seeds = None
        else:
            seeds = [seed + i for i in range(self.batch_size)]

        obs, info = self.venv.reset(seed=seeds)
        obs = flatten_obs(obs, self.venv.single_observation_space)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        return {"obs": obs}

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
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

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def action_mask(self):
        return None  # PopGym has no illegal actions

    def get_episode_stats(self):
        return {}
