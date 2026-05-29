from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import torch
from gymnasium.spaces import Discrete
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit

import popgym  # noqa: F401  # ensures PopGym registers its environments
from dreamerrl.env.env import EnvInterface
from dreamerrl.utils.types import EnvironmentConfig

from .popgym_preprocessing import flatten_obs


def make_env(env_cfg: EnvironmentConfig, idx: int) -> Callable[[], gym.Env]:
    def thunk():
        env = gym.make(env_cfg.env_id)
        env = TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)

        # Deterministic seeding per environment
        if getattr(env_cfg, "deterministic", False):
            env.reset(seed=env_cfg.seed + idx)

        return env

    return thunk


class PopGymVecEnv(EnvInterface):
    """
    Dreamer-native vector env wrapper:
      - observation key: `state`
      - boundary flags: is_first, is_last, is_terminal
      - seed fix: pass `seed` as int|None directly to SyncVectorEnv.reset()
    """

    def __init__(self, env_cfg: EnvironmentConfig, device: torch.device):
        self._batch_size = env_cfg.num_envs
        self.device = device
        self.deterministic = env_cfg.deterministic
        self.base_seed = env_cfg.seed

        self.venv = SyncVectorEnv([make_env(env_cfg, idx) for idx in range(self._batch_size)])

        # Observation dimension (flattened)
        self._obs_dim = int(torch.tensor(self.venv.single_observation_space.shape).prod())

        # Action dimension (discrete)
        assert isinstance(self.venv.single_action_space, Discrete)
        self._action_dim: int = int(self.venv.single_action_space.n)

        # Track "first step" markers for Dreamer-style streaming
        self._needs_first = torch.ones(self._batch_size, dtype=torch.bool, device=self.device)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Seed fix:
          Gymnasium SyncVectorEnv.reset accepts:
            seed: int | list[int | None] | None
          Passing an int expands internally to [seed, seed+1, ..., seed+n]. [2]
          (https://deepwiki.com/burchim/DreamerV3-PyTorch/5.1-environment-wrappers-and-interface)
        """
        obs, info = self.venv.reset(seed=seed)

        obs = flatten_obs(obs, self.venv.single_observation_space)
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # After reset, next emitted transition is the first step for each env
        self._needs_first[:] = True

        # Return Dreamer-style initial sample (reward=0, is_first=True)
        return {
            "state": state,
            "reward": torch.zeros(self._batch_size, dtype=torch.float32, device=self.device),
            "is_first": torch.ones(self._batch_size, dtype=torch.bool, device=self.device),
            "is_last": torch.zeros(self._batch_size, dtype=torch.bool, device=self.device),
            "is_terminal": torch.zeros(self._batch_size, dtype=torch.bool, device=self.device),
            "info": info,
        }

    def step(self, actions: torch.Tensor) -> Dict[str, Any]:
        # Normalize actions to shape (B,)
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions_np = actions.squeeze(-1).detach().cpu().numpy()
        else:
            actions_np = actions.detach().cpu().numpy()

        obs, reward, terminated, truncated, info = self.venv.step(actions_np)

        obs = flatten_obs(obs, self.venv.single_observation_space)
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

        terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

        # Dreamer flags:
        is_terminal = terminated_t
        is_last = terminated_t | truncated_t
        is_first = self._needs_first.clone()

        # Auto-reset ended envs and stitch reset obs into returned `state`
        # This yields a continuous stream of transitions for fixed-horizon sampling.
        if bool(is_last.any()):
            if self.deterministic:
                # Only reset envs where is_last[i] is True
                seeds: list[int | None] = [
                    (self.base_seed + i) if is_last[i].item() else None for i in range(self._batch_size)
                ]
                reset_obs, _ = self.venv.reset(seed=seeds)
            else:
                reset_obs, _ = self.venv.reset()

            reset_obs = flatten_obs(reset_obs, self.venv.single_observation_space)
            reset_state = torch.as_tensor(reset_obs, dtype=torch.float32, device=self.device)

            # Replace only the finished envs
            state = torch.where(is_last[:, None], reset_state, state)

            # Mark those envs as first on the next transition
            self._needs_first = is_last.clone()
        else:
            self._needs_first.zero_()

        return {
            "state": state,
            "reward": reward_t,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "info": info,
        }

    def action_mask(self):
        return None

    def get_episode_stats(self):
        return {}
