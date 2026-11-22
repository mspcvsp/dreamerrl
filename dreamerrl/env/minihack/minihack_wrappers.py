from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import TimeLimit

import minihack  # noqa: F401
from dreamerrl.env.env import EnvInterface
from dreamerrl.env.minihack.minihack_utils import PreserveFinalObservation, extract_glyphs, glyph_vocab_size
from dreamerrl.utils.types import EnvironmentConfig


def make_mhack_env(env_cfg: EnvironmentConfig, idx: int) -> Callable[[], gym.Env]:
    def thunk():
        seed = env_cfg.seed + idx
        env = gym.make(env_cfg.env_id)
        env = PreserveFinalObservation(TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps))
        env.reset(seed=seed)
        return env

    return thunk


class MiniHackVecEnv(EnvInterface):
    """
    Dreamer‑V3 vector environment wrapper for MiniHack.
    - Extracts glyphs from dict observations
    - Flattens symbolic grid
    - Per‑environment resets
    - Dreamer‑native return dict
    """

    def __init__(self, env_cfg: EnvironmentConfig, device: torch.device, probe=None):
        self.device = device
        self.probe = probe
        self._batch_size = env_cfg.num_envs
        self.deterministic = env_cfg.deterministic
        self.base_seed = env_cfg.seed

        # Vectorized MiniHack
        self.venv = SyncVectorEnv(
            [make_mhack_env(env_cfg, i) for i in range(self._batch_size)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )

        # Infer observation shape from real reset
        obs, _ = self.venv.reset(seed=self.base_seed)
        glyphs = extract_glyphs(obs)

        # Flatten once and use that to define obs_dim
        state0 = self._flatten_obs(glyphs)  # (batch_size, obs_dim)
        self._obs_dim = state0.shape[1]

        # You don't actually need the grid shape for Dreamer, only obs_dim.
        self._obs_shape = (self._obs_dim,)

        self._obs_space = Box(
            low=0,
            high=glyph_vocab_size() - 1,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # Discrete action space
        act_space = self.venv.single_action_space
        assert isinstance(act_space, Discrete)
        self._action_dim = int(act_space.n)

        # Track previous action (one-hot)
        self._prev_action = torch.zeros(
            self._batch_size,
            self._action_dim,
            dtype=torch.float32,
            device=self.device,
        )

        self._needs_first = torch.ones(self._batch_size, dtype=torch.bool, device=self.device)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _flatten_obs(self, obs: np.ndarray) -> torch.Tensor:
        flat = obs.reshape(self._batch_size, -1)
        return torch.tensor(flat, dtype=torch.float32, device=self.device)

    def _flatten_single(self, obs: np.ndarray) -> torch.Tensor:
        flat = obs.reshape(1, -1)
        return torch.tensor(flat, dtype=torch.float32, device=self.device)

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        obs, info = self.venv.reset(seed=self.base_seed if seed is None else seed)
        glyphs = extract_glyphs(obs)
        state = self._flatten_obs(glyphs)

        self._prev_action.zero_()
        self._needs_first[:] = True

        if self.probe:
            self.probe.env_reset(glyphs.tolist())

        return {
            "state": state,
            "prev_action": self._prev_action.clone(),
            "reward": torch.zeros(self._batch_size, dtype=torch.float32, device=self.device),
            "is_first": torch.ones(self._batch_size, dtype=torch.bool, device=self.device),
            "is_last": torch.zeros(self._batch_size, dtype=torch.bool, device=self.device),
            "is_terminal": torch.zeros(self._batch_size, dtype=torch.bool, device=self.device),
            "info": info,
        }

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------
    def step(self, actions: torch.Tensor) -> Dict[str, Any]:
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions_np = actions.squeeze(-1).cpu().numpy()
        else:
            actions_np = actions.cpu().numpy()

        obs, reward, terminated, truncated, info = self.venv.step(actions_np)

        glyphs = extract_glyphs(obs)
        state = self._flatten_obs(glyphs)

        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
        terminated_t = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        truncated_t = torch.tensor(truncated, dtype=torch.bool, device=self.device)

        is_last = terminated_t | truncated_t
        is_terminal = terminated_t
        # SAME_STEP autoreset returns reset observations in state. Preserve the
        # actual transition observations separately for reward/model supervision.
        transition_state = state.clone()
        for i in range(self._batch_size):
            if is_last[i]:
                final = info["final_obs"][i]
                transition_state[i] = self._flatten_single(extract_glyphs(final))[0]
        is_first = is_last.clone()
        actions_t = actions.reshape(-1).long().to(self.device)
        prev = torch.nn.functional.one_hot(actions_t, self._action_dim).float()
        prev[is_last] = 0
        self._prev_action = prev
        self._needs_first = is_first.clone()

        if self.probe:
            self.probe.env_step(
                glyphs.tolist(),
                reward_t.tolist(),
                terminated_t.tolist(),
                truncated_t.tolist(),
                is_last.tolist(),
            )

        return {
            "state": state,
            "transition_state": transition_state,
            "prev_action": self._prev_action.clone(),
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
