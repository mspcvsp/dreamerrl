# dreamer_replay_buffer.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Episode:
    """
    A completed episode stored time-major.
    Shapes:
      state:       (T, obs_dim)
      action:      (T,)  (discrete) or (T, act_dim) (continuous later)
      reward:      (T,)
      is_first:    (T,)
      is_last:     (T,)
      is_terminal: (T,)
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    is_first: torch.Tensor
    is_last: torch.Tensor
    is_terminal: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.state.size(0))


class DreamerReplayBuffer:
    """
    Dreamer-lite / Dreamer replay buffer:
      - Builds per-env episodes online from vectorized env transitions.
      - Stores completed episodes in a ring buffer.
      - Samples random fixed-length sequences (B, L, ...) for training.

    Notes:
      * Dreamer distinguishes episode boundary (is_last) from terminal bootstrap (is_terminal).
      [5](https://github.com/openai/gym/blob/master/gym/vector/vector_env.py)
      * Gymnasium vector env autoreset may overwrite terminal obs; true final obs can live in info['final_observation']
      [6](https://gymnasium.farama.org/api/vector/sync_vector_env/)
      [7](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        capacity_episodes: int,
        device: torch.device,
        store_device: torch.device = torch.device("cpu"),
        min_episode_len: int = 2,
    ):
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.capacity_episodes = int(capacity_episodes)
        self.device = device  # default device for sampled batches
        self.store_device = store_device  # where episodes are stored (CPU recommended)
        self.min_episode_len = int(min_episode_len)

        # Per-env episode builders (lists of per-step tensors)
        self._cur_state: List[List[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._cur_action: List[List[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._cur_reward: List[List[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._cur_is_first: List[List[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._cur_is_last: List[List[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._cur_is_terminal: List[List[torch.Tensor]] = [[] for _ in range(self.num_envs)]

        # Episode ring buffer
        self._episodes: List[Episode] = []
        self._ep_ptr = 0

        # Optional bookkeeping
        self.total_episodes_added = 0
        self.total_steps_added = 0

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_transitions(self) -> int:
        return self.total_steps_added

    def get_episode(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = self._episodes[idx]
        return {
            "state": ep.state,
            "action": ep.action,
            "reward": ep.reward,
            "is_first": ep.is_first,
            "is_last": ep.is_last,
            "is_terminal": ep.is_terminal,
            # Optional: add timestamps for contiguity tests
            "t": torch.arange(ep.length, device=ep.state.device),
        }

    @torch.no_grad()
    def add(self, **kwargs):
        # Wrap single-env transition into batch of size 1
        trans = {k: v.unsqueeze(0) for k, v in kwargs.items()}
        self.add_batch(trans)

    @torch.no_grad()
    def add_batch(self, trans: Dict[str, Any]) -> None:
        """
        Add one environment step for all envs.
        Required keys (batched):
          state:       (B, obs_dim)
          action:      (B,) or (B,1)
          reward:      (B,)
          is_first:    (B,)
          is_last:     (B,)
          is_terminal: (B,)
        Optional:
          info: dict that may include 'final_observation' if autoreset overwrote obs.
          [6](https://gymnasium.farama.org/api/vector/sync_vector_env/)
          [7](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
        """
        state = trans["state"]
        action = trans["action"]
        reward = trans["reward"]
        is_first = trans["is_first"]
        is_last = trans["is_last"]
        is_terminal = trans["is_terminal"]
        info = trans.get("info", None)

        # Normalize shapes
        if action.dim() == 2 and action.size(-1) == 1:
            action = action.squeeze(-1)

        # Basic safety checks
        B = state.size(0)
        assert B == self.num_envs, f"Expected batch {self.num_envs}, got {B}"
        assert state.dim() == 2 and state.size(1) == self.obs_dim, f"state shape {tuple(state.shape)}"
        assert reward.shape == (B,), f"reward shape {tuple(reward.shape)}"
        assert is_first.shape == (B,), f"is_first shape {tuple(is_first.shape)}"
        assert is_last.shape == (B,), f"is_last shape {tuple(is_last.shape)}"
        assert is_terminal.shape == (B,), f"is_terminal shape {tuple(is_terminal.shape)}"
        assert action.shape == (B,), f"action shape {tuple(action.shape)}"

        # ---- Terminal observation overwrite fix (Gymnasium vector autoreset) ----
        # If the vector env autoreset overwrote the terminal observation, some implementations store
        # the real final observation in info['final_observation'].
        # [6](https://gymnasium.farama.org/api/vector/sync_vector_env/)
        # [7](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
        # We only attempt this if info is a dict and the key exists.
        if isinstance(info, dict) and "final_observation" in info:
            final_obs = info["final_observation"]
            # If final_obs is already flattened and batched, this will work directly.
            # If final_obs is unflattened (dict/tuple), adapt upstream (env wrapper) to flatten it.
            final_obs_t = torch.as_tensor(final_obs, device=state.device, dtype=state.dtype)
            if final_obs_t.shape == state.shape:
                state = torch.where(is_last[:, None], final_obs_t, state)

        # Append each env's step
        for i in range(self.num_envs):
            self._cur_state[i].append(state[i].detach().to(self.store_device))
            self._cur_action[i].append(action[i].detach().to(self.store_device))
            self._cur_reward[i].append(reward[i].detach().to(self.store_device))
            self._cur_is_first[i].append(is_first[i].detach().to(self.store_device))
            self._cur_is_last[i].append(is_last[i].detach().to(self.store_device))
            self._cur_is_terminal[i].append(is_terminal[i].detach().to(self.store_device))

            self.total_steps_added += 1

            # Episode ends when is_last is True (next step belongs to new episode).
            # [5](https://github.com/openai/gym/blob/master/gym/vector/vector_env.py)
            if bool(is_last[i].item()):
                self._finalize_episode(i)

    def _finalize_episode(self, env_i: int) -> None:
        T = len(self._cur_state[env_i])
        if T < self.min_episode_len:
            # Drop too-short episodes (often just reset artifacts)
            self._clear_builder(env_i)
            return

        state = torch.stack(self._cur_state[env_i], dim=0)  # (T, obs_dim)
        action = torch.stack(self._cur_action[env_i], dim=0)  # (T,)
        reward = torch.stack(self._cur_reward[env_i], dim=0)  # (T,)
        is_first = torch.stack(self._cur_is_first[env_i], dim=0).bool()  # (T,)
        is_last = torch.stack(self._cur_is_last[env_i], dim=0).bool()  # (T,)
        is_terminal = torch.stack(self._cur_is_terminal[env_i], dim=0).bool()

        ep = Episode(
            state=state,
            action=action,
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )

        if len(self._episodes) < self.capacity_episodes:
            self._episodes.append(ep)
        else:
            self._episodes[self._ep_ptr] = ep
            self._ep_ptr = (self._ep_ptr + 1) % self.capacity_episodes

        self.total_episodes_added += 1
        self._clear_builder(env_i)

    def _clear_builder(self, env_i: int) -> None:
        self._cur_state[env_i].clear()
        self._cur_action[env_i].clear()
        self._cur_reward[env_i].clear()
        self._cur_is_first[env_i].clear()
        self._cur_is_last[env_i].clear()
        self._cur_is_terminal[env_i].clear()

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Uniformly sample batch of sequences of length L from stored episodes.
        Returns time-major or batch-major? Here we return (B, L, ...).
        """
        assert len(self._episodes) > 0, "Replay buffer is empty."
        device = device or self.device
        B = int(batch_size)
        L = int(seq_len)

        episodes: List[Episode] = []
        starts: List[int] = []

        # Rejection sampling until we have B valid sequences
        while len(episodes) < B:
            ep = random.choice(self._episodes)
            if ep.length < L:
                continue
            start = random.randint(0, ep.length - L)
            episodes.append(ep)
            starts.append(start)

        # Stack into (B, L, ...)
        state = torch.stack([ep.state[s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device)
        action = torch.stack([ep.action[s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device)
        reward = torch.stack([ep.reward[s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device)
        is_first = torch.stack([ep.is_first[s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device)
        is_last = torch.stack([ep.is_last[s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device)
        is_terminal = torch.stack([ep.is_terminal[s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device)

        return {
            "state": state,  # (B, L, obs_dim)
            "action": action,  # (B, L)
            "reward": reward,  # (B, L)
            "is_first": is_first,  # (B, L)
            "is_last": is_last,  # (B, L)
            "is_terminal": is_terminal,  # (B, L)
        }

    def stats(self) -> Dict[str, int]:
        return {
            "episodes": len(self._episodes),
            "capacity_episodes": self.capacity_episodes,
            "total_episodes_added": self.total_episodes_added,
            "total_steps_added": self.total_steps_added,
        }
