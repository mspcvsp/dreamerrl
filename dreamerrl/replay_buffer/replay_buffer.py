from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from dreamerrl.utils.types import TrainingConfig


class Episode:
    """Stores a single episode for Dreamer-V3."""

    def __init__(self):
        self.data: Dict[str, List[torch.Tensor]] = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "is_terminal": [],
            "is_first": [],
        }

    def add(self, obs, action, reward, done, is_terminal, is_first=False):
        self.data["is_first"].append(torch.as_tensor(is_first, device=obs.device))
        self.data["obs"].append(obs)
        self.data["action"].append(action)
        self.data["reward"].append(reward)
        self.data["done"].append(done)
        self.data["is_terminal"].append(is_terminal)

    def finalize(self) -> Dict[str, torch.Tensor]:
        return {k: torch.stack(v, dim=0) for k, v in self.data.items()}


class ReplayBuffer:
    """
    Dreamer-V3 replay buffer:
      - stores full episodes
      - samples fixed-length sequences
      - supports vectorized envs
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        probe=None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.probe = probe

        self.capacity = self.cfg.replay_capacity
        self.device = device
        self.seq_len = self.cfg.seq_len

        self.episodes: List[Dict[str, torch.Tensor]] = []
        self.current_eps: List[Episode] = []
        self.num_envs = None

        self.size = 0
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.rng = torch.Generator("cpu")
        self.rng.manual_seed(seed)

    def _ensure_envs(self, num_envs: int):
        if self.num_envs is None:
            self.num_envs = num_envs
            self.current_eps = [Episode() for _ in range(num_envs)]

    def add(self, obs, action, reward, done, is_terminal, initial_obs=None):
        """
        obs: (num_envs, obs_dim)
        action: (num_envs,)
        reward: (num_envs,)
        done: (num_envs,)
        """
        num_envs = obs.shape[0]
        self._ensure_envs(num_envs)

        for i in range(num_envs):
            if initial_obs is not None and not self.current_eps[i].data["obs"]:
                self.current_eps[i].add(
                    initial_obs[i].detach().to(self.device).clone(),
                    torch.zeros_like(action[i], device=self.device),
                    torch.zeros_like(reward[i], device=self.device),
                    torch.zeros_like(done[i], device=self.device),
                    torch.zeros_like(is_terminal[i], device=self.device),
                    is_first=True,
                )
            self.current_eps[i].add(
                obs[i].detach().to(self.device).clone(),
                action[i].detach().to(self.device).clone(),
                reward[i].detach().to(self.device).clone(),
                done[i].detach().to(self.device).clone(),
                is_terminal[i].detach().to(self.device).clone(),
            )

            if bool(done[i]):
                self._finalize_episode(i)

    def _finalize_episode(self, idx: int):
        ep = self.current_eps[idx].finalize()
        self.episodes.append(ep)
        self.current_eps[idx] = Episode()

        # capacity control
        self.size += ep["obs"].shape[0]
        while self.size > self.capacity:
            removed = self.episodes.pop(0)
            self.size -= removed["obs"].shape[0]

        if self.probe:
            self.probe.replay_finalize(
                idx=idx,
                length=ep["obs"].shape[0],
                total_size=self.size,
            )

    def state_dict(self):
        """Persist completed episodes only; live environments cannot be restored."""
        return {
            "episodes": [{k: v.detach().cpu() for k, v in ep.items()} for ep in self.episodes],
            "rng": self.rng.get_state(),
            "seq_len": self.seq_len,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
        }

    def load_state_dict(self, state):
        for key in ("seq_len", "obs_dim", "action_dim", "capacity"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Checkpoint replay {key} does not match the current configuration")
        self.episodes = [{k: v.to(self.device) for k, v in ep.items()} for ep in state["episodes"]]
        self.size = sum(ep["obs"].shape[0] for ep in self.episodes)
        self.rng.set_state(state["rng"].cpu())
        self.current_eps = []
        self.num_envs = None

    def sample(self, batch_size: int, seed: int | None = None) -> Dict[str, torch.Tensor]:
        assert len(self.episodes) > 0, "Replay buffer is empty"

        if seed is not None:
            self.rng.manual_seed(seed)
            np.random.seed(seed)

        valid_indices = [i for i, ep in enumerate(self.episodes) if ep["obs"].shape[0] >= self.seq_len]
        assert len(valid_indices) > 0, f"No episodes with length >= seq_len ({self.seq_len})"

        obs_batch = []
        act_batch = []
        rew_batch = []
        done_batch = []
        terminal_batch = []
        first_batch = []

        for _ in range(batch_size):
            idx = valid_indices[int(torch.randint(0, len(valid_indices), (1,), generator=self.rng))]
            ep = self.episodes[idx]

            length = ep["obs"].shape[0]

            if length <= self.seq_len:
                start = 0
            else:
                start = int(torch.randint(0, length - self.seq_len + 1, (1,), generator=self.rng))

            if self.probe:
                self.probe.replay_sample(seed=seed, ep_idx=idx, ep_len=length, start=start)

            end = start + self.seq_len

            obs_batch.append(ep["obs"][start:end])
            act_batch.append(ep["action"][start:end])
            rew_batch.append(ep["reward"][start:end])
            done_batch.append(ep["done"][start:end])
            first_batch.append(ep["is_first"][start:end])
            terminal_batch.append(ep["is_terminal"][start:end])

        return {
            "obs": torch.stack(obs_batch).to(self.device),
            "action": torch.stack(act_batch).to(self.device),
            "reward": torch.stack(rew_batch).to(self.device),
            "done": torch.stack(done_batch).to(self.device),
            "is_terminal": torch.stack(terminal_batch).to(self.device),
            "is_first": torch.stack(first_batch).to(self.device),
            "is_last": torch.stack(done_batch).to(self.device),
        }
