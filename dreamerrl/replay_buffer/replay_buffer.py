from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Episode:
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
    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        capacity_episodes: int,
        device: torch.device,
        store_device: torch.device = torch.device("cpu"),
        min_episode_len: int = 2,
    ):
        # CPU generator ensures CPU/GPU equivalence
        self._g = torch.Generator(device="cpu")
        self._g.manual_seed(0)

        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.capacity = int(capacity_episodes)
        self.device = device
        self.store_device = store_device
        self.min_episode_len = int(min_episode_len)

        self._cur_state = [[] for _ in range(self.num_envs)]
        self._cur_action = [[] for _ in range(self.num_envs)]
        self._cur_reward = [[] for _ in range(self.num_envs)]
        self._cur_is_first = [[] for _ in range(self.num_envs)]
        self._cur_is_last = [[] for _ in range(self.num_envs)]
        self._cur_is_terminal = [[] for _ in range(self.num_envs)]

        self._episodes: List[Episode] = []
        self._start = 0

        self.total_steps_added = 0
        self.total_episodes_added = 0

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_transitions(self) -> int:
        return self.total_steps_added

    def get_episode(self, idx: int) -> Dict[str, torch.Tensor]:
        physical = (self._start + idx) % self.capacity
        ep = self._episodes[physical]
        T = ep.length
        return {
            "state": ep.state,
            "action": ep.action,
            "reward": ep.reward,
            "is_first": ep.is_first,
            "is_last": ep.is_last,
            "is_terminal": ep.is_terminal,
            "t": torch.arange(T, device=ep.state.device),
        }

    @torch.no_grad()
    def add(self, **kwargs):
        trans = {k: v.unsqueeze(0) for k, v in kwargs.items()}
        self.add_batch(trans)

    @torch.no_grad()
    def add_batch(self, trans: Dict[str, Any]) -> None:
        state = trans["state"]
        action = trans["action"]
        reward = trans["reward"]
        is_first = trans["is_first"]
        is_last = trans["is_last"]
        is_terminal = trans["is_terminal"]

        if action.dim() == 2 and action.size(-1) == 1:
            action = action.squeeze(-1)

        B = state.size(0)
        assert B == self.num_envs

        for i in range(self.num_envs):
            self._cur_state[i].append(state[i].detach().to(self.store_device))
            self._cur_action[i].append(action[i].detach().to(self.store_device))
            self._cur_reward[i].append(reward[i].detach().to(self.store_device))
            self._cur_is_first[i].append(is_first[i].detach().to(self.store_device))
            self._cur_is_last[i].append(is_last[i].detach().to(self.store_device))
            self._cur_is_terminal[i].append(is_terminal[i].detach().to(self.store_device))

            self.total_steps_added += 1

            if bool(is_last[i].item()):
                self._finalize_episode(i)

    def _finalize_episode(self, env_i: int) -> None:
        T = len(self._cur_state[env_i])
        if T < self.min_episode_len:
            self._clear_builder(env_i)
            return

        ep = Episode(
            state=torch.stack(self._cur_state[env_i], dim=0),
            action=torch.stack(self._cur_action[env_i], dim=0),
            reward=torch.stack(self._cur_reward[env_i], dim=0),
            is_first=torch.stack(self._cur_is_first[env_i], dim=0).bool(),
            is_last=torch.stack(self._cur_is_last[env_i], dim=0).bool(),
            is_terminal=torch.stack(self._cur_is_terminal[env_i], dim=0).bool(),
        )

        if len(self._episodes) < self.capacity:
            self._episodes.append(ep)
        else:
            self._episodes[self._start] = ep
            self._start = (self._start + 1) % self.capacity

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
        assert self.num_episodes > 0
        device = device or self.device

        B = int(batch_size)
        L = int(seq_len)

        # CPU generator ensures identical sampling on CPU/GPU
        g = self._g

        ep_indices = torch.randint(
            low=0,
            high=self.num_episodes,
            size=(B,),
            generator=g,
            device="cpu",
        ).tolist()

        episodes = []
        starts = []

        for ep_idx in ep_indices:
            ep = self.get_episode(ep_idx)
            T = ep["state"].shape[0]

            if T <= L:
                start = 0
            else:
                start = torch.randint(
                    low=0,
                    high=T - L + 1,
                    size=(1,),
                    generator=g,
                    device="cpu",
                ).item()

            episodes.append(ep)
            starts.append(start)

        return {
            "state": torch.stack([ep["state"][s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device),
            "action": torch.stack([ep["action"][s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device),
            "reward": torch.stack([ep["reward"][s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device),
            "is_first": torch.stack([ep["is_first"][s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device),
            "is_last": torch.stack([ep["is_last"][s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(device),
            "is_terminal": torch.stack([ep["is_terminal"][s : s + L] for ep, s in zip(episodes, starts)], dim=0).to(
                device
            ),
        }

    def stats(self) -> Dict[str, int]:
        return {
            "episodes": self.num_episodes,
            "capacity_episodes": self.capacity,
            "total_episodes_added": self.total_episodes_added,
            "total_steps_added": self.total_steps_added,
        }
