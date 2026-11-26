"""Episode and model contracts for fixed-history diagnostic inference."""
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class Episode:
    """Decision frames. Action t enters frame t+1, never frame t."""

    episode_id: Hashable
    group_id: Hashable
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: bool
    truncated: bool

    def __post_init__(self):
        length = len(self.observations)
        if not length or self.actions.shape != (length,) or self.rewards.shape != (length,):
            raise ValueError("Decision frames, outgoing actions and rewards must align")


@dataclass(frozen=True)
class Labels:
    """Offline targets and validity, keyed independently of policy frames."""

    values: np.ndarray
    valid: np.ndarray


class EnvironmentAdapter(Protocol):
    def collect(self, episode_ids: Sequence[Hashable]) -> Sequence[Episode]: ...
    def labels(self, episode_id: Hashable) -> Labels: ...
    def model_input(self, observations: torch.Tensor) -> torch.Tensor: ...


class ModelAdapter(Protocol):
    def initial_state(self, batch: int) -> Any: ...
    def filter_step(self, state, obs, incoming, first, *, mode="argmax", rng=None, noise=None): ...
    def imagine_step(self, state, action, *, mode="argmax", rng=None): ...
    def readouts(self, state) -> dict[str, torch.Tensor]: ...


def paired_episodes(left: Episode, right: Episode, intervention_frame: int) -> None:
    """Validate the history held fixed in a controlled observation intervention."""
    if left.group_id != right.group_id or left.observations.shape != right.observations.shape:
        raise ValueError("Matched histories require a common group and frame shape")
    frames = torch.arange(len(left.observations)) != intervention_frame
    if not torch.equal(left.observations[frames], right.observations[frames]):
        raise ValueError("Matched histories differ outside the intervention frame")
    if not torch.equal(left.actions[:-1], right.actions[:-1]):
        raise ValueError("Matched histories have different pre-decision actions")
