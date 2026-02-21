from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class EnvInterface(ABC):
    """
    Unified interface for vectorized environments.

    Dreamer-native contract:
      - state-based observations
      - explicit episode boundary flags
    """

    # ---------------------------------------------------------
    # Core API
    # ---------------------------------------------------------

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Reset environment(s).

        Returns:
        {
            "state": Tensor (B, obs_dim)
            "reward": Tensor (B,)              # zeros
            "is_first": Tensor (B,)            # True
            "is_last": Tensor (B,)             # False
            "is_terminal": Tensor (B,)         # False
            "info": optional dict
        }
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Step environment(s).

        Args:
            actions: Tensor (B,) of discrete actions

        Returns:
        {
            "state": Tensor (B, obs_dim)
            "reward": Tensor (B,)
            "is_first": Tensor (B,)
            "is_last": Tensor (B,)
            "is_terminal": Tensor (B,)
            "info": optional dict
        }
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def batch_size(self) -> int:
        raise NotImplementedError

    # ---------------------------------------------------------
    # Optional: Action masking
    # ---------------------------------------------------------

    def action_mask(self) -> Optional[torch.Tensor]:
        return None

    # ---------------------------------------------------------
    # Optional: Episode stats
    # ---------------------------------------------------------

    def get_episode_stats(self) -> Dict[str, Any]:
        return {}
