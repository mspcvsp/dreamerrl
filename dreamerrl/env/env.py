from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch


class EnvInterface(ABC):
    """
    Unified interface for vectorized environments.
    All envs (PopGym, CAGE-2) must implement this.
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
                "obs": Tensor (B, obs_dim)
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
                "obs": Tensor (B, obs_dim)
                "reward": Tensor (B,)
                "done": Tensor (B,)
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
        """
        Returns Tensor (B, action_dim) or None.
        PopGym returns None.
        CAGE-2 returns legality mask.
        """
        return None

    # ---------------------------------------------------------
    # Optional: Episode stats
    # ---------------------------------------------------------
    def get_episode_stats(self) -> Dict[str, Any]:
        return {}
