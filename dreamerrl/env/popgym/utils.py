import gymnasium as gym

import popgym  # noqa: F401  # ensures PopGym registers its envs


def list_popgym_envs() -> list[str]:
    """
    Return all registered PopGym environment IDs.

    Gymnasium >=0.29 exposes the registry via gym.registry (public API).
    """
    return sorted([eid for eid in gym.registry.keys() if "popgym" in eid.lower()])
