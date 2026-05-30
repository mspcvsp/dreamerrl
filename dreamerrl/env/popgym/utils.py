import gymnasium as gym

import popgym  # noqa: F401  # ensures PopGym registers its envs


def list_popgym_envs() -> list[str]:
    """
    Return all registered PopGym environment IDs.

    Gymnasium >=0.29 exposes the registry via gym.registry (public API).
    """
    return sorted([eid for eid in gym.registry.keys() if "popgym" in eid.lower()])


if __name__ == "__main__":
    print("Registered PopGym environments:")
    for env_id in list_popgym_envs():
        print(f"  - {env_id}")
