import pytest


def require_popgym_env(env_id: str) -> None:
    import gymnasium as gym

    registered = [e.id for e in gym.envs.registry.values()]
    if env_id not in registered:
        pytest.skip(f"PopGym environment not installed: {env_id}")
