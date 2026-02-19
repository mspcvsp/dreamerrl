import gymnasium as gym
import pytest


@pytest.fixture
def require_popgym_env():
    def _check(env_id: str):
        registered = [e.id for e in gym.envs.registry.values()]
        if env_id not in registered:
            pytest.skip(f"PopGym environment not installed: {env_id}")

    return _check
