import gymnasium as gym
import pytest
import torch


def pytest_runtest_setup(item):
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("Skipping GPU test because CUDA is not available")


def require_popgym_env(env_id: str):
    """Skip test if the PopGym environment is not installed."""
    registered = [e.id for e in gym.envs.registry.values()]
    if env_id not in registered:
        pytest.skip(f"PopGym environment not installed: {env_id}")
