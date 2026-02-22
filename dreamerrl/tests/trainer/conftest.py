import pytest

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import DreamerConfig


@pytest.fixture(scope="session")
def cfg():
    # Use a lightweight Dreamer-Lite config for tests
    return DreamerConfig(mode="lite")


@pytest.fixture
def trainer(cfg):
    # Trainer builds its own env, world model, actor, critic, replay, etc.
    return DreamerTrainer(cfg)
