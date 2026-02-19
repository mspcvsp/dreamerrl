import pytest

from tests.helpers.fake_batch import make_fake_batch
from tests.helpers.fake_buffer_loader import load_rollout_into_buffer
from tests.helpers.fake_rollout import FakeRolloutBuilder
from tests.helpers.fake_state import make_fake_state


# ------------------------------------------------------------
# Fake rollout fixture (time-major, CPU)
# ------------------------------------------------------------
@pytest.fixture
def fake_rollout():
    def _factory(
        T: int = 8,
        B: int = 2,
        obs_dim: int = 4,
        pattern: str = "range",
    ):
        builder = FakeRolloutBuilder(T, B, obs_dim, device="cpu")
        builder = builder.with_pattern(pattern)
        return builder.build()

    return _factory


# ------------------------------------------------------------
# Fake state fixture (CPU)
# ------------------------------------------------------------
@pytest.fixture
def fake_state():
    def _factory(
        rollout_steps: int = 8,
        num_envs: int = 2,
        obs_dim: int = 4,
        hidden_size: int = 4,
    ):
        return make_fake_state(
            rollout_steps=rollout_steps,
            num_envs=num_envs,
            obs_dim=obs_dim,
            hidden_size=hidden_size,
        )

    return _factory


# ------------------------------------------------------------
# Fake buffer loader fixture (CPU)
# ------------------------------------------------------------
@pytest.fixture
def fake_buffer_loader(fake_state, fake_rollout):
    def _factory(
        T: int = 8,
        B: int = 2,
        obs_dim: int = 4,
        pattern: str = "range",
    ):
        state = fake_state(
            rollout_steps=T,
            num_envs=B,
            obs_dim=obs_dim,
        )
        rollout = fake_rollout(T=T, B=B, obs_dim=obs_dim, pattern=pattern)
        return load_rollout_into_buffer(state, rollout, device="cpu")

    return _factory


# ------------------------------------------------------------
# Fake batch fixture (CPU)
# ------------------------------------------------------------
@pytest.fixture
def fake_batch(fake_state, fake_rollout):
    def _factory(
        T: int = 8,
        B: int = 2,
        obs_dim: int = 4,
        pattern: str = "range",
    ):
        state = fake_state(
            rollout_steps=T,
            num_envs=B,
            obs_dim=obs_dim,
        )
        rollout = fake_rollout(T=T, B=B, obs_dim=obs_dim, pattern=pattern)
        return make_fake_batch(state, rollout, device="cpu")

    return _factory
