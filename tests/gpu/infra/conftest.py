import pytest
import torch

from lstmppo.trainer import LSTMPPOTrainer
from tests.helpers.fake_batch import make_fake_batch
from tests.helpers.fake_buffer_loader import load_rollout_into_buffer
from tests.helpers.fake_policy import make_fake_policy
from tests.helpers.fake_rollout import FakeRolloutBuilder
from tests.helpers.fake_state import make_fake_state


@pytest.fixture
def deterministic_trainer():
    trainer = LSTMPPOTrainer.for_validation()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer.policy.eval()
    trainer.state.cfg.trainer.debug_mode = True

    return trainer


# ------------------------------------------------------------
# GPU fake_state fixture
# ------------------------------------------------------------
@pytest.fixture
def fake_state():
    """
    Creates a TrainerStateProtocol on CUDA.
    """

    def _factory(
        rollout_steps=8,
        num_envs=2,
        obs_dim=4,
        hidden_size=4,
    ):
        state = make_fake_state(
            rollout_steps=rollout_steps,
            num_envs=num_envs,
            obs_dim=obs_dim,
            hidden_size=hidden_size,
        )
        return state

    return _factory


# ------------------------------------------------------------
# GPU fake_rollout fixture
# ------------------------------------------------------------
@pytest.fixture
def fake_rollout():
    """
    Builds a FakeRollout on CUDA using FakeRolloutBuilder.
    """

    def _factory(T=16, B=8, obs_dim=4, pattern="range", include_hidden=True, hidden_size=4):
        builder = FakeRolloutBuilder(T=T, B=B, obs_dim=obs_dim, device="cuda")
        builder = builder.with_pattern(pattern)
        if include_hidden:
            builder = builder.with_hidden(hidden_size)
        return builder.build()

    return _factory


# ------------------------------------------------------------
# GPU fake_buffer_loader fixture
# ------------------------------------------------------------
@pytest.fixture
def fake_buffer_loader(fake_state):
    """
    Loads a FakeRollout into a RecurrentRolloutBuffer on CUDA.
    """

    def _factory(rollout, state=None):
        if state is None:
            state = fake_state()
        return load_rollout_into_buffer(state, rollout, device="cuda")

    return _factory


# ------------------------------------------------------------
# GPU fake_batch fixture
# ------------------------------------------------------------
@pytest.fixture
def fake_batch(fake_state, fake_rollout):
    """
    Creates a PPO minibatch using make_fake_batch on CUDA.
    """

    def _factory(T=16, B=8, obs_dim=4):
        state = fake_state(
            rollout_steps=T,
            num_envs=B,
            obs_dim=obs_dim,
            hidden_size=4,
        )
        rollout = fake_rollout(T=T, B=B, obs_dim=obs_dim, include_hidden=True)
        batch = make_fake_batch(state, rollout, device="cuda")
        return batch

    return _factory


# ------------------------------------------------------------
# GPU fake_policy fixture
# ------------------------------------------------------------
@pytest.fixture
def fake_policy():
    """
    Creates a valid LSTMPPOPolicy on CUDA.
    """

    def _factory(
        rollout_steps=8,
        num_envs=2,
        obs_dim=4,
        hidden_size=4,
    ):
        policy = make_fake_policy(
            rollout_steps=rollout_steps,
            num_envs=num_envs,
            obs_dim=obs_dim,
            hidden_size=hidden_size,
        )
        return policy.to("cuda")

    return _factory
