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


@pytest.fixture
def trainer_state(deterministic_trainer):
    """GPU GAE test wants a TrainerState, mirror CPU test_gae_computation_basic."""
    return deterministic_trainer.state


@pytest.fixture
def fake_state():
    def _factory(rollout_steps=8, num_envs=2, obs_dim=4, hidden_size=4):
        return make_fake_state(
            rollout_steps=rollout_steps,
            num_envs=num_envs,
            obs_dim=obs_dim,
            hidden_size=hidden_size,
        )

    return _factory


@pytest.fixture
def fake_rollout():
    def _factory(
        T=None,
        B=None,
        obs_dim=4,
        pattern="range",
        include_hidden=True,
        hidden_size=4,
        device="cuda",
        seq_len=None,
        batch_size=None,
        force_done_at=None,
    ):
        if seq_len is not None:
            T = seq_len
        if batch_size is not None:
            B = batch_size
        if T is None:
            T = 16
        if B is None:
            B = 8

        builder = FakeRolloutBuilder(T=T, B=B, obs_dim=obs_dim, device=device)
        builder = builder.with_pattern(pattern)
        if include_hidden:
            builder = builder.with_hidden(hidden_size)

        rollout = builder.build()

        if force_done_at is not None:
            # masks: 1.0 alive, 0.0 dead
            rollout.masks[force_done_at:] = 0.0

        return rollout

    return _factory


@pytest.fixture
def fake_buffer_loader(fake_state):
    def _factory(
        rollout,
        device="cuda",
        state=None,
        chunk_size=None,  # accepted for TBPTT test, but not used here
    ):
        if state is None:
            state = fake_state(
                rollout_steps=rollout.obs.shape[0],
                num_envs=rollout.obs.shape[1],
                obs_dim=rollout.obs.shape[2],
                hidden_size=4,
            )
        return load_rollout_into_buffer(state, rollout, device=device)

    return _factory


@pytest.fixture
def fake_batch(fake_state, fake_rollout):
    def _factory(
        batch_size=8,
        seq_len=16,
        obs_dim=4,
        device="cuda",
    ):
        state = fake_state(
            rollout_steps=seq_len,
            num_envs=batch_size,
            obs_dim=obs_dim,
            hidden_size=4,
        )
        rollout = fake_rollout(
            batch_size=batch_size,
            seq_len=seq_len,
            obs_dim=obs_dim,
            device=device,
        )
        return make_fake_batch(state, rollout, device=device)

    return _factory


@pytest.fixture
def fake_policy():
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
