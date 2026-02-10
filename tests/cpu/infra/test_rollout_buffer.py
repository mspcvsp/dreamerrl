"""
CPU-side validation for RecurrentRolloutBuffer.

This suite enforces the structural invariants required for correct PPO rollouts:

- shape regressions
- pointer/indexing bugs
- mask logic correctness
- device/dtype drift
- correct initialization/reset behavior
- RolloutStep structure and copy semantics
- gate detachment invariants

These tests ensure that the rollout buffer remains a stable, predictable
data structure across refactors. Any violation here silently breaks PPO,
TBPTT, or auxiliary prediction alignment.
"""

import pytest
import torch

from lstmppo.buffer import RecurrentRolloutBuffer
from lstmppo.trainer_state import TrainerState
from lstmppo.types import LSTMGates, RolloutStep

pytestmark = pytest.mark.infra


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def _make_buffer(trainer_state: TrainerState):
    trainer_state.cfg.trainer.cuda = False  # force CPU for tests
    device = torch.device("cpu")
    return trainer_state.cfg, device, RecurrentRolloutBuffer(trainer_state, device)


def make_step(B: int, D: int, H: int) -> RolloutStep:
    """Factory for a fully-populated RolloutStep with correct shapes."""
    obs = torch.randn(B, D)
    act = torch.randn(B)
    rew = torch.randn(B)
    val = torch.randn(B)
    logp = torch.randn(B)
    term = torch.zeros(B, dtype=torch.bool)
    trunc = torch.zeros(B, dtype=torch.bool)
    hxs = torch.randn(B, H)
    cxs = torch.randn(B, H)

    gates = LSTMGates(
        i_gates=hxs,
        f_gates=hxs,
        g_gates=hxs,
        o_gates=hxs,
        c_gates=hxs,
        h_gates=hxs,
    )

    return RolloutStep(
        obs=obs,
        actions=act,
        rewards=rew,
        values=val,
        logprobs=logp,
        terminated=term,
        truncated=trunc,
        hxs=hxs,
        cxs=cxs,
        gates=gates,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_buffer_initialization_shapes(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    cfg, _, buf = _make_buffer(trainer_state)

    T = cfg.trainer.rollout_steps
    B = cfg.env.num_envs
    D = trainer_state.env_info.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    expected_shapes = {
        "obs": (T, B, D),
        "actions": (T, B, 1),
        "rewards": (T, B),
        "values": (T, B),
        "logprobs": (T, B),
        "terminated": (T, B),
        "truncated": (T, B),
        "hxs": (T, B, H),
        "cxs": (T, B, H),
        "mask": (T, B),
    }

    for name, shape in expected_shapes.items():
        assert getattr(buf, name).shape == shape


def test_buffer_device_and_dtype(trainer_state: TrainerState):
    _, device, buf = _make_buffer(trainer_state)

    # dtype invariants
    assert buf.obs.dtype == torch.float32
    assert buf.rewards.dtype == torch.float32
    assert buf.values.dtype == torch.float32
    assert buf.logprobs.dtype == torch.float32
    assert buf.terminated.dtype == torch.bool
    assert buf.truncated.dtype == torch.bool
    assert buf.hxs.dtype == torch.float32
    assert buf.cxs.dtype == torch.float32

    # device invariants
    for t in [buf.obs, buf.actions, buf.rewards, buf.values, buf.logprobs, buf.terminated, buf.truncated, buf.mask]:
        assert t.device.type == device.type


def test_add_increments_pointer_and_writes_data(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    cfg, _, buf = _make_buffer(trainer_state)

    B = cfg.env.num_envs
    D = trainer_state.env_info.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    step = make_step(B, D, H)
    buf.add(step)

    assert buf.step == 1
    assert torch.allclose(buf.obs[0], step.obs)
    assert torch.allclose(buf.actions[0].squeeze(-1), step.actions)

    # Ensure data is copied, not referenced
    assert buf.obs[0].data_ptr() != step.obs.data_ptr()


def test_fill_buffer_reaches_full_pointer(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    cfg, _, buf = _make_buffer(trainer_state)

    B = cfg.env.num_envs
    D = trainer_state.env_info.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    for _ in range(cfg.trainer.rollout_steps):
        buf.add(make_step(B, D, H))

    assert buf.step == cfg.trainer.rollout_steps


def test_pointer_does_not_wrap(trainer_state: TrainerState):
    """
    The rollout buffer pointer must:
    - advance from 0 to rollout_steps - 1
    - stop exactly at rollout_steps
    - raise AssertionError on any further write
    - never wrap around to 0
    """

    assert trainer_state.env_info is not None
    cfg, _, buf = _make_buffer(trainer_state)

    B = cfg.env.num_envs
    D = trainer_state.env_info.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    # Fill the buffer exactly to capacity
    for _ in range(cfg.trainer.rollout_steps):
        buf.add(make_step(B, D, H))

    # Pointer should now be exactly at rollout_steps
    assert buf.step == cfg.trainer.rollout_steps

    # Any further add must raise overflow
    with pytest.raises(AssertionError):
        buf.add(make_step(B, D, H))

    # Pointer must remain clamped, never wrap
    assert buf.step == cfg.trainer.rollout_steps


def test_mask_logic_cpu(trainer_state: TrainerState):
    _, _, buf = _make_buffer(trainer_state)

    buf.terminated[0] = True
    buf.truncated[1] = True

    assert (buf.mask[0] == 0).all()
    assert (buf.mask[1] == 0).all()
    assert (buf.mask[2:] == 1).all()


def test_reset_clears_state(trainer_state: TrainerState):
    _, _, buf = _make_buffer(trainer_state)

    buf.terminated[0] = True
    buf.truncated[0] = True
    buf.obs[0] = 123.0  # ensure reset clears data

    buf.reset()

    assert buf.step == 0
    assert (buf.terminated == 0).all()
    assert (buf.truncated == 0).all()
    assert (buf.mask == 1).all()
    assert (buf.obs == 0).all()  # full clear invariant


def test_rollout_step_structure(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    cfg, _, buf = _make_buffer(trainer_state)

    B = cfg.env.num_envs
    D = trainer_state.env_info.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    step = make_step(B, D, H)
    buf.add(step)

    assert buf.obs[0].shape == (B, D)
    assert buf.actions[0].shape == (B, 1)
    assert buf.rewards[0].shape == (B,)
    assert buf.hxs[0].shape == (B, H)
    assert buf.cxs[0].shape == (B, H)
