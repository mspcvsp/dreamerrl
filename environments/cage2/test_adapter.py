"""Regression checks for the real pinned wrapper and Dreamer's boundary contract."""

import sys

import torch

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv, boundary_flags
from dreamerrl.utils.types import EnvironmentConfig


def test_native_done_and_cap_are_not_conflated():
    assert boundary_flags(True, {"done": False}, 30, 30) == (True, False, True)
    assert boundary_flags(True, {"done": True}, 30, 30) == (True, True, False)
    assert boundary_flags(True, {}, 30, 30) == (True, False, True)
    assert boundary_flags(True, {}, 2, 30) == (True, True, False)


def test_real_blue_horizon_preserves_final_observation_and_resets_flags():
    env = Cage2BlueEnv(
        EnvironmentConfig(env_id="CAGE2-Blue-BLine-v0", num_envs=1, max_episode_steps=30, seed=0), torch.device("cpu")
    )
    first = env.reset(seed=0)
    assert first["is_first"].item() and first["state"].shape == (1, 52)
    assert env.native_dtype == "int64" and env.action_dim == 145
    monitor = next(row["index"] for row in env.action_mapping if row["name"] == "Monitor")
    for _ in range(30):
        last = env.step(torch.tensor([monitor]))
    assert last["is_last"].item() and last["is_first"].item() and not last["is_terminal"].item()
    assert last["info"]["done"] is False and last["info"]["adapter_truncated"]
    torch.testing.assert_close(last["transition_state"][0], torch.tensor(env.journal[-1]["observation_after"]))
    assert not torch.equal(last["transition_state"], last["state"])
    assert not last["prev_action"].any()
    env.step(torch.tensor([monitor]))
    assert env.journal[-1]["seed"] == 1 and env.journal[-1]["step"] == 1


def test_lean_learner_import_does_not_require_minihack():
    from dreamerrl.training.trainer import DreamerTrainer

    assert DreamerTrainer is not None
    assert "minihack" not in sys.modules
