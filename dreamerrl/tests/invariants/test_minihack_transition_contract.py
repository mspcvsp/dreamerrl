"""Both vector implementations must preserve the transition across autoreset."""

import importlib

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.replay_buffer.replay_buffer import ReplayBuffer
from dreamerrl.utils.types import EnvironmentConfig, TrainingConfig


class SentinelEnv(gym.Env):
    def __init__(self, index):
        self.index = index
        self.observation_space = gym.spaces.Dict({"glyphs": gym.spaces.Box(0, 100, (2, 2), np.int32)})
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"glyphs": np.full((2, 2), 1, dtype=np.int32)}, {}

    def step(self, action):
        return (
            {"glyphs": np.full((2, 2), 10 + self.index, dtype=np.int32)},
            float(self.index + 1),
            self.index == 0,
            self.index == 1,
            {},
        )


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("minihack_wrappers", "MiniHackVecEnv"),
        ("minihack_parallel_env", "MiniHackParallelEnv"),
    ],
)
def test_final_observation_and_independent_reset(module_name, class_name, monkeypatch):
    module = importlib.import_module("dreamerrl.env.minihack." + module_name)
    monkeypatch.setattr(module, "make_mhack_env", lambda cfg, i: lambda: SentinelEnv(i))
    env = getattr(module, class_name)(EnvironmentConfig(num_envs=3), torch.device("cpu"))
    try:
        initial = env.reset()["state"]
        out = env.step(torch.tensor([0, 1, 2]))
        assert out["is_last"].tolist() == [True, True, False]
        assert out["is_terminal"].tolist() == [True, False, False]
        assert out["is_first"].tolist() == [True, True, False]
        torch.testing.assert_close(out["transition_state"][:, 0], torch.tensor([10.0, 11.0, 12.0]))
        torch.testing.assert_close(out["state"][:, 0], torch.tensor([1.0, 1.0, 12.0]))
        torch.testing.assert_close(out["prev_action"].sum(-1), torch.tensor([0.0, 0.0, 1.0]))
        replay = ReplayBuffer(TrainingConfig(seq_len=2), 4, 3, torch.device("cpu"))
        replay.add(
            out["transition_state"],
            torch.tensor([0, 1, 2]),
            out["reward"],
            out["is_last"],
            out["is_terminal"],
            initial_obs=initial,
        )
        for i, ep in enumerate(replay.episodes):
            assert ep["is_first"].tolist() == [True, False]
            assert ep["is_terminal"].tolist() == [False, i == 0]
            assert (ep["obs"][0] == 1).all()
            assert (ep["obs"][-1] == 10 + i).all()
        batch = replay.sample(4)
        assert batch["is_first"][:, 0].all()
        assert (batch["obs"][:, -1, 0] >= 10).all()
        # No phantom next-step reset: these actions are executed immediately.
        out2 = env.step(torch.tensor([0, 1, 2]))
        assert out2["reward"].tolist() == [1.0, 2.0, 3.0]
    finally:
        env.venv.close()


def test_glyph_extraction_preserves_vector_batch_axis():
    from dreamerrl.env.minihack.minihack_utils import extract_glyphs

    single = {"glyphs": np.zeros((21, 79))}
    batched = {"glyphs": np.zeros((3, 21, 79))}
    assert extract_glyphs(single).shape == (1, 21, 79)
    assert extract_glyphs(batched).shape == (3, 21, 79)


def test_replay_final_window_is_sampleable():
    replay = ReplayBuffer(TrainingConfig(seq_len=2), 1, 3, torch.device("cpu"))
    for t in range(3):
        replay.add(
            torch.tensor([[float(t)]]),
            torch.tensor([0]),
            torch.tensor([0.0]),
            torch.tensor([t == 2]),
            torch.tensor([t == 2]),
        )
    batch = replay.sample(32, seed=123)
    assert set(batch["obs"][:, -1, 0].tolist()) == {1.0, 2.0}
    assert batch["is_terminal"][:, -1].any()
