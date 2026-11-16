"""Reward shaping affects replay, preserves reported task returns, and survives resume."""

import pytest
import torch

from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.trainer import DreamerTrainer


@pytest.mark.parametrize("cost", [0.0, 0.01])
def test_action_cost_shapes_replay_but_not_episode_metrics(tmp_path, monkeypatch, cost):
    cfg = minihack_config(0, tmp_path / "logs")
    cfg.train.action_cost = cost
    cfg.train.collect_steps = cfg.env.max_episode_steps
    trainer = DreamerTrainer(cfg)
    resumed = None
    try:
        original_step = trainer.env.step

        def step(actions):
            out = dict(original_step(actions))
            out["reward"] = torch.ones_like(out["reward"])
            return out

        monkeypatch.setattr(trainer.env, "step", step)
        episodes = []
        trainer.on_episode = episodes.append
        trainer.collect_env_steps()
        assert episodes
        # Every original reward is one, including the final transition.
        for episode in trainer.replay.episodes:
            assert episode["reward"][0] == 0  # Initial observation has no action cost.
            rewards = torch.as_tensor(episode["reward"][1:])
            torch.testing.assert_close(rewards, torch.full_like(rewards, 1.0 - cost))
        assert all(e["episode_return"] == cfg.env.max_episode_steps for e in episodes)
        checkpoint = tmp_path / "shaped.pt"
        trainer.save_checkpoint(checkpoint)
        resumed = DreamerTrainer.from_checkpoint(checkpoint, cuda=False, log_dir=tmp_path / "resumed")
        assert resumed.cfg.train.action_cost == cost
        for episode in resumed.replay.episodes:
            assert episode["reward"][0] == 0  # Initial observation has no action cost.
            rewards = torch.as_tensor(episode["reward"][1:])
            torch.testing.assert_close(rewards, torch.full_like(rewards, 1.0 - cost))
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
        if resumed is not None:
            resumed.env.venv.close()
            resumed.tb.close()


@pytest.mark.parametrize("cost", [-0.1, float("nan"), float("inf")])
def test_invalid_action_cost_rejected(tmp_path, cost):
    cfg = minihack_config(0, tmp_path)
    cfg.train.action_cost = cost
    with pytest.raises(ValueError, match="action_cost"):
        DreamerTrainer(cfg)
