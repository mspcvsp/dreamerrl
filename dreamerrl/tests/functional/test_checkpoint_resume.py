"""Checkpoint round trips preserve updates, optimizers, replay and RNG state."""

import random

import numpy as np
import pytest
import torch

from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.trainer import DreamerTrainer


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def close(trainer):
    trainer.env.venv.close()
    trainer.tb.close()


def test_checkpoint_roundtrip_and_next_optimizer_update(tmp_path):
    cfg = minihack_config(11, tmp_path / "logs")
    cfg.train.collect_steps = 7  # leave unfinished episodes at the checkpoint
    trainer = DreamerTrainer(cfg)
    resumed = None
    checkpoint = tmp_path / "checkpoints" / "latest.pt"
    try:
        trainer.train(2, checkpoint_path=checkpoint, checkpoint_every=10)  # final save, even off interval
        assert checkpoint.exists()
        assert any(ep.data["obs"] for ep in trainer.replay.current_eps)
        expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
        batch = trainer.replay.sample(2)
        resumed = DreamerTrainer.from_checkpoint(checkpoint, cuda=False)
        actual_rng = (random.random(), float(np.random.random()), torch.rand(3))
        assert_state_equal(expected_rng, actual_rng)
        assert resumed.completed_updates == resumed.total_updates_target == 2
        assert resumed.total_env_steps == trainer.total_env_steps
        assert resumed.sample_step == trainer.sample_step
        assert resumed.recent_returns == trainer.recent_returns
        assert resumed.replay.current_eps == []
        assert resumed.replay.num_envs is None
        assert resumed.episode_returns.count_nonzero() == 0
        assert_state_equal(trainer.replay.episodes, resumed.replay.episodes)
        assert_state_equal(batch, resumed.replay.sample(2))
        for name in ["world", "actor", "critic", "slow_critic", "model_opt", "actor_opt", "critic_opt"]:
            assert_state_equal(getattr(trainer, name).state_dict(), getattr(resumed, name).state_dict())
        # Same fixed replay batch and RNG must give the same next optimizer step.
        rng = torch.get_rng_state()
        a = trainer.update_world_model(batch, 2).total_loss.item()
        b, c = trainer.update_actor_critic(batch, 2)
        torch.set_rng_state(rng)
        x = resumed.update_world_model(batch, 2).total_loss.item()
        y, z = resumed.update_actor_critic(batch, 2)
        assert (a, b, c) == (x, y, z)
        for name in ["world", "actor", "critic", "slow_critic", "model_opt", "actor_opt", "critic_opt"]:
            assert_state_equal(getattr(trainer, name).state_dict(), getattr(resumed, name).state_dict())
        old_steps = resumed.total_env_steps
        resumed.train(4, checkpoint_path=checkpoint, checkpoint_every=1)
        assert resumed.completed_updates == resumed.sample_step == 4
        assert resumed.total_env_steps == old_steps + 2 * cfg.train.collect_steps * cfg.env.num_envs
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert saved["completed_updates"] == saved["total_updates_target"] == 4
    finally:
        close(trainer)
        if resumed is not None:
            close(resumed)


def test_failed_checkpoint_write_keeps_previous_file(tmp_path, monkeypatch):
    trainer = DreamerTrainer(minihack_config(0, tmp_path / "logs"))
    try:
        path = tmp_path / "latest.pt"
        trainer.save_checkpoint(path)
        original = path.read_bytes()

        def fail(*args, **kwargs):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(torch, "save", fail)
        with pytest.raises(OSError, match="simulated disk failure"):
            trainer.save_checkpoint(path)
        assert path.read_bytes() == original
        assert list(tmp_path.glob("*.tmp")) == []
    finally:
        close(trainer)


def test_checkpoint_rejects_mismatched_configuration(tmp_path):
    trainer = DreamerTrainer(minihack_config(0, tmp_path / "logs"))
    try:
        path = tmp_path / "latest.pt"
        trainer.save_checkpoint(path)
        trainer.cfg.env.env_id = "MiniHack-LockedDoor-v0"
        with pytest.raises(ValueError, match="configuration"):
            trainer.load_checkpoint(path)
    finally:
        close(trainer)


def test_periodic_checkpoint_resumes_original_schedule(tmp_path, monkeypatch):
    cfg = minihack_config(0, tmp_path / "logs")
    cfg.train.warmup_steps = 0
    trainer = DreamerTrainer(cfg)
    resumed = None
    path = tmp_path / "latest.pt"
    calls = 0
    original_collect = trainer.collect_env_steps

    def interrupted_collect():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        original_collect()

    monkeypatch.setattr(trainer, "collect_env_steps", interrupted_collect)
    try:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            trainer.train(4, checkpoint_path=path, checkpoint_every=1)
        resumed = DreamerTrainer.from_checkpoint(path, cuda=False)
        assert resumed.completed_updates == 1
        assert resumed.total_updates_target == 4
        resumed.train(4, checkpoint_path=path, checkpoint_every=1)
        assert resumed.completed_updates == resumed.sample_step == 4
        from dreamerrl.training.trainer import CosineWarmupScheduler
        from dreamerrl.utils.types import LRScheduleConfig

        expected_lr = CosineWarmupScheduler(
            LRScheduleConfig(
                base_lr=cfg.train.model_lr,
                warmup_steps=0,
                total_steps=4,
                lr_floor=0.1,
            )
        )(3)
        assert resumed.model_opt.param_groups[0]["lr"] == expected_lr
    finally:
        close(trainer)
        if resumed is not None:
            close(resumed)


def test_mlp_checkpoint_is_rejected_before_loading(tmp_path):
    trainer = DreamerTrainer(minihack_config(0, tmp_path / "logs"))
    try:
        path = tmp_path / "old_mlp.pt"
        trainer.save_checkpoint(path)
        payload = torch.load(path, weights_only=False, map_location="cpu")
        payload["version"] = 3
        torch.save(payload, path)
        before = {key: value.clone() for key, value in trainer.world.state_dict().items()}
        with pytest.raises(ValueError, match="MLP-RSSM checkpoint v3"):
            trainer.load_checkpoint(path)
        with pytest.raises(ValueError, match="MLP-RSSM checkpoint v3"):
            DreamerTrainer.from_checkpoint(path, cuda=False)
        assert_state_equal(before, trainer.world.state_dict())
    finally:
        close(trainer)
