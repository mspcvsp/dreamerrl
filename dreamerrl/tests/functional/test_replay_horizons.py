"""Short trials remain sampleable without padding or crossing episode ends."""

import torch

from dreamerrl.replay_buffer.replay_buffer import ReplayBuffer
from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import TrainingConfig


def test_mixed_trial_lengths_include_short_trials():
    replay = ReplayBuffer(TrainingConfig(seq_len=50), 1, 3, torch.device("cpu"))
    for episode, length in [(1, 30), (2, 100)]:
        for t in range(length):
            replay.add(
                torch.tensor([[float(episode * 1000 + t)]]),
                torch.tensor([0]),
                torch.tensor([1.0]),
                torch.tensor([t == length - 1]),
                torch.tensor([t == length - 1]),
            )
    batch = replay.sample(32, seed=0)
    assert batch["obs"].shape == (32, 30, 1)
    episode_ids = (batch["obs"][..., 0] // 1000).long()
    assert set(episode_ids[:, 0].tolist()) == {1, 2}
    assert (episode_ids == episode_ids[:, :1]).all()
    assert (batch["obs"][:, 1:] - batch["obs"][:, :-1] == 1).all()
    assert not batch["is_last"][:, :-1].any()
    assert (batch["reward"] == 1).all()  # no padded or synthesized rows


def test_single_step_episode_can_be_sampled():
    replay = ReplayBuffer(TrainingConfig(seq_len=16), 1, 3, torch.device("cpu"))
    replay.add(
        torch.tensor([[5.0]]), torch.tensor([1]), torch.tensor([-2.0]), torch.tensor([True]), torch.tensor([True])
    )
    batch = replay.sample(2)
    assert batch["obs"].shape == (2, 1, 1)
    assert batch["is_terminal"].all()


def test_collection_chunks_do_not_limit_replay_context(tmp_path):
    cfg = minihack_config(0, tmp_path)
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"
    cfg.env.max_episode_steps = 6
    cfg.train.collect_steps = 2
    cfg.train.seq_len = 16
    cfg.train.enforce_length_invariants = True
    trainer = DreamerTrainer(cfg)
    try:
        trainer.train(1)
        assert trainer.completed_updates == 1
        assert trainer.total_env_steps == 6 * cfg.env.num_envs
        assert cfg.train.seq_len == trainer.replay.seq_len == 16
        # Full six-action episode plus its initial observation, across three collections.
        batch = trainer.replay.sample(2)
        assert batch["obs"].shape[1] == 7
        assert batch["is_first"][:, 0].all()
        assert batch["is_last"][:, -1].all()
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
