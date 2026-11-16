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
    seen_lengths = set()
    seen_episodes = set()
    for seed in range(8):
        batch = replay.sample(32, seed=seed)
        length = batch["obs"].shape[1]
        seen_lengths.add(length)
        episode_ids = (batch["obs"][..., 0] // 1000).long()
        seen_episodes.update(episode_ids[:, 0].tolist())
        assert (episode_ids == episode_ids[:, :1]).all()
        assert (batch["obs"][:, 1:] - batch["obs"][:, :-1] == 1).all()
        assert not batch["is_last"][:, :-1].any()
        assert (batch["reward"] == 1).all()  # no padded or synthesized rows
        assert length == (30 if episode_ids[0, 0] == 1 else 50)
    assert seen_lengths == {30, 50}
    assert seen_episodes == {1, 2}


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


def add_episode(replay, episode_id, length):
    for t in range(length):
        replay.add(
            torch.tensor([[float(episode_id * 1000 + t)]]),
            torch.tensor([0]),
            torch.tensor([float(t)]),
            torch.tensor([t == length - 1]),
            torch.tensor([t == length - 1]),
        )


def test_bucket_sampling_preserves_episode_probabilities_and_long_windows():
    replay = ReplayBuffer(TrainingConfig(seq_len=5, burn_in=2), 1, 3, torch.device("cpu"))
    lengths = {1: 1, 2: 2, 3: 2, 4: 8, 5: 12}
    for episode_id, length in lengths.items():
        add_episode(replay, episode_id, length)
    counts = {episode_id: 0 for episode_id in lengths}
    mixed_long_batch = False
    for _ in range(600):
        batch = replay.sample(8)
        ids = (batch["obs"][:, 0, 0] // 1000).long().tolist()
        sampled_length = batch["obs"].shape[1]
        for episode_id in ids:
            counts[episode_id] += 1
            assert sampled_length == min(5, lengths[episode_id])
        if set(ids) == {4, 5}:
            mixed_long_batch = True
        # Burn-in still immediately precedes the supervised window, in its episode.
        for row in range(8):
            context = batch["burn_in_obs"][row][batch["burn_in_valid"][row]]
            if len(context):
                assert context[-1, 0] + 1 == batch["obs"][row, 0, 0]
    assert mixed_long_batch
    # Unequal bucket sizes (1, 2, 2 episodes) must not overweight small buckets.
    assert all(abs(count / 4800 - 0.2) < 0.04 for count in counts.values())


def test_bucket_cache_updates_on_addition_eviction_and_checkpoint_restore():
    cfg = TrainingConfig(seq_len=5, replay_capacity=8)
    replay = ReplayBuffer(cfg, 1, 3, torch.device("cpu"))
    add_episode(replay, 1, 2)
    assert replay.sample(2)["obs"].shape[1] == 2  # populate cached bucket
    add_episode(replay, 2, 6)
    assert any(replay.sample(2, seed=seed)["obs"].shape[1] == 5 for seed in range(8))
    add_episode(replay, 3, 1)  # capacity evicts episode 1; its bucket must disappear
    samples = [replay.sample(8, seed=seed) for seed in range(8)]
    assert {batch["obs"].shape[1] for batch in samples} == {1, 5}
    assert all((batch["obs"][..., 0] >= 2000).all() for batch in samples)
    checkpoint = replay.state_dict()
    restored = ReplayBuffer(cfg, 1, 3, torch.device("cpu"))
    add_episode(restored, 99, 3)
    restored.sample(1)  # force a cache with different episode layout
    restored.load_state_dict(checkpoint)
    for _ in range(4):
        expected, actual = replay.sample(4), restored.sample(4)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
