"""Cadence uses observed collection, valid context, and completed optimizer calls."""

import pytest
import torch

from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.trainer import DreamerTrainer


def cadence_config(tmp_path):
    cfg = minihack_config(0, tmp_path)
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"
    cfg.env.max_episode_steps = 6
    cfg.train.collect_steps = 2
    cfg.train.seq_len = 4
    cfg.train.burn_in = 2
    cfg.train.freeze_actor_critic_steps = 1
    return cfg


def close(trainer):
    trainer.env.venv.close()
    trainer.tb.close()


def test_cadence_counts_warmup_burn_in_and_frozen_actor(tmp_path, monkeypatch):
    trainer = DreamerTrainer(cadence_config(tmp_path))
    records, batches = [], []
    original = trainer.replay.sample

    def sample(*args, **kwargs):
        batch = original(*args, **kwargs)
        batches.append(batch)
        return batch

    monkeypatch.setattr(trainer.replay, "sample", sample)
    try:
        trainer.train(2, on_update=lambda record: records.append(record))
        assert [r["cadence"]["env_transitions"] for r in records] == [12, 4]
        for update, (record, batch) in enumerate(zip(records, batches)):
            cadence = record["cadence"]
            tokens = batch["obs"].shape[0] * batch["obs"].shape[1]
            burn_in = int(batch["burn_in_valid"].sum().item())
            assert cadence["replay_tokens"] == tokens == 8
            assert cadence["model_burn_in_tokens"] == burn_in
            assert cadence["actor_critic_burn_in_tokens"] == (burn_in if update else 0)
            context_tokens = batch["burn_in_obs"].shape[0] * batch["burn_in_obs"].shape[1]
            passes = 2 if update else 1
            assert cadence["posterior_tokens"] == (tokens + context_tokens) * passes
            assert cadence["burn_in_padding_tokens"] == (context_tokens - burn_in) * passes
            assert cadence["world_model_optimizer_steps"] == 1
            assert cadence["actor_optimizer_steps"] == cadence["critic_optimizer_steps"] == update
            assert cadence["imagination_starts"] == tokens * update
            assert cadence["imagined_transitions"] == tokens * update * trainer.cfg.world.imagination_horizon
            assert cadence["replay_tokens_per_env_transition"] == tokens / cadence["env_transitions"]
            assert record["model_tokens_per_env_step"] == cadence["replay_tokens_per_env_transition"]
            for key in ("collection_seconds", "replay_sample_seconds", "world_model_seconds", "actor_critic_seconds"):
                assert 0 <= cadence[key] <= cadence["update_seconds"]
        totals = records[-1]["cadence_total"]
        assert totals["env_transitions"] == 16
        assert totals["replay_tokens"] == 16
        assert totals["replay_tokens_per_env_transition"] == 1
        assert totals["world_model_optimizer_steps"] == 2
        assert totals["actor_optimizer_steps"] == totals["critic_optimizer_steps"] == 1
        # Verify counters against independent optimizer state, not scheduled updates.
        for optimizer, steps in ((trainer.model_opt, 2), (trainer.actor_opt, 1), (trainer.critic_opt, 1)):
            assert max(float(state["step"]) for state in optimizer.state.values()) == steps
    finally:
        close(trainer)


@pytest.mark.parametrize("legacy", [False, True])
def test_cadence_totals_resume_without_inventing_legacy_history(tmp_path, legacy):
    trainer = DreamerTrainer(cadence_config(tmp_path))
    resumed = None
    try:
        trainer.train(2)
        path = tmp_path / "cadence.pt"
        trainer.save_checkpoint(path)
        expected = dict(trainer.cadence_totals)
        if legacy:
            checkpoint = torch.load(path, weights_only=False)
            del checkpoint["cadence_totals"]
            torch.save(checkpoint, path)
        resumed = DreamerTrainer.from_checkpoint(path, cuda=False)
        assert resumed.cadence_totals == ({} if legacy else expected)
        records = []
        resumed.train(3, on_update=lambda record: records.append(record))
        cadence, total = records[0]["cadence"], records[0]["cadence_total"]
        for key in ("env_transitions", "replay_tokens", "world_model_optimizer_steps", "actor_optimizer_steps"):
            assert total[key] == cadence[key] + (0 if legacy else expected[key])
    finally:
        close(trainer)
        if resumed is not None:
            close(resumed)


def test_skipped_optimizer_call_does_not_count_as_update(tmp_path, monkeypatch):
    cfg = cadence_config(tmp_path)
    trainer = DreamerTrainer(cfg)
    monkeypatch.setattr(trainer.model_opt, "step", lambda *args, **kwargs: None)
    try:
        records = []
        trainer.train(1, on_update=lambda record: records.append(record))
        cadence = records[0]["cadence"]
        assert cadence["world_model_optimizer_steps"] == 0
        assert cadence["world_model_updates_per_env_transition"] == 0
        assert cadence["replay_tokens"] > 0
    finally:
        close(trainer)


def test_unsupported_update_setting_is_not_silently_ignored(tmp_path):
    cfg = cadence_config(tmp_path)
    cfg.train.updates_per_step = 2
    with pytest.raises(ValueError, match="one update per collection cycle"):
        DreamerTrainer(cfg)
