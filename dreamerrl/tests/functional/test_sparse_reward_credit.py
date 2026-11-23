"""Recorded reward alignment, delayed credit, boundary masks, and gradient isolation."""

import importlib

import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.replay_buffer.replay_buffer import ReplayBuffer
from dreamerrl.training.core.actor_critic_update import actor_critic_update
from dreamerrl.training.core.replay_value import replay_value_targets
from dreamerrl.utils.types import TrainingConfig


def test_delayed_recorded_reward_reaches_earlier_states():
    # reward[t] resulted in obs[t]; the initial dummy reward must be ignored.
    batch = dict(reward=torch.tensor([[999.0, 0.0, 0.0, 8.0]]), is_terminal=torch.tensor([[False, False, False, True]]))
    targets, valid = replay_value_targets(batch, torch.zeros(1, 4), 0.5, 1)
    torch.testing.assert_close(targets, torch.tensor([[2.0, 4.0, 8.0]]))
    assert valid.all()
    targets, _ = replay_value_targets(batch, torch.zeros(1, 4), 0.5, 0)
    torch.testing.assert_close(targets, torch.tensor([[0.0, 0.0, 8.0]]))


def test_terminal_cuts_bootstrap_but_truncation_keeps_final_value():
    batch = dict(
        reward=torch.tensor([[0.0, 2.0], [0.0, 2.0]]),
        is_terminal=torch.tensor([[False, True], [False, False]]),
        is_last=torch.tensor([[False, True], [False, True]]),
    )
    targets, valid = replay_value_targets(batch, torch.tensor([[0.0, 100.0], [0.0, 100.0]]), 0.5, 1)
    torch.testing.assert_close(targets, torch.tensor([[2.0], [52.0]]))
    assert valid.all()


def test_reset_crossings_do_not_leak_future_episode_rewards():
    batch = dict(
        reward=torch.tensor([[0.0, 2.0, 0.0, 1000.0]]),
        is_last=torch.tensor([[False, True, False, True]]),
        is_terminal=torch.zeros(1, 4, dtype=torch.bool),
        is_first=torch.tensor([[True, False, True, False]]),
    )
    targets, valid = replay_value_targets(batch, torch.tensor([[0.0, 10.0, 0.0, 0.0]]), 0.5, 1)
    torch.testing.assert_close(targets[:, 0], torch.tensor([7.0]))
    assert valid.tolist() == [[True, False, True]]


def test_replay_critic_loss_uses_recorded_rewards_and_isolates_gradients(world_model, monkeypatch):
    module = importlib.import_module("dreamerrl.training.core.actor_critic_update")
    actor = Actor(latent=world_model.latent, net=world_model.net_cfg)
    critic = ValueHead(latent=world_model.latent, net=world_model.net_cfg)
    world_model.posterior.deterministic_latent_for_tests = True
    world_model.prior.deterministic_latent_for_tests = True
    batch = dict(
        obs=torch.randn(1, 4, world_model.flat_obs_dim),
        action=torch.zeros(1, 4, dtype=torch.long),
        reward=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        is_terminal=torch.tensor([[False, False, False, True]]),
    )
    original = module.imagine_trajectory_for_training

    def imagine(*args):
        trajectory = original(*args)
        trajectory["reward"].zero_()  # critic supervision must not depend on imagined reward
        return trajectory

    monkeypatch.setattr(module, "imagine_trajectory_for_training", imagine)
    diagnostics = {}
    rng = torch.get_rng_state()
    a0, c0 = actor_critic_update(world_model, actor, critic, batch, 2, 0.99, 0.95)
    torch.set_rng_state(rng)
    a1, c1 = actor_critic_update(
        world_model, actor, critic, batch, 2, 0.99, 0.95, replay_value_scale=1.0, diagnostics=diagnostics
    )
    torch.testing.assert_close(a0, a1)
    torch.testing.assert_close(c1 - c0, diagnostics["replay_value_loss"])
    assert diagnostics["replay_value_targets"] == 3
    assert diagnostics["imagined_reward_mean"] == 0
    (c1 - c0).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())
    assert all(p.grad is None for p in world_model.parameters())
    assert all(p.grad is None for p in actor.parameters())


def test_positive_reward_coverage_tracks_eviction_and_restore():
    replay = ReplayBuffer(TrainingConfig(replay_capacity=4, seq_len=2), 1, 2, torch.device("cpu"))

    def episode(reward):
        for t in range(2):
            replay.add(
                torch.tensor([[float(t)]]),
                torch.tensor([0]),
                torch.tensor([reward if t else 0.0]),
                torch.tensor([t == 1]),
                torch.tensor([t == 1]),
            )

    episode(1.0)
    episode(-1.0)
    assert replay.positive_reward_steps == replay.positive_reward_episodes == 1
    restored = ReplayBuffer(replay.cfg, 1, 2, torch.device("cpu"))
    restored.load_state_dict(replay.state_dict())
    assert restored.positive_reward_steps == restored.positive_reward_episodes == 1
    episode(0.0)  # evicts the positive episode
    assert replay.positive_reward_steps == replay.positive_reward_episodes == 0


@pytest.mark.parametrize("length", [1, 3])
def test_no_valid_replay_targets_produce_zero_loss(world_model, length):
    from dreamerrl.training.core.replay_value import replay_value_loss

    critic = ValueHead(latent=world_model.latent, net=world_model.net_cfg)
    logits = torch.randn(2, length, critic.value_bins, requires_grad=True)
    batch = dict(reward=torch.zeros(2, length), is_terminal=torch.ones(2, length, dtype=torch.bool))
    loss, metrics = replay_value_loss(critic, logits, batch, 0.99, 0.95)
    loss.backward()
    assert loss == 0 and metrics == {}
    assert logits.grad is not None and logits.grad.count_nonzero() == 0


def test_trainer_reports_credit_metrics_and_resumes_replay_critic(tmp_path):
    from dreamerrl.tests.manual.utils import minihack_config
    from dreamerrl.training.trainer import DreamerTrainer

    cfg = minihack_config(0, tmp_path)
    cfg.ac.replay_value_scale = 1
    cfg.ac.discount = 0.997
    records = []
    trainer = DreamerTrainer(cfg)
    restored = None
    try:
        trainer.train(1, on_update=lambda record: records.append(record))
        record = records[0]
        assert record["credit_assignment"]["replay_value_targets"] > 0
        assert "reward_prediction_mae" in record["reward_coverage"]
        assert record["replay_positive_reward_steps"] >= 0
        path = tmp_path / "replay-critic.pt"
        trainer.save_checkpoint(path)
        restored = DreamerTrainer.from_checkpoint(path, cuda=False)
        assert restored.cfg.ac.replay_value_scale == 1
        assert restored.cfg.ac.discount == 0.997
        batch = trainer.replay.sample(2, seed=0)
        rng = torch.get_rng_state()
        losses = trainer.update_actor_critic(batch, 1)
        torch.set_rng_state(rng)
        assert losses == restored.update_actor_critic(batch, 1)
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
        if restored is not None:
            restored.env.venv.close()
            restored.tb.close()


def test_longer_imagination_can_reach_a_delayed_reward(world_model, monkeypatch):
    from dreamerrl.models.world_model import WorldModelState
    from dreamerrl.training.core.imagination import imagine_trajectory_for_training
    from dreamerrl.training.core.lambda_return import lambda_return

    actor = Actor(latent=world_model.latent, net=world_model.net_cfg)
    critic = ValueHead(latent=world_model.latent, net=world_model.net_cfg)
    monkeypatch.setattr(
        world_model, "imagine_step", lambda state, *args, **kwargs: WorldModelState(state.h + 1, state.z)
    )
    monkeypatch.setattr(world_model.reward_heads, "forward", lambda h, z: ((h[:, :1] == 20).float(), None))
    monkeypatch.setattr(world_model.reward_heads.main, "readout", lambda logits: logits.squeeze(-1))
    monkeypatch.setattr(critic, "readout", lambda logits: torch.zeros(logits.shape[:-1]))
    short = imagine_trajectory_for_training(world_model, actor, critic, world_model.init_state(1), 15)
    long = imagine_trajectory_for_training(world_model, actor, critic, world_model.init_state(1), 30)
    assert short["reward"].sum() == 0
    assert long["reward"].sum() == 1
    returns = lambda_return(long["reward"], torch.zeros(31, 1), 0.997, 1)
    torch.testing.assert_close(returns[0, 0], torch.tensor(0.997**19))


def test_reward_diagnostics_expose_error_on_rare_positive_rewards(world_model, monkeypatch):
    from dreamerrl.training.core.world_model_update import world_model_training_step

    batch = dict(
        obs=torch.zeros(1, 2, world_model.flat_obs_dim),
        action=torch.zeros(1, 2, dtype=torch.long),
        reward=torch.tensor([[0.0, 1.0]]),
    )
    monkeypatch.setattr(world_model.reward_heads.main, "readout", lambda logits: torch.full(logits.shape[:-1], 0.25))
    metrics = world_model_training_step(world_model, batch).reward_metrics
    assert metrics["positive_reward_fraction"] == 0.5
    assert metrics["reward_prediction_mae"] == 0.5
    assert metrics["positive_reward_prediction_mae"] == 0.75
    assert metrics["positive_reward_prediction_mean"] == 0.25
    batch["reward"].zero_()
    assert "positive_reward_prediction_mae" not in world_model_training_step(world_model, batch).reward_metrics


def test_r2_reward_loss_counts_shaped_win_and_excludes_reset_token(tmp_path, monkeypatch):
    from dreamerrl.tests.manual.utils import minihack_config
    from dreamerrl.training.trainer import DreamerTrainer

    cfg = minihack_config(0, tmp_path)
    cfg.env.env_id = "MiniHack-Corridor-R2-v0"
    cfg.env.num_envs = 1
    cfg.train.action_cost = 1.0
    trainer = DreamerTrainer(cfg)
    try:
        obs = trainer.env_state["state"][:1]
        zero = obs.new_zeros(1)
        trainer.replay.add(obs, zero.long(), zero, zero + 1, zero + 1, initial_obs=obs)
        monkeypatch.setattr(trainer, "collect_env_steps", lambda: None)
        records = []
        trainer.train(1, on_update=records.append)
        coverage = records[0]["reward_coverage"]
        assert coverage["positive_reward_fraction"] == 0
        assert coverage["plus_one_targets"] == cfg.train.batch_size
        assert coverage["plus_one_provenance"] == {
            "unknown": cfg.train.batch_size,
            "scripted": None,
            "actor_collected": None,
        }
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
