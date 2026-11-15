"""Finite task endings and continuing-task cutoffs have different return semantics."""

import importlib

import pytest
import torch

from dreamerrl.eval.popgym_eval import evaluate_popgym
from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.core.actor_critic_update import actor_critic_update
from dreamerrl.training.trainer import DreamerTrainer


@pytest.mark.parametrize("finite", [False, True])
def test_task_end_continuation_and_time_feature(tmp_path, finite):
    cfg = minihack_config(0, tmp_path)
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"
    cfg.env.max_episode_steps = 3
    cfg.env.finite_horizon = finite
    cfg.train.collect_steps = 3
    cfg.train.seq_len = 16
    trainer = DreamerTrainer(cfg)
    resumed = None
    try:
        assert trainer.world.flat_obs_dim == trainer.env.obs_dim + int(finite)
        if finite:
            assert (trainer.env_state["state"][:, -1] == 1).all()
        trainer.collect_env_steps()
        ep = trainer.replay.episodes[0]
        assert ep["done"][-1]
        assert bool(ep["is_terminal"][-1]) == finite  # TimeLimit expiry, not natural termination
        if finite:
            torch.testing.assert_close(ep["obs"][:, -1], torch.tensor([1.0, 2 / 3, 1 / 3, 0.0]))
            assert (trainer.env_state["state"][:, -1] == 1).all()
            assert (trainer.env_state["transition_state"][:, -1] == 0).all()
            assert trainer.episode_steps.count_nonzero() == 0
        batch = trainer.replay.sample(2)
        metrics = trainer.update_world_model(batch, 0)
        assert torch.isfinite(metrics.cont_loss)
        actor_loss, critic_loss = trainer.update_actor_critic(batch, 0)
        assert torch.isfinite(torch.tensor([actor_loss, critic_loss])).all()
        stats = evaluate_popgym(
            trainer.env, trainer.world, trainer.actor, episodes=1, task_horizon=3 if finite else None
        )
        assert torch.isfinite(torch.tensor(stats["mean"]))
        # Restoring a finite run starts fresh clocks, consistent with fresh simulator episodes.
        path = tmp_path / "latest.pt"
        trainer.save_checkpoint(path)
        resumed = DreamerTrainer.from_checkpoint(path, cuda=False)
        assert resumed.cfg.env.finite_horizon == finite
        assert resumed.world.flat_obs_dim == trainer.world.flat_obs_dim
        if finite:
            assert (resumed.env_state["state"][:, -1] == 1).all()
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
        if resumed is not None:
            resumed.env.venv.close()
            resumed.tb.close()


def test_known_remaining_steps_cut_imagined_returns(world_model, monkeypatch):
    module = importlib.import_module("dreamerrl.training.core.actor_critic_update")
    actor = Actor(latent=world_model.latent, net=world_model.net_cfg)
    critic = ValueHead(latent=world_model.latent, net=world_model.net_cfg)
    batch = dict(obs=torch.zeros(1, 2, 8), action=torch.zeros(1, 2, dtype=torch.long), is_terminal=torch.zeros(1, 2))

    def imagine(world, actor, critic, start, horizon, deterministic):
        return dict(
            h=torch.zeros(4, 2, world.latent.deter_size),
            z=torch.zeros(4, 2, world.latent.num_classes, world.latent.stoch_size),
            reward=torch.ones(4, 2),
            action=torch.zeros(4, 2, dtype=torch.long),
            **{"continue": torch.ones(4, 2)},
            bootstrap_value=torch.full((2,), 1000.0),
        )

    original_return = module.lambda_return
    checked = []

    def returns(reward, value, discounts, lam):
        result = original_return(reward, value, discounts, lam)
        torch.testing.assert_close(discounts, torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
        torch.testing.assert_close(result[0], torch.tensor([1.0, 3.0]))
        checked.append(True)
        return result

    monkeypatch.setattr(module, "imagine_trajectory_for_training", imagine)
    monkeypatch.setattr(module, "lambda_return", returns)
    a, c = actor_critic_update(world_model, actor, critic, batch, 4, 1.0, 1.0, remaining_steps=torch.tensor([[1, 3]]))
    assert checked
    assert torch.isfinite(a) and torch.isfinite(c)


@pytest.mark.parametrize("parallel", [False, True])
def test_remaining_time_resets_independently_by_environment(tmp_path, monkeypatch, parallel):
    from dreamerrl.tests.popgym.test_popgym_contracts import CueSentinel

    name = "popgym_parallel_env" if parallel else "popgym_wrappers"
    module = importlib.import_module("dreamerrl.env.popgym." + name)
    monkeypatch.setattr(module, "make_env", lambda cfg, i: lambda: CueSentinel(i))
    cfg = minihack_config(0, tmp_path)
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"
    cfg.env.num_envs = 3
    cfg.env.parallel = parallel
    cfg.env.max_episode_steps = 3
    cfg.env.finite_horizon = True
    cfg.train.collect_steps = 2
    trainer = DreamerTrainer(cfg)
    try:
        trainer.collect_env_steps()
        assert trainer.episode_steps.tolist() == [0, 0, 2]
        torch.testing.assert_close(trainer.env_state["state"][:, -1], torch.tensor([1.0, 1.0, 1 / 3]))
        torch.testing.assert_close(trainer.env_state["transition_state"][:, -1], torch.tensor([2 / 3, 1 / 3, 1 / 3]))
        assert trainer.env_state["is_terminal"].tolist() == [True, True, False]
        clocks = []
        hook = trainer.world.encoder.register_forward_pre_hook(
            lambda module, args: clocks.append(args[0][:, -1].clone())
        )
        try:
            stats = evaluate_popgym(trainer.env, trainer.world, trainer.actor, episodes=1, task_horizon=3)
        finally:
            hook.remove()
        torch.testing.assert_close(
            torch.stack(clocks), torch.tensor([[1.0, 1.0, 1.0], [1.0, 2 / 3, 2 / 3], [1.0, 1.0, 1 / 3]])
        )
        assert stats["episode_returns"] == [[-0.25, -0.75, -1.5]]
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
