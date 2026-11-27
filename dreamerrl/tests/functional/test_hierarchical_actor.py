"""Synthetic state/cadence/gradient contracts and bounded non-cyber integration."""

import copy
import io

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.hierarchical_actor import HierarchicalActor, HierarchicalCritic, policy_action
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.core.actor_critic_update import actor_critic_update
from dreamerrl.training.core.hierarchical_update import imagine_hierarchy, manager_returns
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


def components(interval=3, *, progress=False):
    torch.manual_seed(42)
    latent = LatentConfig(deter_size=12, num_classes=3, stoch_size=4)
    net = NetworkConfig(hidden_size=16, action_dim=5, value_bins=21)
    actor = HierarchicalActor(
        latent=latent, net=net, goals=4, goal_dim=6, interval=interval,
        director_goals=progress, goal_progress=progress,
    )
    critic = HierarchicalCritic(latent=latent, net=net, goals=4, interval=interval)
    cfg = DreamerConfig()
    cfg.world.disable_aux_losses = True
    world = WorldModel(obs_space=gym.spaces.Box(-10, 10, (7,), np.float32), dreamer_cfg=cfg, latent=latent, net=net)
    return world, actor, critic


def replay():
    return dict(
        obs=torch.randn(2, 4, 7),
        action=torch.randint(5, (2, 4)),
        reward=torch.randn(2, 4),
        is_terminal=torch.zeros(2, 4),
        is_first=torch.zeros(2, 4, dtype=torch.bool),
    )


def test_batch_shapes_cadence_persistence_and_row_reset():
    world, actor, _ = components()
    state = world.init_state(2)
    policy = actor.initial_state(2)
    decisions = []
    for step in range(7):
        result = actor.step(state.h, state.z, policy, deterministic=True)
        decisions.append(result.manager_decision.tolist())
        assert result.logits.shape == (2, 5) and result.action.shape == (2,)
        if step % 3:
            torch.testing.assert_close(result.state.goal, policy.goal)
        policy = result.state
    assert decisions == [
        [True, True],
        [False, False],
        [False, False],
        [True, True],
        [False, False],
        [False, False],
        [True, True],
    ]
    result = actor.step(state.h, state.z, policy, reset=torch.tensor([True, False]), deterministic=True)
    assert result.manager_decision.tolist() == [True, False]
    assert result.state.remaining.tolist() == [2, 1]


def test_goal_vector_latches_across_parameter_updates():
    world, actor, _ = components(interval=3)
    state = world.init_state(2)
    first = actor.step(state.h, state.z, actor.initial_state(2), deterministic=True)
    with torch.no_grad():
        actor.goal_embedding.weight.add_(100)
    second = actor.step(state.h, state.z, first.state, deterministic=True)
    third = actor.step(state.h, state.z, second.state, deterministic=True)
    fourth = actor.step(state.h, state.z, third.state, deterministic=True)
    torch.testing.assert_close(first.state.goal, second.state.goal)
    torch.testing.assert_close(first.state.goal, third.state.goal)
    assert not torch.equal(first.state.goal, fourth.state.goal)


def test_worker_uses_goal_and_phase():
    world, actor, critic = components()
    state = world.init_state(2)
    a = actor.worker_logits(state.h, state.z, torch.zeros(2, 6), torch.ones(2, dtype=torch.long))
    b = actor.worker_logits(state.h, state.z, torch.ones(2, 6), torch.ones(2, dtype=torch.long))
    c = actor.worker_logits(state.h, state.z, torch.zeros(2, 6), torch.full((2,), 3))
    assert not torch.allclose(a, b) and not torch.allclose(a, c)
    a = critic.worker_logits(state.h, state.z, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    b = critic.worker_logits(state.h, state.z, torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("interval,horizon,remaining", [(1, 5, 0), (3, 5, 1), (3, 6, 0), (8, 3, 5)])
def test_imagination_starts_fresh_and_preserves_final_partial_goal(interval, horizon, remaining):
    world, actor, _ = components(interval)
    result = imagine_hierarchy(world, actor, world.init_state(2), horizon, True)
    assert result["tail_remaining"].tolist() == [remaining, remaining]
    assert result["decision"][:, 0].tolist() == [step % interval == 0 for step in range(horizon)]
    assert result["h"].shape == (horizon, 2, 12)
    assert result["z"].shape == (horizon, 2, 3, 4)


def test_macro_rewards_discount_products_and_partial_tail_reference():
    rewards = torch.arange(1.0, 6.0)[:, None]
    discounts = torch.full_like(rewards, 0.5)
    values = torch.arange(10.0, 60.0, 10)[:, None]
    tail = torch.tensor([6.0])
    torch.testing.assert_close(
        manager_returns(rewards, discounts, values, tail, 2, 1), torch.tensor([[3.75], [7.0], [8.0]])
    )
    torch.testing.assert_close(
        manager_returns(rewards, discounts, values, tail, 2, 0), torch.tensor([[9.5], [17.5], [8.0]])
    )
    discounts[1] = 0
    assert manager_returns(rewards, discounts, values, tail, 2, 1)[0].item() == 2
    torch.testing.assert_close(
        manager_returns(rewards[:2], discounts[:2] * 0 + 0.5, values[:2], tail, 3, 1), torch.tensor([[3.5]])
    )


def has_gradient(module):
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters()
    )


def test_gradient_ownership_and_independent_optimizers():
    world, actor, critic = components()
    slow = copy.deepcopy(critic).requires_grad_(False)
    diagnostics = {}
    a, c = actor_critic_update(world, actor, critic, replay(), 7, 0.99, 0.95, slow_critic=slow, diagnostics=diagnostics)
    assert torch.isfinite(a) and torch.isfinite(c)
    a.backward()
    for module in (actor.manager, actor.worker, actor.goal_embedding):
        assert has_gradient(module)
    assert all(p.grad is None for module in (world, critic, slow) for p in module.parameters())
    c.backward()
    assert has_gradient(critic.manager) and has_gradient(critic.worker)
    assert all(p.grad is None for module in (world, slow) for p in module.parameters())
    groups = [{id(p) for p in m.parameters()} for m in (world, actor, critic, slow)]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1 :])
    assert diagnostics["manager_decisions"] == 3 * 8
    assert diagnostics["worker_decisions"] == 7 * 8


@pytest.mark.parametrize("progress", [False, True])
def test_terminal_starts_and_task_horizon_mask_all_policy_losses(progress):
    world, actor, critic = components(progress=progress)
    batch = replay()
    batch["is_terminal"].fill_(1)
    a, c = actor_critic_update(world, actor, critic, batch, 5, 0.99, 0.95)
    (a + c).backward()
    assert a == 0 and c == 0
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for m in (actor, critic) for p in m.parameters())
    batch["is_terminal"].zero_()
    a, c = actor_critic_update(world, actor, critic, batch, 5, 0.99, 0.95, remaining_steps=torch.zeros(2, 4))
    assert a == 0 and c == 0


def test_serialization_and_independent_policy_states():
    world, actor, critic = components()
    state = world.init_state(2)
    first = actor.step(state.h, state.z, actor.initial_state(2), deterministic=True)
    payload = io.BytesIO()
    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), policy=first.state), payload)
    payload.seek(0)
    saved = torch.load(payload, weights_only=False)
    _, restored, restored_critic = components()
    restored.load_state_dict(saved["actor"])
    restored_critic.load_state_dict(saved["critic"])
    a = actor.step(state.h, state.z, first.state, deterministic=True)
    b = restored.step(state.h, state.z, saved["policy"], deterministic=True)
    torch.testing.assert_close(a.logits, b.logits)
    assert a.state.remaining.tolist() == b.state.remaining.tolist()
    _, _, fresh = policy_action(actor, state, None, deterministic=True)
    assert fresh is not None
    assert fresh.remaining.tolist() == [2, 2]
    assert first.state.remaining.tolist() == [2, 2]


@pytest.mark.parametrize("kwargs", [dict(goals=1), dict(goal_dim=0), dict(interval=0), dict(interval=1.5)])
def test_invalid_dimensions_rejected(kwargs):
    latent = LatentConfig(deter_size=12, num_classes=3, stoch_size=4)
    net = NetworkConfig(hidden_size=16, action_dim=5)
    with pytest.raises(ValueError):
        HierarchicalActor(latent=latent, net=net, **kwargs)


def test_flat_evaluation_helper_is_exact_and_does_not_create_state():
    world, _, _ = components()
    actor = Actor(latent=world.latent, net=world.net_cfg)
    state = world.init_state(2)
    rng = torch.get_rng_state()
    expected = actor.act(state)
    torch.set_rng_state(rng)
    action, logits, policy = policy_action(actor, state)
    torch.testing.assert_close(action, expected[0])
    torch.testing.assert_close(logits, expected[1])
    assert policy is None


def small_config(tmp_path, hierarchical=True):
    cfg = DreamerConfig()
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"
    cfg.env.num_envs = 2
    cfg.env.max_episode_steps = 6
    cfg.env.parallel = False
    cfg.env.seed = 19
    cfg.world.deter_size = 12
    cfg.world.num_classes = 3
    cfg.world.stoch_size = 4
    cfg.world.hidden_size = 16
    cfg.world.imagination_horizon = 5
    cfg.world.disable_aux_losses = True
    cfg.ac.actor_hidden = cfg.ac.critic_hidden = 16
    cfg.ac.hierarchical = hierarchical
    cfg.ac.hierarchy_goals = 4
    cfg.ac.hierarchy_goal_dim = 6
    cfg.ac.hierarchy_interval = 3
    cfg.train.cuda = False
    cfg.train.use_amp = False
    cfg.train.enable_fused_adamw_kernels = False
    cfg.train.seed = 19
    cfg.train.freeze_actor_critic_steps = 0
    cfg.train.random_exploration_steps = 0
    cfg.train.collect_steps = 12
    cfg.train.seq_len = 4
    cfg.train.burn_in = 3
    cfg.train.batch_size = 2
    cfg.log.tb_logdir = str(tmp_path)
    cfg.log.run_name = "hierarchy_test"
    cfg.log.enable_wandb = False
    return cfg


@pytest.mark.parametrize("director", [False, True])
def test_noncyber_collect_update_restart_evaluate(tmp_path, director):
    from dreamerrl.eval.popgym_eval import evaluate_popgym

    cfg = small_config(tmp_path / "train")
    cfg.ac.director_goals = director
    trainer = DreamerTrainer(cfg)
    restored = None
    try:
        trainer.collect_env_steps()
        batch = trainer.replay.sample(2, seed=3)
        before = {k: v.detach().clone() for k, v in trainer.actor.named_parameters()}
        model = trainer.update_world_model(batch, 0)
        a, c = trainer.update_actor_critic(batch, 0)
        assert np.isfinite([model.total_loss.item(), a, c]).all()
        assert trainer._optimizer_steps == dict(world_model=1, actor=1, critic=1)
        for prefix in ("manager.", "worker.", "goal_autoencoder." if director else "goal_embedding."):
            assert any(
                not torch.equal(before[k], v) for k, v in trainer.actor.named_parameters() if k.startswith(prefix)
            )
        trainer.completed_updates = 1
        path = tmp_path / "restart.pt"
        trainer.save_checkpoint(path)
        assert torch.load(path, weights_only=False)["version"] == (6 if director else 5)
        restored = DreamerTrainer.from_checkpoint(path, cuda=False, log_dir=tmp_path / "reload")
        assert restored.completed_updates == 1 and len(restored.replay.episodes) == len(trainer.replay.episodes)
        assert restored.policy_state is not None
        assert restored.policy_state.remaining.tolist() == [0, 0]
        for key, value in trainer.actor.state_dict().items():
            torch.testing.assert_close(value, restored.actor.state_dict()[key], rtol=0, atol=0)
        for name in ("model_opt", "actor_opt", "critic_opt"):
            assert all(int(v["step"]) == 1 for v in getattr(restored, name).state.values())
        if director:
            assert all(int(v["step"]) == 1 for v in restored.goal_opt.state.values())
            goal_ids = {id(p) for g in restored.goal_opt.param_groups for p in g["params"]}
            actor_ids = {id(p) for g in restored.actor_opt.param_groups for p in g["params"]}
            assert goal_ids and actor_ids and goal_ids.isdisjoint(actor_ids)
        saved_policy = restored.policy_state.remaining.clone()
        from dreamerrl.env.env_factory import make_env

        evaluation_env = make_env(copy.deepcopy(restored.cfg.env), restored.device)
        try:
            result = evaluate_popgym(evaluation_env, restored.world, restored.actor, episodes=1, seed=10000)
        finally:
            evaluation_env.venv.close()
        assert np.isfinite(result["mean"])
        torch.testing.assert_close(restored.policy_state.remaining, saved_policy)
        new_batch = restored.replay.sample(2, seed=7)
        a, c = restored.update_actor_critic(new_batch, 1)
        assert np.isfinite([a, c]).all()
        assert all(int(v["step"]) == 2 for v in restored.actor_opt.state.values())
    finally:
        for item in (trainer, restored):
            if item is not None:
                item.env.venv.close()
                item.tb.close()


def test_collector_resets_goal_at_both_truncation_and_terminal(tmp_path, monkeypatch):
    trainer = DreamerTrainer(small_config(tmp_path))
    assert isinstance(trainer.actor, HierarchicalActor)
    original = trainer.env.step
    try:
        trainer.cfg.train.collect_steps = 1

        # Interpose a valid end signal; no next-step action can borrow this goal.
        def ended(actions):
            result = original(actions)
            result["is_last"] = torch.ones(2, dtype=torch.bool)
            result["is_terminal"] = torch.tensor([True, False])
            return result

        monkeypatch.setattr(trainer.env, "step", ended)
        trainer.collect_env_steps()
        assert trainer.policy_state is not None
        assert trainer.policy_state.remaining.tolist() == [0, 0]
        assert trainer.policy_state.goal.count_nonzero() == 0
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def test_checkpoint_mode_rejected_before_parameter_mutation(tmp_path):
    hierarchy = DreamerTrainer(small_config(tmp_path / "hier"))
    flat = DreamerTrainer(small_config(tmp_path / "flat", False))
    try:
        a = tmp_path / "hier.pt"
        b = tmp_path / "flat.pt"
        hierarchy.save_checkpoint(a)
        flat.save_checkpoint(b)
        assert torch.load(b, weights_only=False)["version"] == 4
        before = {k: v.clone() for k, v in flat.world.state_dict().items()}
        with pytest.raises(ValueError, match="mode/schema"):
            flat.load_checkpoint(a)
        with pytest.raises(ValueError, match="mode/schema"):
            hierarchy.load_checkpoint(b)
        for key, value in flat.world.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    finally:
        for trainer in (hierarchy, flat):
            trainer.env.venv.close()
            trainer.tb.close()


def test_imagination_goal_updates_do_not_mutate_collection_state():
    world, actor, _ = components()
    s = world.init_state(2)
    live = actor.step(s.h, s.z, actor.initial_state(2), deterministic=True).state
    before = (live.goal_id.clone(), live.goal.clone(), live.remaining.clone())
    imagine_hierarchy(world, actor, s, 7)
    for a, b in zip(before, (live.goal_id, live.goal, live.remaining)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_invalid_state_rejected_instead_of_borrowing_goal():
    world, actor, _ = components()
    s = world.init_state(2)
    with pytest.raises(ValueError, match="shape"):
        actor.step(s.h, s.z, actor.initial_state(1))
    bad = actor.initial_state(2)
    bad.remaining[0] = -1
    with pytest.raises(ValueError, match="counter"):
        actor.step(s.h, s.z, bad)
    with pytest.raises(ValueError, match="explicit goal state"):
        actor(s.h, s.z)


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("progress", [False, True])
def test_gpu_finite_backward_and_live_goal_state(amp, progress):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    world, actor, critic = components(progress=progress)
    world, actor, critic = world.cuda(), actor.cuda(), critic.cuda()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        a, c = actor_critic_update(world, actor, critic, replay(), 5, 0.99, 0.95)
    (a + c).backward()
    assert torch.isfinite(a) and torch.isfinite(c)
    for module in (actor.manager, actor.worker, critic.manager, critic.worker):
        assert has_gradient(module)
        assert all(p.device.type == "cuda" for p in module.parameters())
    initial = world.init_state(2)
    chosen = actor.step(initial.h, initial.z, actor.initial_state(2))
    assert chosen.action.device.type == "cuda" and chosen.state.goal.device.type == "cuda"


def test_config_preserves_flat_default_and_rejects_flat_replay_loss_in_hierarchy(tmp_path):
    assert not DreamerConfig().ac.hierarchical
    cfg = small_config(tmp_path)
    cfg.ac.replay_value_scale = 0.1
    with pytest.raises(ValueError, match="replay_value_scale"):
        DreamerTrainer(cfg)


def test_real_episode_reset_and_collection_chunks_keep_correct_cadence(tmp_path, monkeypatch):
    cfg = small_config(tmp_path)
    cfg.ac.hierarchy_interval = 4
    cfg.train.collect_steps = 1
    trainer = DreamerTrainer(cfg)
    assert isinstance(trainer.actor, HierarchicalActor)
    decisions = []
    original = trainer.actor.step

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        decisions.append(result.manager_decision.tolist())
        return result

    monkeypatch.setattr(trainer.actor, "step", recorded)
    try:
        for _ in range(7):
            trainer.collect_env_steps()
        assert decisions == [
            [True, True],
            [False, False],
            [False, False],
            [False, False],
            [True, True],
            [False, False],
            [True, True],
        ]
        assert trainer.completed_episodes == 2
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def test_finite_horizon_cuts_rewards_after_last_action_for_both_levels(monkeypatch):
    from dreamerrl.training.core import hierarchical_update as module
    from dreamerrl.training.core.world_model_update import observe_batch

    world, actor, critic = components()
    batch = replay()
    with torch.no_grad():
        posts = observe_batch(world, batch)
        from dreamerrl.models.world_model import WorldModelState

        start = WorldModelState(
            torch.stack([p["h"] for p in posts], 1).reshape(8, -1),
            torch.stack([p["z"] for p in posts], 1).reshape(8, 3, 4),
        )
        trajectory = imagine_hierarchy(world, actor, start, 7, True)
    suffix = [0.0]

    def fixed(*args):
        result = dict(trajectory)
        result["reward"] = trajectory["reward"].clone()
        result["reward"][2:] += suffix[0]
        return result

    monkeypatch.setattr(module, "imagine_hierarchy", fixed)
    remaining = torch.full((2, 4), 2)
    a0, c0 = actor_critic_update(world, actor, critic, batch, 7, 0.99, 0.95, remaining_steps=remaining)
    suffix[0] = 10000.0
    a1, c1 = actor_critic_update(world, actor, critic, batch, 7, 0.99, 0.95, remaining_steps=remaining)
    torch.testing.assert_close(a0, a1, rtol=0, atol=0)
    torch.testing.assert_close(c0, c1, rtol=0, atol=0)


@pytest.mark.parametrize(
    "directory,update,expected_sha",
    [
        ("cage2_blue_cycle_seed0_20261123", 1, "94f378899c0422e0af2db2e07cbb13eab9b27a1448cfbe2deb955617c0e5d2ff"),
        (
            "cage2_host_aware_pilot_seed0_20261124",
            16899,
            "8a2483a6add81e1db709f35079f086c50778b900c24164b99fa29b62240cc1bd",
        ),
    ],
)
def test_historical_flat_checkpoint_weights_and_optimizers_still_load(directory, update, expected_sha, tmp_path):
    import hashlib
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "runs" / directory / "latest.pt"
    if not path.exists():
        pytest.skip("Historical checkpoint not distributed with source")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == expected_sha
    saved = torch.load(path, map_location="cpu", weights_only=False)
    trainer = DreamerTrainer.from_checkpoint(path, cuda=False, log_dir=tmp_path)
    try:
        assert not trainer.cfg.ac.hierarchical and isinstance(trainer.actor, Actor)
        assert saved["version"] == 4 and trainer.completed_updates == update
        for name in ("world", "actor", "critic", "slow_critic"):
            for key, value in saved[name].items():
                torch.testing.assert_close(getattr(trainer, name).state_dict()[key], value, rtol=0, atol=0)
        for name in ("model_opt", "actor_opt", "critic_opt"):
            actual = getattr(trainer, name).state_dict()["state"]
            for key, state in saved[name]["state"].items():
                for field, value in state.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(actual[key][field].cpu(), value.cpu(), rtol=0, atol=0)
        assert len(trainer.replay.episodes) == len(saved["replay"]["episodes"])
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("device,amp", [("cpu", False), ("cpu", True), ("cuda", True)])
def test_director_tensor_gradient_and_achievement_contract(device, amp):
    from dreamerrl.models.goal_autoencoder import goal_features, goal_similarity

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(10)
    actor = HierarchicalActor(
        latent=LatentConfig(deter_size=12, num_classes=3, stoch_size=4),
        net=NetworkConfig(hidden_size=16, action_dim=5),
        goals=4, goal_dim=6, interval=2, director_goals=True,
    ).to(device)
    h = torch.randn(2, 12, device=device, requires_grad=True)
    z = torch.randn(2, 4, 3, device=device, requires_grad=True)
    features = goal_features(h, z)
    ae = actor.goal_autoencoder
    logits, decoded = ae(features)
    assert logits.shape == (2, 4)
    assert decoded.shape == features.shape == (2, 24)
    torch.testing.assert_close(goal_similarity(features, features), torch.ones(2, device=device))
    torch.testing.assert_close(goal_similarity(-features, features), -torch.ones(2, device=device))
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device, dtype=dtype, enabled=amp):
        first = actor.step(h, z, actor.initial_state(2), deterministic=True)
        with torch.no_grad():
            ae.decoder[-1].bias.add_(2)
        second = actor.step(h, z, first.state, deterministic=True)
        third = actor.step(h, z, second.state, deterministic=True)
        assert first.manager_decision.all() and not second.manager_decision.any() and third.manager_decision.all()
        torch.testing.assert_close(first.state.goal, second.state.goal)
        assert not torch.equal(second.state.goal, third.state.goal)
        assert not torch.allclose(first.logits, third.logits)
        objective, mse, _ = ae.reconstruction(features, torch.ones(2, device=device))
    assert objective.dtype == torch.float32 and mse.item() > 0
    objective.backward()
    assert has_gradient(ae.encoder) and has_gradient(ae.decoder)
    assert h.grad is None and z.grad is None
    assert not has_gradient(actor.manager) and not has_gradient(actor.worker)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_flat_amp_trainer_preserves_actor_and_critic_gradients(tmp_path):
    cfg = small_config(tmp_path, hierarchical=False)
    cfg.train.cuda = cfg.train.use_amp = True
    trainer = DreamerTrainer(cfg)
    try:
        trainer.collect_env_steps()
        losses = trainer.update_actor_critic(trainer.replay.sample(2, seed=3), 0)
        assert np.isfinite(losses).all()
        assert has_gradient(trainer.actor) and has_gradient(trainer.critic)
        assert trainer._optimizer_steps["actor"] == trainer._optimizer_steps["critic"] == 1
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def test_signed_progress_uses_same_goal_on_both_sides_of_transition(monkeypatch):
    from dreamerrl.models.world_model import WorldModelState

    world, actor, _ = components(interval=3, progress=True)
    states = iter([0, 0, 1, 0, 1, 1])
    z = torch.zeros(1, 4, 3)

    def state(axis):
        return WorldModelState(torch.nn.functional.one_hot(torch.tensor([axis]), 12).float(), z)

    goals = iter([0, 1])
    monkeypatch.setattr(actor, "decode_goals", lambda ids: torch.nn.functional.one_hot(
        torch.tensor([next(goals)]), actor.goal_dim
    ).float())
    monkeypatch.setattr(world, "imagine_step", lambda *args, **kwargs: state(next(states)))
    result = imagine_hierarchy(world, actor, state(1), 6, deterministic=True)
    torch.testing.assert_close(result["progress"][:, 0], torch.tensor([1., 0., -1., -1., 1., 0.]))
    torch.testing.assert_close(result["achievement"][:, 0], torch.tensor([1., 1., 0., 0., 1., 1.]))
    assert result["decision"][:, 0].tolist() == [True, False, False, True, False, False]


def test_standardized_goal_buffers_and_native_decode():
    from dreamerrl.models.goal_autoencoder import GoalAutoencoder, fit_goal_scale

    fitting = torch.tensor([[1., 4., 0.], [1., 4.0001, 2.]])
    center, scale = fit_goal_scale(fitting, .01)
    torch.testing.assert_close(scale, torch.tensor([.01, .01, 1.]))
    model = GoalAutoencoder(3, 4, 6, standardized=True)
    model.set_normalization(center, scale)
    torch.testing.assert_close(model.to_native_coordinates(model.to_training_coordinates(fitting)), fitting)
    logits, native = model(fitting)
    ids = logits.argmax(-1)
    torch.testing.assert_close(native, model.decode_ids(ids))
    objective, _, _ = model.reconstruction(fitting.requires_grad_(), torch.ones(2))
    objective.backward()
    assert fitting.grad is None
    assert not model.goal_center.requires_grad and not model.goal_scale.requires_grad
    assert has_gradient(model.encoder) and has_gradient(model.decoder)
    restored = GoalAutoencoder(3, 4, 6, standardized=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.decode_ids(ids), native)
