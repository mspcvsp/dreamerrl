"""GB10 gradient-cache ablations and real optimizer cycles; no formal RL experiment."""

import json
from typing import Any

import pytest
import torch

from dreamerrl.models.hierarchical_actor import HierarchicalActor, HierarchicalCritic
from dreamerrl.tests.functional.test_hierarchical_actor import components, replay, small_config
from dreamerrl.training.core.actor_critic_update import actor_critic_update
from dreamerrl.training.trainer import DreamerTrainer


def roles(actor, critic):
    return dict(
        manager=actor.manager,
        worker=actor.worker,
        goals=actor.goal_embedding,
        manager_value=critic.manager,
        worker_value=critic.worker,
    )


def gradient_stats(module):
    parameters = list(module.parameters())
    present = [p.grad for p in parameters if p.grad is not None]
    return dict(
        parameters=len(parameters),
        with_gradient=len(present),
        finite=all(bool(torch.isfinite(g).all()) for g in present),
        absolute_sum=sum(float(g.abs().sum()) for g in present),
    )


@pytest.mark.parametrize("unsafe_section", ["imagination", "baseline", "both"])
def test_amp_cache_failure_reproduced_and_fixed(unsafe_section, monkeypatch, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pinned Spark AMP regression")
    original_autocast = torch.autocast

    def backward_case(ablate) -> dict[str, Any]:
        world, actor, critic = components()
        world, actor, critic = world.cuda(), actor.cuda(), critic.cuda()
        guards = []

        def autocast_without_selected_guard(*args, **kwargs):
            if kwargs.get("cache_enabled") is False:
                # Two guarded scopes: imagination, then value-baseline inference.
                section = "imagination" if not guards else "baseline"
                guards.append(section)
                if ablate and (unsafe_section == "both" or unsafe_section == section):
                    kwargs.pop("cache_enabled")  # Inherit the enabled outer cache.
            return original_autocast(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch, "autocast", autocast_without_selected_guard)
            with original_autocast("cuda", dtype=torch.float16, cache_enabled=True):
                actor_loss, critic_loss = actor_critic_update(world, actor, critic, replay(), 5, 0.99, 0.95)
            assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)
            (actor_loss + critic_loss).backward()
        assert guards == ["imagination", "baseline"]
        assert all(p.grad is None for p in world.parameters())
        return dict(
            losses=[float(actor_loss.detach()), float(critic_loss.detach())],
            roles={name: gradient_stats(module) for name, module in roles(actor, critic).items()},
        )

    unsafe = backward_case(True)
    fixed = backward_case(False)
    # The defect is invisible from forward losses: only gradient connectivity differs.
    torch.testing.assert_close(torch.tensor(unsafe["losses"]), torch.tensor(fixed["losses"]), rtol=0, atol=0)
    for name, stats in unsafe["roles"].items():
        blocked = (name in ("manager", "worker") and unsafe_section in ("imagination", "both")) or (
            name in ("manager_value", "worker_value") and unsafe_section in ("baseline", "both")
        )
        if blocked:
            assert stats["with_gradient"] == 0
        else:
            assert stats["with_gradient"] == stats["parameters"] and stats["finite"] and stats["absolute_sum"] > 0
    for stats in fixed["roles"].values():
        assert stats["with_gradient"] == stats["parameters"] and stats["finite"] and stats["absolute_sum"] > 0
    (tmp_path / "cache_ablation.json").write_text(
        json.dumps(dict(unsafe_section=unsafe_section, unsafe=unsafe, fixed=fixed), indent=2) + "\n"
    )


@pytest.mark.parametrize("amp", [False, True])
def test_repeated_noncyber_optimizer_cycles_preserve_ownership_and_world(amp, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FP32/AMP comparison on Spark")
    cfg = small_config(tmp_path / "train")
    cfg.train.cuda = True
    cfg.train.use_amp = amp
    cfg.train.collect_steps = 7
    trainer = DreamerTrainer(cfg)
    assert isinstance(trainer.actor, HierarchicalActor) and isinstance(trainer.critic, HierarchicalCritic)
    try:
        trainer.collect_env_steps()
        world_before = {key: value.detach().clone() for key, value in trainer.world.state_dict().items()}
        assert trainer.policy_state is not None
        policy_before = [
            value.clone()
            for value in (trainer.policy_state.goal_id, trainer.policy_state.goal, trainer.policy_state.remaining)
        ]
        role_modules = roles(trainer.actor, trainer.critic)
        actor_ids = {id(p) for p in trainer.actor.parameters()}
        critic_ids = {id(p) for p in trainer.critic.parameters()}
        world_ids = {id(p) for p in trainer.world.parameters()}
        slow_ids = {id(p) for p in trainer.slow_critic.parameters()}
        actor_opt_ids = {id(p) for g in trainer.actor_opt.param_groups for p in g["params"]}
        critic_opt_ids = {id(p) for g in trainer.critic_opt.param_groups for p in g["params"]}
        assert actor_opt_ids == actor_ids and critic_opt_ids == critic_ids
        assert actor_ids == {id(p) for name in ("manager", "worker", "goals") for p in role_modules[name].parameters()}
        assert critic_ids == {
            id(p) for name in ("manager_value", "worker_value") for p in role_modules[name].parameters()
        }
        assert not (actor_ids & critic_ids or (actor_ids | critic_ids) & (world_ids | slow_ids))
        receipt = []
        for cycle in range(3):
            trainer.actor_opt.zero_grad(set_to_none=True)
            trainer.critic_opt.zero_grad(set_to_none=True)
            before = {name: [p.detach().clone() for p in module.parameters()] for name, module in role_modules.items()}
            batch = trainer.replay.sample(2, seed=40 + cycle)
            actor_loss, critic_loss = trainer.update_actor_critic(batch, cycle)
            assert torch.isfinite(torch.tensor([actor_loss, critic_loss])).all()
            record: dict[str, Any] = dict(
                cycle=cycle + 1,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                optimizer_steps=dict(trainer._optimizer_steps),
                roles={},
            )
            for name, module in role_modules.items():
                stats = gradient_stats(module)
                assert stats["with_gradient"] == stats["parameters"] and stats["finite"] and stats["absolute_sum"] > 0
                stats["max_parameter_delta"] = max(
                    float((p.detach() - old).abs().max()) for p, old in zip(module.parameters(), before[name])
                )
                assert stats["max_parameter_delta"] > 0
                record["roles"][name] = stats
            assert trainer._optimizer_steps == dict(world_model=0, actor=cycle + 1, critic=cycle + 1)
            for optimizer in (trainer.actor_opt, trainer.critic_opt):
                assert optimizer.state and all(int(state["step"]) == cycle + 1 for state in optimizer.state.values())
            assert not trainer.model_opt.state
            assert all(p.grad is None for p in trainer.world.parameters())
            assert all(not p.requires_grad and p.grad is None for p in trainer.slow_critic.parameters())
            for key, value in trainer.world.state_dict().items():
                torch.testing.assert_close(value, world_before[key], rtol=0, atol=0)
            assert trainer.policy_state is not None
            for a, b in zip(
                policy_before, (trainer.policy_state.goal_id, trainer.policy_state.goal, trainer.policy_state.remaining)
            ):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            receipt.append(record)
        (tmp_path / "optimizer_ownership.json").write_text(
            json.dumps(
                dict(
                    amp=amp,
                    device=torch.cuda.get_device_name(),
                    torch=torch.__version__,
                    cycles=receipt,
                    world_unchanged=True,
                    world_optimizer_steps=0,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
