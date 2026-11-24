"""Numerical and gradient checks through the actual training paths."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.categorical_kl import structured_kl
from dreamerrl.models.reward_head import RewardHead
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel, WorldModelState
from dreamerrl.training.core.actor_critic_update import actor_critic_update
from dreamerrl.training.core.imagination import imagine_trajectory_for_training
from dreamerrl.training.core.lambda_return import lambda_return
from dreamerrl.training.core.world_model_update import observe_batch, world_model_training_step
from dreamerrl.utils.transforms import symexp, symlog
from dreamerrl.utils.twohot import twohot_encode
from dreamerrl.utils.types import DreamerConfig, KLConfig, LatentConfig, NetworkConfig


@pytest.fixture
def small_model():
    cfg = DreamerConfig()
    cfg.world.kl_cfg = replace(cfg.world.kl_cfg, free_nats=0)
    latent = LatentConfig(deter_size=16, num_classes=3, stoch_size=4)
    net = NetworkConfig(hidden_size=16, action_dim=3, value_bins=21, bin_min=-5, bin_max=5)
    wm = WorldModel(obs_space=gym.spaces.Box(-100, 100, (4,), np.float32), dreamer_cfg=cfg, latent=latent, net=net)
    return wm


@pytest.fixture
def sequence():
    return dict(
        obs=torch.randn(2, 4, 4),
        action=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]),
        reward=torch.randn(2, 4),
        is_terminal=torch.zeros(2, 4),
        is_first=torch.zeros(2, 4, dtype=torch.bool),
        action_mask=torch.ones(2, 4, 3),
    )


@pytest.mark.parametrize("head_type", [RewardHead, ValueHead])
def test_twohot_uses_original_units_once(head_type, small_model):
    head = head_type(latent=small_model.latent, net=small_model.net_cfg)
    logits = torch.linspace(-2, 2, 21).repeat(3, 1).requires_grad_()
    raw = torch.tensor([-20.0, 0.0, 7.0])
    expected = -(twohot_encode(symlog(raw), head.bin_values) * logits.log_softmax(-1)).sum(-1).mean()
    torch.testing.assert_close(head.loss_from_logits(logits, raw), expected)
    torch.testing.assert_close(head.readout(logits), symexp((logits.softmax(-1) * head.bin_values).sum(-1)))
    assert "bin_values" in head.state_dict()


def test_lambda_return_reference_and_terminal_cutoff():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    torch.testing.assert_close(lambda_return(rewards, values, 0.5, 0), torch.tensor([[11.0], [17.0], [23.0]]))
    torch.testing.assert_close(lambda_return(rewards, values, 0.5, 1), torch.tensor([[7.75], [13.5], [23.0]]))
    discounts = torch.tensor([[0.5], [0.0], [0.5]])
    expected = torch.tensor([[2.0], [2.0], [23.0]])
    torch.testing.assert_close(lambda_return(rewards, values, discounts, 1), expected)


def test_kl_gradient_direction_and_floor():
    q = torch.tensor([[[2.0, -1.0], [1.0, -1.0]]], requires_grad=True)
    p = torch.zeros_like(q, requires_grad=True)
    out = structured_kl(q.softmax(-1), p.softmax(-1), KLConfig(free_nats=0))
    gq, gp = torch.autograd.grad(out["kl_dyn"].sum(), (q, p), allow_unused=True, retain_graph=True)
    assert gq is None and gp.abs().sum() > 0
    gq, gp = torch.autograd.grad(out["kl_rep"].sum(), (q, p), allow_unused=True)
    assert gp is None and gq.abs().sum() > 0
    out = structured_kl(q.softmax(-1), p.softmax(-1), KLConfig(free_nats=20))
    grads = torch.autograd.grad(out["kl_total"].sum(), (q, p))
    assert all(g.count_nonzero() == 0 for g in grads)


def test_observe_transition_timing_and_row_reset(small_model):
    wm = small_model
    state = WorldModelState(torch.randn(2, 16), torch.randn(2, 3, 4))
    obs, action = torch.randn(2, 4), torch.eye(3)[:2]
    out = wm.observe_step(state, obs, action)
    expected_h = wm.rssm(state.h, state.z, action)
    torch.testing.assert_close(out["post"].h, expected_h)
    torch.testing.assert_close(out["prior"].h, expected_h)
    torch.testing.assert_close(out["prior_stats"]["probs"], wm.prior(expected_h)["probs"])
    torch.testing.assert_close(out["post_stats"]["probs"], wm.posterior(expected_h, wm.encoder(obs))["probs"])
    reset = wm.observe_step(state, obs, action, is_first=torch.tensor([True, False]))
    fresh = wm.observe_step(wm.init_state(2), obs, torch.zeros_like(action))
    torch.testing.assert_close(reset["post"].h[0], fresh["post"].h[0])
    torch.testing.assert_close(reset["post_stats"]["probs"][0], fresh["post_stats"]["probs"][0])
    torch.testing.assert_close(reset["post"].h[1], out["post"].h[1])


def test_imagination_action_state_reward_alignment(small_model, monkeypatch):
    wm = small_model
    actor = Actor(latent=wm.latent, net=wm.net_cfg)
    critic = ValueHead(latent=wm.latent, net=wm.net_cfg)
    start = wm.init_state(2)
    used_actions = []
    original = wm.imagine_step

    def step(state, policy, **kwargs):
        used_actions.append(kwargs["action"].argmax(-1))
        return original(state, policy, **kwargs)

    monkeypatch.setattr(wm, "imagine_step", step)
    calls = []
    hook = actor.register_forward_hook(lambda *args: calls.append(1))
    traj = imagine_trajectory_for_training(wm, actor, critic, start, 3)
    hook.remove()
    assert len(calls) == 3  # one policy call and action sample per transition
    torch.testing.assert_close(traj["action"], torch.stack(used_actions))
    torch.testing.assert_close(traj["h"][0], start.h)
    for t in range(2):
        expected_h = wm.rssm(traj["h"][t], traj["z"][t], torch.nn.functional.one_hot(traj["action"][t], 3).float())
        torch.testing.assert_close(traj["h"][t + 1], expected_h)
        logits = wm.reward_heads.main(traj["h"][t + 1], traj["z"][t + 1])
        torch.testing.assert_close(traj["reward"][t], wm.reward_head.readout(logits))


def test_actor_critic_uses_replay_and_isolates_gradients(small_model, sequence, monkeypatch):
    import importlib

    module = importlib.import_module("dreamerrl.training.core.actor_critic_update")
    wm = small_model
    wm.posterior.deterministic_latent_for_tests = True
    wm.prior.deterministic_latent_for_tests = True
    actor = Actor(latent=wm.latent, net=wm.net_cfg)
    critic = ValueHead(latent=wm.latent, net=wm.net_cfg)
    observed = observe_batch(wm, sequence)
    expected_h = torch.stack([p["h"] for p in observed], 1).reshape(8, -1)
    original = module.imagine_trajectory_for_training

    def imagine(*args):
        torch.testing.assert_close(args[3].h, expected_h)
        return original(*args)

    monkeypatch.setattr(module, "imagine_trajectory_for_training", imagine)
    actor_loss, critic_loss = actor_critic_update(wm, actor, critic, sequence, 3, 0.99, 0.95)
    actor_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert all(p.grad is None for p in critic.parameters())
    critic_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())
    assert all(p.grad is None for p in wm.parameters())


def test_world_loss_kl_weights_and_aux_switch(small_model, sequence):
    wm = small_model
    wm.cfg.disable_aux_losses = True
    m = world_model_training_step(wm, sequence, kl_scale=2)
    assert m.aux_losses == []
    expected = m.recon_loss + m.reward_loss + m.cont_loss + 2 * (1.0 * m.kl_dyn + 0.1 * m.kl_rep)
    torch.testing.assert_close(m.total_loss, expected)


def test_actor_critic_terminal_starts_have_zero_weight(small_model, sequence):
    wm = small_model
    sequence["is_terminal"].fill_(1)
    actor = Actor(latent=wm.latent, net=wm.net_cfg)
    critic = ValueHead(latent=wm.latent, net=wm.net_cfg)
    a, c = actor_critic_update(wm, actor, critic, sequence, 3, 0.99, 0.95)
    (a + c).backward()
    assert a == 0 and c == 0
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in actor.parameters())
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in critic.parameters())


def test_actor_critic_continuation_masks_later_losses(small_model, sequence, monkeypatch):
    import importlib

    module = importlib.import_module("dreamerrl.training.core.actor_critic_update")
    wm = small_model
    wm.posterior.deterministic_latent_for_tests = True
    actor = Actor(latent=wm.latent, net=wm.net_cfg)
    critic = ValueHead(latent=wm.latent, net=wm.net_cfg)
    later_reward = [1.0]

    def imagine(world_model, actor, critic, start, horizon, deterministic):
        N = start.h.shape[0]
        return dict(
            h=start.h.unsqueeze(0).repeat(3, 1, 1),
            z=start.z.unsqueeze(0).repeat(3, 1, 1, 1),
            reward=torch.tensor([1.0, later_reward[0], later_reward[0]]).unsqueeze(1).repeat(1, N),
            action=torch.zeros(3, N, dtype=torch.long),
            **{"continue": torch.zeros(3, N)},
            bootstrap_value=torch.zeros(N),
        )

    monkeypatch.setattr(module, "imagine_trajectory_for_training", imagine)
    first = actor_critic_update(wm, actor, critic, sequence, 3, 0.99, 0.95)
    later_reward[0] = 10000.0
    second = actor_critic_update(wm, actor, critic, sequence, 3, 0.99, 0.95)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)


def test_configured_kl_scale_is_applied_once(small_model, sequence):
    wm = small_model
    wm.cfg.disable_aux_losses = True
    wm.cfg.kl_cfg = replace(wm.cfg.kl_cfg, scale=3.0)
    m = world_model_training_step(wm, sequence, kl_scale=wm.cfg.kl_cfg.scale)
    expected = m.recon_loss + m.reward_loss + m.cont_loss + 3 * (1.0 * m.kl_dyn + 0.1 * m.kl_rep)
    torch.testing.assert_close(m.total_loss, expected)
