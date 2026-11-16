"""Replay context warms recurrence without becoming supervision or imagination starts."""

import gymnasium as gym
import numpy as np
import torch

from dreamerrl.models.world_model import WorldModel
from dreamerrl.replay_buffer.replay_buffer import ReplayBuffer
from dreamerrl.training.core.world_model_update import observe_batch, world_model_training_step
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig, TrainingConfig


def model():
    cfg = DreamerConfig()
    cfg.world.disable_aux_losses = True
    wm = WorldModel(
        obs_space=gym.spaces.Box(-100, 100, (2,), np.float32),
        dreamer_cfg=cfg,
        latent=LatentConfig(deter_size=8, num_classes=2, stoch_size=3),
        net=NetworkConfig(hidden_size=12, action_dim=3, value_bins=21, bin_min=-5, bin_max=5),
    )
    wm.posterior.deterministic_latent_for_tests = True
    wm.prior.deterministic_latent_for_tests = True
    return wm


def replay(burn_in):
    buf = ReplayBuffer(TrainingConfig(seq_len=3, burn_in=burn_in), 2, 3, torch.device("cpu"))
    for t in range(9):
        buf.add(
            torch.tensor([[float(t), float(t + 1)]]),
            torch.tensor([t % 3]),
            torch.tensor([float(t)]),
            torch.tensor([t == 8]),
            torch.tensor([t == 8]),
            initial_obs=torch.tensor([[-1.0, 0.0]]),
        )
    return buf


def test_context_alignment_and_episode_start_padding():
    buf = replay(16)
    batch = buf.sample(32, seed=0)
    starts = batch["obs"][:, 0, 0].long() + 1
    assert (starts == 0).any() and (starts > 0).any()
    for row, start in enumerate(starts.tolist()):
        valid = batch["burn_in_valid"][row]
        assert valid.sum() == start
        for key in ("obs", "action", "is_first"):
            torch.testing.assert_close(batch[f"burn_in_{key}"][row][valid], buf.episodes[0][key][:start])
    wm = model()
    full = {k: v.unsqueeze(0) for k, v in buf.episodes[0].items()}
    expected = observe_batch(wm, full)
    actual = observe_batch(wm, batch)
    for row, start in enumerate(starts.tolist()):
        for t, post in enumerate(actual):
            for key in ("h", "z", "probs"):
                torch.testing.assert_close(post[key][row], expected[start + t][key][0])


def test_burn_in_detaches_prefix_but_training_window_has_gradients():
    wm = model()
    batch = replay(16).sample(8, seed=0)
    batch["burn_in_obs"].requires_grad_()
    batch["obs"].requires_grad_()
    metrics = world_model_training_step(wm, batch)
    metrics.total_loss.backward()
    assert batch["burn_in_obs"].grad is None
    assert batch["obs"].grad is not None and batch["obs"].grad.abs().sum() > 0
    recurrent_grad = wm.rssm.gru.weight_hh.grad
    assert recurrent_grad is not None
    assert recurrent_grad.abs().sum() > 0
    assert len(observe_batch(wm, batch)) == 3


def test_bounded_context_and_disabled_burn_in():
    batch = replay(2).sample(32, seed=0)
    starts = batch["obs"][:, 0, 0].long() + 1
    for row, start in enumerate(starts.tolist()):
        expected = replay(2).episodes[0]["obs"][max(0, start - 2) : start]
        torch.testing.assert_close(batch["burn_in_obs"][row][batch["burn_in_valid"][row]], expected)
    wm = model()
    disabled = replay(0).sample(4, seed=0)
    assert disabled["burn_in_obs"].shape[1] == 0
    plain = {k: v for k, v in disabled.items() if not k.startswith("burn_in_")}
    for a, b in zip(observe_batch(wm, disabled), observe_batch(wm, plain)):
        torch.testing.assert_close(a["h"], b["h"])


def test_replay_checkpoint_preserves_context_configuration_and_sampling():
    source = replay(2)
    state = source.state_dict()
    restored = replay(0)
    restored.load_state_dict(state)
    assert restored.burn_in == restored.cfg.burn_in == 2
    for key, value in source.sample(4).items():
        # Restore RNG again so each comparison uses the same next draw.
        restored.load_state_dict(state)
        torch.testing.assert_close(value, restored.sample(4)[key])


def test_actor_imagination_starts_exclude_burn_in(monkeypatch):
    import importlib

    from dreamerrl.models.actor import Actor
    from dreamerrl.models.value_head import ValueHead

    module = importlib.import_module("dreamerrl.training.core.actor_critic_update")
    wm = model()
    batch = replay(16).sample(8, seed=0)
    posts = observe_batch(wm, batch)
    expected = torch.stack([p["h"] for p in posts], dim=1).reshape(24, -1)
    original = module.imagine_trajectory_for_training
    calls = []

    def imagine(*args, **kwargs):
        torch.testing.assert_close(args[3].h, expected)
        calls.append(args[3].h.shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "imagine_trajectory_for_training", imagine)
    actor = Actor(latent=wm.latent, net=wm.net_cfg)
    critic = ValueHead(latent=wm.latent, net=wm.net_cfg)
    losses = module.actor_critic_update(wm, actor, critic, batch, 3, 0.99, 0.95)
    assert calls == [24]
    assert all(torch.isfinite(loss) for loss in losses)


def test_legacy_replay_checkpoint_keeps_burn_in_disabled():
    state = replay(0).state_dict()
    del state["burn_in"]
    restored = replay(16)
    restored.load_state_dict(state)
    assert restored.burn_in == restored.cfg.burn_in == 0


def test_padding_cannot_change_reconstructed_state():
    wm = model()
    batch = replay(16).sample(32, seed=0)
    expected = observe_batch(wm, batch)
    batch["burn_in_obs"][~batch["burn_in_valid"]] = 99
    batch["burn_in_action"][~batch["burn_in_valid"]] = 2
    actual = observe_batch(wm, batch)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left["h"], right["h"])
        torch.testing.assert_close(left["z"], right["z"])
