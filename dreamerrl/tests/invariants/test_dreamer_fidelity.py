"""Reference-derived probability, gradient, and EMA-critic contracts."""

import importlib

import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.categorical import mixed_probs
from dreamerrl.models.posterior import Posterior
from dreamerrl.models.prior import Prior
from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.core.actor_critic_update import actor_critic_update
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.transforms import symlog
from dreamerrl.utils.twohot import twohot_encode
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.mark.parametrize("kind", [Prior, Posterior])
def test_latent_samples_have_probability_gradients(kind):
    latent = LatentConfig(deter_size=8, num_classes=3, stoch_size=4)
    net = NetworkConfig(hidden_size=8, action_dim=3)
    model = kind(latent=latent, net=net)
    h, embed = torch.randn(8, 8), torch.randn(8, 8)
    coefficients = torch.tensor([-0.7, 0.2, 0.8, 1.3])
    gradients, samples = [], []
    for seed in [0, 1]:
        torch.manual_seed(seed)
        stats = model(h, embed) if kind is Posterior else model(h)
        z, probs = stats["z"], stats["probs"]
        assert ((z == 0) | (z == 1)).all()
        assert (z.sum(-1) == 1).all()
        assert (probs >= 0.01 / latent.stoch_size).all()
        torch.testing.assert_close(stats["logits"].softmax(-1), probs)
        actual = torch.autograd.grad((z * coefficients).sum(), model.fc_logits.weight, retain_graph=True)[0]
        expected = torch.autograd.grad((probs * coefficients).sum(), model.fc_logits.weight)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.abs().sum() > 0
        gradients.append(actual)
        samples.append(z.detach())
    # The Jacobian must not depend on noise in a relaxed Gumbel sample.
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    assert not torch.equal(samples[0], samples[1])


def test_uniform_mix_matches_formula_at_extreme_logits():
    logits = torch.tensor([[10000.0, -10000.0, 0.0]], dtype=torch.float16, requires_grad=True)
    probs = mixed_probs(logits)
    expected = 0.99 * logits.float().softmax(-1) + 0.01 / 3
    torch.testing.assert_close(probs, expected)
    assert torch.isfinite(probs.log()).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(1))
    with pytest.raises(ValueError, match="unimix"):
        mixed_probs(logits, -0.1)


def test_actor_uses_same_mixed_distribution_for_sampling_and_scoring():
    latent = LatentConfig(deter_size=8, num_classes=3, stoch_size=4)
    actor = Actor(latent=latent, net=NetworkConfig(hidden_size=8, action_dim=3))
    final = actor.net[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(torch.tensor([100.0, -100.0, 0.0]))
    logits = actor(torch.zeros(2, 8), torch.zeros(2, 3, 4))
    expected = torch.tensor([[0.99 + 0.01 / 3, 0.01 / 3, 0.01 / 3]]).repeat(2, 1)
    dist = torch.distributions.Categorical(logits=logits)
    torch.testing.assert_close(dist.probs, expected)
    torch.testing.assert_close(dist.log_prob(torch.ones(2, dtype=torch.long)), expected[:, 1].log())
    assert torch.isfinite(dist.entropy()).all()


@pytest.fixture
def trainer(tmp_path):
    instance = DreamerTrainer(minihack_config(0, tmp_path))
    instance.world.posterior.deterministic_latent_for_tests = True
    instance.world.prior.deterministic_latent_for_tests = True
    yield instance
    instance.env.venv.close()
    instance.tb.close()


def batch_for(trainer):
    return dict(
        obs=torch.randn(2, 3, trainer.env.obs_dim),
        action=torch.zeros(2, 3, dtype=torch.long),
        is_terminal=torch.zeros(2, 3),
    )


def test_critic_regularizes_to_detached_slow_scalar_prediction(trainer, monkeypatch):
    module = importlib.import_module("dreamerrl.training.core.actor_critic_update")
    original = module.imagine_trajectory_for_training
    trajectories = []

    def imagine(*args):
        result = original(*args)
        trajectories.append(result)
        return result

    monkeypatch.setattr(module, "imagine_trajectory_for_training", imagine)
    batch = batch_for(trainer)
    args = (trainer.world, trainer.actor, trainer.critic, batch, 3, 0.99, 0.95)
    _, base = actor_critic_update(
        *args, deterministic_imagination=True, slow_critic=trainer.slow_critic, slow_critic_scale=0.0
    )
    _, regularized = actor_critic_update(
        *args, deterministic_imagination=True, slow_critic=trainer.slow_critic, slow_critic_scale=1.0
    )
    traj = trajectories[-1]
    h, z = traj["h"].flatten(0, 1), traj["z"].flatten(0, 1)
    with torch.no_grad():
        target = twohot_encode(
            symlog(trainer.slow_critic.readout(trainer.slow_critic(h, z))), trainer.critic.bin_values
        )
        ce = -(target * trainer.critic(h, z).log_softmax(-1)).sum(-1).reshape(3, -1)
        discounts = 0.99 * traj["continue"]
        weights = torch.cumprod(torch.cat([torch.ones_like(discounts[:1]), discounts[:-1]]), 0)
        expected = (weights * ce).mean()
    torch.testing.assert_close(regularized - base, expected)
    regularized.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.critic.parameters())
    for model in [trainer.slow_critic, trainer.actor, trainer.world]:
        assert all(p.grad is None for p in model.parameters())


def test_slow_critic_updates_once_after_online_optimizer(trainer, monkeypatch):
    before = [p.detach().clone() for p in trainer.slow_critic.parameters()]
    events = []
    step, ema = trainer.critic_opt.step, trainer.update_slow_critic

    def tracked_step(*args, **kwargs):
        result = step(*args, **kwargs)
        events.append("optimizer")
        return result

    def tracked_ema():
        events.append("ema")
        ema()

    monkeypatch.setattr(trainer.critic_opt, "step", tracked_step)
    monkeypatch.setattr(trainer, "update_slow_critic", tracked_ema)
    trainer.update_actor_critic(batch_for(trainer), 0)
    assert events == ["optimizer", "ema"]
    for old, online, slow in zip(before, trainer.critic.parameters(), trainer.slow_critic.parameters()):
        torch.testing.assert_close(slow, 0.98 * old + 0.02 * online)
        assert not slow.requires_grad
    assert any(not torch.equal(old, slow) for old, slow in zip(before, trainer.slow_critic.parameters()))


def test_v1_checkpoint_initializes_missing_slow_critic(trainer, tmp_path):
    path = tmp_path / "old.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["version"] = 1
    del payload["slow_critic"]
    torch.save(payload, path)
    with torch.no_grad():
        for param in trainer.slow_critic.parameters():
            param.add_(1)
    with pytest.warns(RuntimeWarning, match="no EMA critic"):
        trainer.load_checkpoint(path)
    for online, slow in zip(trainer.critic.parameters(), trainer.slow_critic.parameters()):
        torch.testing.assert_close(online, slow, rtol=0, atol=0)
        assert not slow.requires_grad
