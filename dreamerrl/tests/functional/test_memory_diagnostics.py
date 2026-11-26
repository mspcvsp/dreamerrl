import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from dreamerrl.eval.memory.dreamer import DreamerAdapter, categorical_state
from dreamerrl.eval.memory.interfaces import Episode, paired_episodes
from dreamerrl.eval.memory.replay import replay_coverage
from dreamerrl.eval.memory.statistics import binary_metrics, binary_probe, grouped_split, paired_categorical
from dreamerrl.eval.memory.trajectory import filter_trajectory
from dreamerrl.models.actor import Actor
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.core.world_model_update import observe_batch
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


def test_grouped_probe_control_and_validation():
    y = np.tile([False, True], 20)
    ids = np.repeat(np.arange(20), 2)
    split, groups = grouped_split(ids, 42)
    assert len(set(groups["fit"]) & set(groups["test"])) == 0
    nuisance = np.empty(len(y))
    for mask in split.values():
        nuisance[mask] = np.tile(np.repeat([-1, 1], 2), mask.sum() // 4)
    xor = np.stack((np.where(y, 1, -1) * nuisance, nuisance), 1)[:, None, :]
    nonlinear = binary_probe(xor, y, split, nonlinear=True)
    test = split["test"]
    np.testing.assert_array_equal(nonlinear["scores"][test, 0] >= 0, y[test])
    changed = y.copy()
    changed[test] = ~changed[test]
    np.testing.assert_array_equal(nonlinear["scores"], binary_probe(xor, changed, split, nonlinear=True)["scores"])
    linear = np.where(y, 1, -1)[:, None, None]
    assert binary_metrics(binary_probe(linear, y, split)["scores"][test], y[test], ids[test], 1)["balanced_accuracy"] == 1
    flat = binary_metrics(np.zeros((test.sum(), 1)), y[test], ids[test], 1)
    assert flat["balanced_accuracy"] == flat["average_precision"] == .5


def test_hard_collapse_and_seeded_sampling():
    q = torch.tensor([[[.901, .099]], [[.899, .101]]])
    stats = paired_categorical(q[:1], q[1:])
    assert stats["argmax_pair_equal_fraction"] == 1
    assert stats["js_nats_by_factor"][0] > 0
    assert .5 < stats["mean_single_draw_bayes_accuracy_upper_bound"] < .51
    probs = q.expand(20000, -1, -1, -1)
    a, b = [categorical_state(probs, "sample", rng=torch.Generator().manual_seed(29)) for _ in range(2)]
    assert torch.equal(a, b)
    torch.testing.assert_close(a.mean(0), q, atol=.01, rtol=0)


def test_matched_trajectory_and_incoming_action_in_history_control():
    class Model:
        action_dim = 2

        def initial_state(self, batch):
            return torch.zeros(batch, 1)

        def filter_step(self, state, obs, incoming, first, **_):
            state = state + obs + incoming[:, 1:]
            return state, {"h": state, "incoming": incoming, "first": first}

    frames = torch.tensor([[2.], [0.], [0.]])
    actions = torch.tensor([1, 0, 0])
    zero_reward = torch.zeros(3)
    a = Episode("left", 1, frames, actions, zero_reward, True, False)
    b = Episode("right", 1, frames + torch.tensor([[3.], [0.], [0.]]), actions, zero_reward, True, False)
    paired_episodes(a, b, 0)
    with pytest.raises(ValueError, match="outside"):
        paired_episodes(a, b, 1)
    obs = torch.stack((a.observations, b.observations))
    decisions = torch.stack((actions, actions))
    _, full = filter_trajectory(Model(), obs, decisions)
    _, fixed = filter_trajectory(Model(), obs, decisions, history=1)
    torch.testing.assert_close(full["incoming"], fixed["incoming"])
    assert torch.equal(full["first"], fixed["first"])
    assert full["h"][:, -1, 0].tolist() == [3., 6.]
    assert fixed["h"][:, -1, 0].tolist() == [0., 0.]


def dreamer_components():
    cfg = DreamerConfig()
    latent = LatentConfig(deter_size=8, num_classes=3, stoch_size=4)
    net = NetworkConfig(hidden_size=16, action_dim=2, unimix=.01, value_bins=41)
    world = WorldModel(obs_space=Box(-1, 1, (4,), np.float32), dreamer_cfg=cfg, latent=latent, net=net)
    return cfg, world, Actor(latent=latent, net=net)


def test_adapter_matches_production_filter_and_imagination():
    cfg, world, actor = dreamer_components()
    frozen = DreamerAdapter(cfg, world, actor, None)
    obs, action = torch.zeros(2, 4), torch.tensor([[1., 0.], [0., 1.]])
    first = torch.ones(2, dtype=torch.bool)
    carry = world.init_state(2)
    h, z = carry.h.clone(), carry.z.clone()
    carry.h.fill_(3)
    carry.z.fill_(1)
    legacy = world.observe_step(carry, obs, action, is_first=first)
    new, stages = frozen.filter_step(carry, obs, action, first)
    torch.testing.assert_close(legacy["post"].h, new.h, atol=0, rtol=0)
    torch.testing.assert_close(legacy["post"].z, new.z, atol=0, rtol=0)
    torch.testing.assert_close(legacy["post_stats"]["probs"], stages["posterior"], atol=0, rtol=0)
    torch.testing.assert_close(stages["pre_unimix"] * .99 + .01 / 4, stages["posterior"])
    assert not torch.equal(stages["raw_logits"], legacy["post_stats"]["logits"])
    assert torch.equal(carry.h, h + 3) and torch.equal(carry.z, z + 1)
    imagined = frozen.imagine_step(new, action)
    original = world.imagine_step(legacy["post"], actor, action=action, deterministic_imagination=True)
    torch.testing.assert_close(imagined.h, original.h, atol=0, rtol=0)
    torch.testing.assert_close(imagined.z, original.z, atol=0, rtol=0)


def test_replay_structure_and_gradient_connectivity():
    _, world, _ = dreamer_components()
    ep = {"obs": torch.zeros(10, 4), "reward": torch.zeros(10),
          "is_first": torch.tensor([True] + [False] * 9)}
    ep["reward"][-1] = 1
    result = replay_coverage(dict(seq_len=8, burn_in=8, episodes=[ep]))
    assert result["eligible_starts"] == {10: 3}
    assert result["delayed_endpoint_supervised"] == {10: 1}
    assert result["reset_available_only_through_detached_burn_in"] == {10: 1}
    assert not result["reset_and_endpoint_jointly_supervised"]
    current = torch.full((1, 2, 4), .1, requires_grad=True)
    burn = torch.full((1, 1, 4), .2, requires_grad=True)
    batch = dict(obs=current, action=torch.zeros(1, 2, dtype=torch.long),
        is_first=torch.zeros(1, 2, dtype=torch.bool), burn_in_obs=burn,
        burn_in_action=torch.zeros(1, 1, dtype=torch.long), burn_in_is_first=torch.ones(1, 1, dtype=torch.bool),
        burn_in_valid=torch.ones(1, 1, dtype=torch.bool))
    h = observe_batch(world, batch)[-1]["h"]
    grad_current, grad_burn = torch.autograd.grad(h.square().sum(), (current, burn), allow_unused=True)
    assert grad_burn is None and grad_current[0, 0].abs().max() > 0


def test_cage2_truth_fixture_ignores_inactive_and_respects_order():
    from environments.cage2.cage2_memory_pilot import active_host_labels
    def session(host, active):
        return SimpleNamespace(host=host, active=active)

    state = SimpleNamespace(hosts={"one": None, "two": None, "three": None},
                            sessions={"Red": {1: session("one", True), 2: session("two", False)}})
    np.testing.assert_array_equal(active_host_labels(state, ("two", "one", "three")), [False, True, False])
    with pytest.raises(ValueError, match="disagree"):
        active_host_labels(state, ("unrecognized",))


@pytest.mark.minihack_invariants
def test_minihack_existing_filter_parity(tmp_path):
    pytest.importorskip("minihack")
    import gymnasium as gym

    from dreamerrl.eval.location_probe import load_frozen_agent

    cfg = DreamerConfig()
    cfg.env.env_id = "MiniHack-Room-5x5-v0"
    cfg.world.deter_size, cfg.world.num_classes, cfg.world.stoch_size = 8, 3, 4
    cfg.world.hidden_size, cfg.ac.actor_hidden = 16, 16
    with gym.make(cfg.env.env_id) as env:
        obs, _ = env.reset(seed=4)
        first_obs = torch.tensor(obs["glyphs"].reshape(1, -1), dtype=torch.float32)
        action_dim = env.action_space.n
        successor, *_ = env.step(0)
        second_obs = torch.tensor(successor["glyphs"].reshape(1, -1), dtype=torch.float32)
    space = Box(0, 10000, (first_obs.shape[-1],), np.float32)
    latent = LatentConfig(deter_size=8, num_classes=3, stoch_size=4)
    net = NetworkConfig(hidden_size=16, action_dim=action_dim, value_bins=cfg.world.value_bins, unimix=cfg.world.unimix)
    world = WorldModel(obs_space=space, dreamer_cfg=cfg, latent=latent, net=net)
    actor = Actor(latent=latent, net=NetworkConfig(hidden_size=16, action_dim=action_dim, unimix=cfg.ac.unimix))
    payload = dict(version=4, config=cfg, completed_updates=0, world=world.state_dict(), actor=actor.state_dict())
    path = Path(tmp_path) / "synthetic_policy.pt"
    torch.save(payload, path)
    previous = load_frozen_agent(path)
    adapter = DreamerAdapter.from_payload(payload, space, action_dim, load_critic=False)
    orig, new = previous.world.init_state(1), adapter.initial_state(1)
    for step, obs in enumerate((first_obs, second_obs)):
        action = torch.zeros(1, action_dim)
        if step:
            action[0, 0] = 1
        first = torch.tensor([not step])
        orig = previous.world.filter_step(orig, previous.world.encoder(obs), action, first)
        new, actual = adapter.filter_step(new, obs, action, first)
        torch.testing.assert_close(orig.h, new.h, atol=0, rtol=0)
        torch.testing.assert_close(orig.z, new.z, atol=0, rtol=0)
        torch.testing.assert_close(orig.post_stats["probs"], actual["posterior"], atol=0, rtol=0)


def test_accepted_tmaze_artifact_parity():
    root = os.environ.get("DREAMER_MEMORY_ARTIFACT_ROOT")
    if not root:
        pytest.skip("Set DREAMER_MEMORY_ARTIFACT_ROOT for frozen artifact parity")
    from dreamerrl.eval.memory.dreamer import CheckpointSpec, verified_payload
    from environments.minihack.tmaze_intervention import TMazeArchive

    base = Path(root) / "runs/minihack_tmaze_representation_diagnostics_20261126"
    stored = json.loads((base / "checkpoints/seed19_4000.json").read_text())
    path = Path(stored["path"])
    payload = verified_payload(CheckpointSpec(path, stored["sha256"], 19, 4000, 288000))
    model = DreamerAdapter.from_payload(payload, Box(0, 10000, (56,), np.float32), 2)
    del payload
    levels = sorted(sum(stored["split"].values(), []))
    observer = TMazeArchive(path.parent / "heldout_trajectories.json")
    observations, actions = observer.matched_arrays(levels)
    _, stages = filter_trajectory(model, observations["paired"], actions)
    split = {k: np.repeat(np.isin(levels, v), 2) for k, v in stored["split"].items()}
    y = np.tile([0, 1], len(levels))
    test = split["test"]
    for name, old_name in (("encoder", "encoder"), ("raw_logits", "logits"), ("posterior", "probabilities"),
                           ("z", "z"), ("h", "h"), ("hz", "hz")):
        for lag in (0, 8):
            feature = stages[name][:, lag].reshape(len(y), 1, -1)
            for nonlinear in (False, True):
                probe = binary_probe(feature.numpy(), y, split, nonlinear=nonlinear)
                expected = stored["conditions"]["mode"]["probes"][f"{old_name}/{lag}"]["rbf" if nonlinear else "linear"]
                assert float(np.mean((probe["scores"][test, 0] >= 0) == y[test])) == expected["accuracy"]["mean"]
                assert probe["validation_accuracy"] == expected["validation_accuracy"]
                assert probe["penalty"] == expected["penalty"]


def test_cage2_archive_numerical_parity():
    root = os.environ.get("DREAMER_MEMORY_ARTIFACT_ROOT")
    if not root:
        pytest.skip("Set DREAMER_MEMORY_ARTIFACT_ROOT for frozen artifact parity")
    from environments.cage2.cage2_memory_pilot import archive_parity

    folder = Path(root) / "runs/cage2_latent_analysis_host_pilot_20261124"
    results = archive_parity(folder)
    assert results["episodes"] == 360 and results["observations"] == 10800
