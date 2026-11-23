"""Frozen probes use level-held-out labels without leaking them into inputs."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from dreamerrl.eval.location_probe import (
    LocationData,
    collect_locations,
    fit_probe,
    load_frozen_agent,
    position_scores,
    validate_seed_split,
)
from dreamerrl.models.actor import Actor
from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


class ProbeEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "glyphs": gym.spaces.Box(0, 10, (2, 3), np.int32),
            "blstats": gym.spaces.Box(0, 100, (2,), np.int32),
        }
    )
    action_space = gym.spaces.Discrete(3)

    def seed(self, **kwargs):
        self.level_seed = kwargs["core"]

    def observation(self):
        # Position is deliberately absent from glyph inputs in this sentinel.
        return {"glyphs": np.ones((2, 3), np.int32), "blstats": np.array([self.level_seed % 3, self.t % 2], np.int32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.level_seed = seed or 0
        self.t = 0
        self.actions = []
        return self.observation(), {}

    def step(self, action):
        self.actions.append(action)
        self.t += 1
        return self.observation(), 0.0, self.t == 6, False, {}


@pytest.fixture
def frozen_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("dreamerrl.eval.location_probe.gym.make", lambda *args, **kwargs: ProbeEnv())
    cfg = DreamerConfig()
    cfg.env.env_id = "MiniHack-ProbeSentinel-v0"
    cfg.world.disable_aux_losses = True
    cfg.world.deter_size, cfg.world.num_classes, cfg.world.stoch_size = 4, 2, 3
    cfg.world.hidden_size = cfg.ac.actor_hidden = 8
    cfg.world.value_bins = 7
    latent = LatentConfig(deter_size=4, num_classes=2, stoch_size=3)
    net = NetworkConfig(hidden_size=8, action_dim=3, value_bins=7)
    world = WorldModel(obs_space=gym.spaces.Box(0, 10, (6,), np.float32), dreamer_cfg=cfg, latent=latent, net=net)
    actor = Actor(latent=latent, net=NetworkConfig(hidden_size=8, action_dim=3))
    path = tmp_path / "probe.pt"
    torch.save(
        dict(version=4, config=cfg, world=world.state_dict(), actor=actor.state_dict(), completed_updates=0), path
    )
    return load_frozen_agent(path)


def test_collection_alignment_duplicate_cap_and_frozen_parameters(frozen_agent, monkeypatch):
    agent = frozen_agent
    before = {name: value.clone() for name, value in agent.world.state_dict().items()}
    actions, firsts = [], []
    original = agent.world.filter_step

    def step(state, embed, action, first):
        actions.append(action.clone())
        firsts.append(bool(first[0]))
        return original(state, embed, action, first)

    monkeypatch.setattr(agent.world, "filter_step", step)
    rng = torch.get_rng_state().clone()
    data = collect_locations(agent, [10, 11], max_steps=6, max_per_cell=1, initial=agent)
    assert torch.equal(torch.get_rng_state(), rng)
    assert data.positions.tolist() == [[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
    assert data.level_seeds.tolist() == [10, 10, 11, 11]
    assert data.grid_shape == (2, 3)
    assert data.features["hz"].shape == (4, 10)
    torch.testing.assert_close(data.features["hz"], data.features["initial_hz"])
    # Same glyph observations always produce the same observation-only embedding,
    # despite different position labels in blstats.
    torch.testing.assert_close(data.features["encoder"], data.features["encoder"][:1].expand(4, -1))
    assert firsts.count(True) == 4  # trained + initial pass for two seeds
    assert actions[0].sum() == 0
    assert all(not value.requires_grad for value in data.features.values())
    for name, value in agent.world.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in agent.world.parameters())


def test_linear_probe_recovers_known_location_and_beats_shuffled_labels():
    generator = torch.Generator().manual_seed(7)
    positions = torch.stack(
        [torch.randint(12, (640,), generator=generator), torch.randint(8, (640,), generator=generator)], -1
    ).float()
    features = torch.cat([positions / torch.tensor([11.0, 7.0]), torch.randn(640, 4, generator=generator)], -1)
    probe = fit_probe(features[:512], positions[:512], (8, 12), epochs=200)
    test = LocationData({"hz": features[512:]}, positions[512:], torch.full((128,), 99), (8, 12))
    score = position_scores(probe.predict(features[512:]), test)
    permutation = torch.randperm(512, generator=generator)
    shuffled = fit_probe(features[:512], positions[:512][permutation], (8, 12), epochs=200)
    shuffled_score = position_scores(shuffled.predict(features[512:]), test)
    assert score["mean_manhattan_cells"] < 0.15
    assert score["exact_cell_accuracy"] > 0.95
    assert score["mean_manhattan_cells"] < shuffled_score["mean_manhattan_cells"] * 0.2
    torch.testing.assert_close(probe.mean, features[:512].mean(0))
    # Small latent amplitudes must remain decodable after train-only scaling.
    scaled = fit_probe(features[:512] * 1e-4, positions[:512], (8, 12), epochs=200)
    torch.testing.assert_close(
        scaled.predict(features[512:] * 1e-4), probe.predict(features[512:]), atol=0.01, rtol=0.01
    )


@pytest.mark.parametrize("train,test", [([], [2]), ([1], []), ([1, 2], [2, 3]), ([1, 1], [2])])
def test_probe_splits_reject_empty_overlapping_or_duplicate_seeds(train, test):
    with pytest.raises(ValueError, match="seeds"):
        validate_seed_split(train, test)


def test_position_scores_handle_constant_targets_and_out_of_bounds_predictions():
    data = LocationData({}, torch.zeros(2, 2), torch.tensor([1, 2]), (2, 3))
    scores = position_scores(torch.tensor([[-2.0, 0.0], [0.0, 1.0]]), data)
    assert scores["r2"] is None
    assert scores["mean_manhattan_cells"] == 1.5
    assert scores["exact_cell_accuracy"] == 0
    assert scores["within_one_cell_accuracy"] == 0.5


def test_initial_control_requires_matching_world_config(frozen_agent):
    from copy import deepcopy

    initial = deepcopy(frozen_agent)
    initial.cfg.world.hidden_size += 1
    with pytest.raises(ValueError, match="must match"):
        collect_locations(frozen_agent, [1], initial=initial)


def test_initial_control_does_not_change_sampled_features(frozen_agent):
    from copy import deepcopy

    frozen_agent.world.posterior.deterministic_latent_for_tests = False
    initial = deepcopy(frozen_agent)
    without = collect_locations(frozen_agent, [10], max_steps=6)
    with_control = collect_locations(frozen_agent, [10], max_steps=6, initial=initial)
    for name in without.features:
        torch.testing.assert_close(without.features[name], with_control.features[name], rtol=0, atol=0)
    assert with_control.features["posterior_probs"].shape == with_control.features["z"].shape
