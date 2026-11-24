"""Episode leakage, ridge standardization and causal feature/target alignment checks."""

from types import SimpleNamespace

import latent_analysis
import numpy as np
import torch
from latent_probe import episode_split, metrics, paired_episode_interval, ridge, spectral_rank

from dreamerrl.utils.types import EnvironmentConfig


def test_episode_split_never_shares_temporal_rows():
    episode = np.repeat(np.arange(20), 30)
    masks, groups = episode_split(episode, 123)
    assert not (set(groups["fit"]) & set(groups["test"]))
    assert not (set(groups["validation"]) & set(groups["test"]))
    assert np.all(sum(mask.astype(int) for mask in masks.values()) == 1)
    assert sum(masks["fit"]) == 12 * 30 and sum(masks["test"]) == 4 * 30


def test_ridge_learns_linear_target_and_normalizes_fit_only():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 4))
    target = x @ np.array([2.0, -3.0, 1.0, 0.0]) + 5
    split, _ = episode_split(np.repeat(np.arange(20), 30), 123)
    model = ridge(x, target, split, [1e-4, 0.1, 10.0])
    score = metrics(target[split["test"]], model["predictions"][split["test"]])["r2"]
    assert score is not None and score > 0.999
    changed = x.copy()
    changed[split["test"]] += 100000
    shifted = ridge(changed, target, split, [1e-4, 0.1, 10.0])
    np.testing.assert_array_equal(model["mean"], shifted["mean"])
    np.testing.assert_array_equal(model["scale"], shifted["scale"])
    assert shifted["alpha"] == model["alpha"]


def test_rank_and_group_bootstrap_are_episode_aware():
    x = np.linspace(-1, 1, 120)
    assert spectral_rank(np.ones((120, 4)))["effective_rank"] == 0
    assert abs(spectral_rank(np.column_stack([x, 2 * x, 3 * x]))["effective_rank"] - 1) < 1e-6
    actual = np.repeat(np.arange(4), 30).astype(float)
    interval = paired_episode_interval(actual, actual, actual + 1, np.repeat(np.arange(4), 30), 1, 200)
    confidence = interval["delta_r2_ci95"]
    assert interval["episodes"] == 4 and confidence is not None and confidence[0] > 0
    assert interval["delta_mae_ci95"] == [-1.0, -1.0]


def test_collection_features_precede_action_and_exclude_terminal(tmp_path, monkeypatch):
    class Env:
        def __init__(self, *args):
            self.journal = []
            self.step_index = 0

        def reset(self, seed):
            return {"state": torch.zeros(1, 52)}

        def step(self, action):
            self.step_index += 1
            self.journal.append({"reward": float(-self.step_index)})
            return dict(
                state=torch.full((1, 52), float(self.step_index)),
                is_last=torch.tensor([self.step_index == 3]),
                is_terminal=torch.tensor([False]),
            )

        def close(self):
            pass

    class World:
        def init_state(self, batch):
            return None

        def observe_step(self, state, obs, action, is_first):
            step = int(obs[0, 0])
            assert bool(is_first[0]) == (step == 0)
            assert bool(action.any()) == (step > 0)
            posterior = SimpleNamespace(
                h=torch.full((1, 200), float(step)), z=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
            )
            stats = {"probs": torch.full((1, 2, 3), 1 / 3)}
            return dict(post=posterior, post_stats=stats, prior_stats=stats)

    trainer = SimpleNamespace(
        cfg=SimpleNamespace(env=EnvironmentConfig()),
        device=torch.device("cpu"),
        world=World(),
        action_dim=2,
        actor=lambda h, z: torch.tensor([[0.0, 1.0]]),
    )
    config = dict(episodes=1, simulator_seed_start=100, policy_seed_offset=20, horizon=3)
    monkeypatch.setattr(latent_analysis, "Cage2BlueEnv", Env)
    data = latent_analysis.collect(trainer, config, tmp_path)
    np.testing.assert_array_equal(data["observation"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(data["h"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(data["target"], [-6, -5, -3])
    np.testing.assert_array_equal(data["reward"], [-1, -2, -3])
    assert data["timestep"].tolist() == [0, 1, 2]


def test_full_probe_pipeline_produces_quantitative_report_and_plots(tmp_path):
    rng = np.random.default_rng(3)
    episodes = np.repeat(np.arange(10), 30)
    step = np.tile(np.arange(30), 10)
    h = rng.normal(size=(300, 4))
    z = np.eye(3)[rng.integers(3, size=(300, 2))]
    data = dict(
        h=h,
        z=z.astype(np.uint8),
        posterior_probs=0.97 * z + 0.01,
        prior_probs=np.full_like(z, 1 / 3),
        observation=h[:, :2],
        timestep=step,
        episode=episodes,
        target=-(30 - step) + h[:, 0],
    )
    config = dict(split_seed=1, ridge_alphas=[0.01, 1.0], bootstrap_seed=1, bootstrap_draws=100, pca_trajectory_count=2)
    rows = latent_analysis.analyze(data, config, tmp_path)
    assert len(rows) == 12
    assert all(row["test_samples"] == 60 and row["fit_samples"] == 180 for row in rows)
    for name in (
        "linear_probe_comparison.csv",
        "final_report.txt",
        "predicted_vs_actual.png",
        "pca_trajectories.png",
        "latent_diagnostics.json",
        "incremental_information.json",
    ):
        assert (tmp_path / name).stat().st_size > 0
