"""Small mathematical and leakage controls for offline clustering."""

import numpy as np
from latent_clustering import ari, db_index, distances, kmeans, persistence, prepare, silhouette


def test_separated_clusters_and_label_permutation():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(-5, 0.1, (30, 3)), rng.normal(5, 0.1, (30, 3))])
    truth = np.repeat([0, 1], 30)
    labels = distances(x, kmeans(x, 2, 11)).argmin(1)
    assert ari(truth, labels) == 1
    assert ari(labels, 1 - labels) == 1
    quality, db = silhouette(x, labels), db_index(x, labels)
    assert quality is not None and quality > 0.95
    assert db is not None and db < 0.1


def test_preprocessing_ignores_test_values():
    rng = np.random.default_rng(4)
    h = rng.normal(size=(120, 6))
    timestep = np.tile(np.arange(30), 4)
    fit = np.arange(120) < 60
    a, params_a = prepare(h, timestep, fit, 0.95)
    h[~fit] += 1000
    b, params_b = prepare(h, timestep, fit, 0.95)
    for key in params_a:
        np.testing.assert_allclose(params_a[key], params_b[key])
    for key in a:
        np.testing.assert_allclose(a[key][fit], b[key][fit])


def test_persistence_respects_episode_boundaries():
    result = persistence(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert result["stay_probability"] == 1
    assert result["mean_streak"] == 2
    assert result["adjacent_pairs"] == 2
